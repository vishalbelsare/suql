"""Tests for `answer()` comparisons outside the WHERE clause.

`WHERE answer(notes, q) = 'Yes'` is compiled: the comparison goes to the
verification prompt, which asks the LLM whether 'Yes' is a correct answer to
`q` for that document. The same comparison in the SELECT list was not compiled
at all - Postgres ran the raw plpython3u UDF, whose prompt is open-ended
("Answer a question based on the following text"), and then string-compared its
free-form prose against the literal. `Yes. The Zaporizhzhia Nuclear Power
Station is part of energy infrastructure.` != `'Yes'`, so
`CASE WHEN answer(...) = 'Yes' THEN 1 ELSE 0 END` returned 0 for rows the model
had answered correctly, silently under-counting aggregates. Because
`prompt_continuation` forces `temperature=1` for the gpt-5 family, the prose
also varied run to run.

The compiler now recognises `answer(...) <op> <literal>` in the projection,
HAVING, GROUP BY and ORDER BY, verifies it per row through the same path a WHERE
predicate takes, and materializes the result as a synthetic boolean column.

Unit tests check the AST-level helpers in isolation.

Integration tests run the full pipeline against the live Postgres `acled` DB,
with only the LLM verification call stubbed out - so the structural SQL, the
temp table, the rewritten query and the documents that would be handed to the
LLM are all real.
"""

import os
import sys
import threading
import time
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pglast import parse_sql
from pglast.stream import RawStream

import suql.sql_free_text_support.execute_free_text_sql as suql_module
from suql.sql_free_text_support.execute_free_text_sql import (
    _INTERNAL_PRED_COLUMN_PREFIX,
    _contains_aggregate,
    _expand_star_targets,
    _extract_projection_predicates,
    _if_limit_pushdown_safe,
    _if_projection_predicate,
    _parallel_map,
    _projection_predicate_alias,
    _resolve_max_workers,
    suql_execute,
)

QUESTION = "Is this an attack on energy infrastructure?"

# Hand-labelled rows from the acled `events` table: three attacks on energy
# infrastructure and three that are not. The stub below accepts on the keyword
# "power station" / "pipeline", matching the first three.
POSITIVE_IDS = ["UKR73236", "UKR69217", "UKR88237"]
NEGATIVE_IDS = ["UKR127789", "UKR73096"]
ALL_IDS = POSITIVE_IDS + NEGATIVE_IDS
ID_LITERALS = ", ".join("'{}'".format(i) for i in ALL_IDS)


def _stmt(sql):
    return parse_sql(sql)[0].stmt


def _first_target_expr(sql):
    return _stmt(sql).targetList[0].val


# ---------- unit checks -------------------------------------------------------

def check_detects_bare_boolean_projection():
    expr = _first_target_expr(
        "SELECT answer(notes, '{}') = 'Yes' FROM events".format(QUESTION)
    )
    assert _if_projection_predicate(expr)


def check_detects_case_when_and_aggregate_wrappers():
    for sql in (
        "SELECT CASE WHEN answer(notes, 'q') = 'Yes' THEN 1 ELSE 0 END FROM events",
        "SELECT SUM(CASE WHEN answer(notes, 'q') = 'Yes' THEN 1 ELSE 0 END) FROM events",
        "SELECT COUNT(*) FILTER (WHERE answer(notes, 'q') = 'Yes') FROM events",
        "SELECT a FROM events GROUP BY a HAVING bool_or(answer(notes, 'q') = 'Yes')",
        "SELECT a FROM events ORDER BY answer(notes, 'q') = 'Yes' DESC",
    ):
        assert len(_extract_projection_predicates(_stmt(sql))) == 1, sql


def check_ignores_non_predicate_uses():
    """A projection that wants the model's prose, or a comparison the
    verification prompt cannot represent, must be left for Postgres."""
    for sql in (
        # the prose is the point
        "SELECT answer(notes, 'which city?') FROM events",
        # the WHERE pipeline owns this one
        "SELECT event_id_cnty FROM events WHERE answer(notes, 'q') = 'Yes'",
        # RHS is not a literal, so there is no candidate answer to verify
        "SELECT answer(notes, 'q') = other_col FROM events",
        # the call is wrapped, so replacing it with a boolean would not typecheck
        "SELECT lower(answer(notes, 'q')) = 'yes' FROM events",
        # not a free text function at all
        "SELECT string_agg(notes, ' ') = 'x' FROM events",
        # a nested SelectStmt owns its own projection
        "SELECT (SELECT answer(notes, 'q') = 'Yes' FROM events LIMIT 1) FROM events",
    ):
        assert not _extract_projection_predicates(_stmt(sql)), sql


def check_identical_comparisons_share_one_column():
    """The same comparison written twice must be verified once."""
    sql = (
        "SELECT answer(notes, 'q') = 'Yes' AS a, "
        "CASE WHEN answer(notes, 'q') = 'Yes' THEN 1 END AS b FROM events"
    )
    preds = _extract_projection_predicates(_stmt(sql))
    assert len(preds) == 1, preds
    alias = next(iter(preds))
    assert alias.startswith(_INTERNAL_PRED_COLUMN_PREFIX), alias


def check_alias_is_deterministic():
    a = _projection_predicate_alias("answer(notes, 'q') = 'Yes'")
    b = _projection_predicate_alias("answer(notes, 'q') = 'Yes'")
    c = _projection_predicate_alias("answer(notes, 'q') = 'No'")
    assert a == b and a != c


def check_extraction_does_not_mutate_unless_asked():
    node = _stmt("SELECT CASE WHEN answer(notes,'q')='Yes' THEN 1 END FROM events")
    before = RawStream()(node)
    _extract_projection_predicates(node)
    assert RawStream()(node) == before


def check_replacement_removes_the_udf_call():
    node = _stmt(
        "SELECT SUM(CASE WHEN answer(notes,'q')='Yes' THEN 1 ELSE 0 END) "
        "FROM events WHERE country = 'Ukraine'"
    )
    preds = _extract_projection_predicates(node, replace=True)
    out = RawStream()(node)
    assert "answer(" not in out.lower(), out
    assert all(alias in out for alias in preds), out
    # the WHERE clause is untouched - it has its own pipeline
    assert "country = 'Ukraine'" in out, out


def check_aggregate_text_argument_is_refused():
    node = _stmt(
        "SELECT a FROM events GROUP BY a "
        "HAVING answer(string_agg(notes, ' '), 'q') = 'Yes'"
    )
    try:
        _extract_projection_predicates(node)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError")


def check_contains_aggregate():
    assert _contains_aggregate(_first_target_expr("SELECT sum(x) FROM t"))
    assert _contains_aggregate(_first_target_expr("SELECT count(*) FROM t"))
    assert _contains_aggregate(_first_target_expr("SELECT string_agg(a, ',') FROM t"))
    assert not _contains_aggregate(_first_target_expr("SELECT lower(a) FROM t"))
    assert not _contains_aggregate(_first_target_expr("SELECT COALESCE(a, '') FROM t"))


def check_limit_pushdown_safety():
    """A projection predicate filters nothing, so a LIMIT can be applied before
    verification - unless something downstream reorders or regroups first."""
    cases = [
        ("SELECT a FROM t LIMIT 5", True),
        ("SELECT a FROM t", False),
        ("SELECT a FROM t ORDER BY a LIMIT 5", False),
        ("SELECT a FROM t GROUP BY a LIMIT 5", False),
        ("SELECT DISTINCT a FROM t LIMIT 5", False),
        # the rewritten query would apply the OFFSET a second time
        ("SELECT a FROM t LIMIT 5 OFFSET 10", False),
    ]
    for sql, want in cases:
        assert _if_limit_pushdown_safe(_stmt(sql)) == want, sql


def check_star_is_expanded_so_internals_do_not_leak():
    node = _stmt("SELECT *, answer(notes,'q')='Yes' AS f FROM events")
    _extract_projection_predicates(node, replace=True)
    _expand_star_targets(
        node,
        [("event_id_cnty", "text"), ("notes", "text"), ("_suql_pred_dead", "boolean")],
    )
    out = RawStream()(node)
    assert "*" not in out, out
    assert "event_id_cnty" in out and "notes" in out, out
    assert out.count(_INTERNAL_PRED_COLUMN_PREFIX) == 1, out


def check_max_workers_resolution():
    assert _resolve_max_workers(8) == 8
    assert _resolve_max_workers() == 32
    os.environ["SUQL_MAX_VERIFICATION_WORKERS"] = "7"
    try:
        assert _resolve_max_workers() == 7
        assert _resolve_max_workers(3) == 3, "explicit argument wins over env"
    finally:
        os.environ.pop("SUQL_MAX_VERIFICATION_WORKERS")


def check_parallel_map_preserves_order_with_duplicates():
    got = _parallel_map(lambda x: x * 2, [3, 1, 3, 2, 3], max_workers=4)
    assert got == [6, 2, 6, 4, 6], got
    assert _parallel_map(lambda x: x, []) == []


UNIT_CHECKS = [
    check_detects_bare_boolean_projection,
    check_detects_case_when_and_aggregate_wrappers,
    check_ignores_non_predicate_uses,
    check_identical_comparisons_share_one_column,
    check_alias_is_deterministic,
    check_extraction_does_not_mutate_unless_asked,
    check_replacement_removes_the_udf_call,
    check_aggregate_text_argument_is_refused,
    check_contains_aggregate,
    check_limit_pushdown_safety,
    check_star_is_expanded_so_internals_do_not_leak,
    check_max_workers_resolution,
    check_parallel_map_preserves_order_with_duplicates,
]


# ---------- integration -------------------------------------------------------

class _StubbedVerification:
    """Replaces the LLM verification call with a deterministic keyword match,
    recording the concurrency it was driven at. Everything else - the structural
    SQL, the temp table, the rewritten query - stays real."""

    def __init__(self, accept=("power station", "pipeline"), delay=0.0):
        self.accept = accept
        self.delay = delay
        self.seen = []
        self.max_inflight = 0
        self._inflight = 0
        self._lock = threading.Lock()
        self._original = None

    def __enter__(self):
        self._original = suql_module._verify
        # the memo cache would hide repeated calls from this stub
        suql_module._verified_res = {}

        def _fake_verify(document, field, query, operator, value, *args, **kwargs):
            with self._lock:
                self.seen.append((document, field, query, operator, value))
                self._inflight += 1
                self.max_inflight = max(self.max_inflight, self._inflight)
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                self._inflight -= 1
            text = (document or "").lower()
            return any(k in text for k in self.accept)

        suql_module._verify = _fake_verify
        return self

    def __exit__(self, *exc):
        suql_module._verify = self._original
        return False


def _run(sql, **overrides):
    kwargs = dict(
        table_w_ids={"events": "event_id_cnty"},
        database="acled",
        select_username="select_user",
        select_userpswd="select_user",
        host="127.0.0.1",
        port="5432",
        embedding_server_address=os.environ.get(
            "SUQL_EMBEDDING_SERVER", "http://127.0.0.1:8505"
        ),
        llm_model_name="gpt-4o-mini",
        disable_try_catch=True,
        statement_timeout=300000,
    )
    kwargs.update(overrides)
    return suql_execute(sql, **kwargs)


def check_reported_repro():
    """The reported shape. Every row used to come back 0."""
    sql = (
        "SELECT event_id_cnty, "
        "CASE WHEN answer(notes, '{}') = 'Yes' THEN 1 ELSE 0 END AS flag "
        "FROM events WHERE event_id_cnty IN ({}) ORDER BY event_id_cnty"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification() as stub:
        rows, columns, _ = _run(sql)

    assert columns == ["event_id_cnty", "flag"], columns
    got = dict(rows)
    assert len(got) == len(ALL_IDS), got
    for i in POSITIVE_IDS:
        assert got[i] == 1, (i, got)
    for i in NEGATIVE_IDS:
        assert got[i] == 0, (i, got)
    assert len(stub.seen) == len(ALL_IDS), stub.seen


def check_verification_sees_the_question_and_the_literal():
    """The comparison is what gets verified, not just the text: the question and
    the compared-against value both have to reach the prompt."""
    sql = (
        "SELECT answer(notes, '{}') = 'Yes' AS f FROM events "
        "WHERE event_id_cnty IN ({})"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification() as stub:
        _run(sql)

    assert stub.seen, "nothing reached verification"
    for _, field, query, operator, value in stub.seen:
        assert query == QUESTION, query
        assert operator == "=", operator
        assert value == "Yes", value
        assert field == ("events", "notes"), field


def check_aggregate_is_not_undercounted():
    sql = (
        "SELECT SUM(CASE WHEN answer(notes, '{}') = 'Yes' THEN 1 ELSE 0 END) AS n "
        "FROM events WHERE event_id_cnty IN ({})"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert rows and rows[0][0] == len(POSITIVE_IDS), rows


def check_projection_is_boolean_not_prose():
    sql = (
        "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f FROM events "
        "WHERE event_id_cnty IN ({})"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert rows and all(isinstance(r[1], bool) for r in rows), rows


def check_having_and_order_by():
    with _StubbedVerification():
        rows, _, _ = _run(
            "SELECT country, COUNT(*) AS n FROM events WHERE event_id_cnty IN ({}) "
            "GROUP BY country HAVING bool_or(answer(notes, '{}') = 'Yes')".format(
                ID_LITERALS, QUESTION
            )
        )
    assert len(rows) == 1 and rows[0][1] == len(ALL_IDS), rows

    with _StubbedVerification():
        rows, _, _ = _run(
            "SELECT event_id_cnty FROM events WHERE event_id_cnty IN ({}) "
            "ORDER BY answer(notes, '{}') = 'Yes' DESC, event_id_cnty".format(
                ID_LITERALS, QUESTION
            )
        )
    assert set(r[0] for r in rows[: len(POSITIVE_IDS)]) == set(POSITIVE_IDS), rows


def check_internal_column_does_not_leak_into_select_star():
    sql = (
        "SELECT *, answer(notes, '{}') = 'Yes' AS f FROM events "
        "WHERE event_id_cnty IN ({})"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification():
        _, columns, _ = _run(sql)
    assert not [c for c in columns if str(c).startswith("_suql_")], columns
    assert "f" in columns, columns


def check_two_questions_in_one_query():
    """rows x predicates, all independent - both columns must be filled in."""
    sql = (
        "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS energy, "
        "answer(notes, 'Did this happen in Ukraine?') = 'Yes' AS ukraine "
        "FROM events WHERE event_id_cnty IN ({}) ORDER BY event_id_cnty"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification() as stub:
        rows, columns, _ = _run(sql)
    assert columns == ["event_id_cnty", "energy", "ukraine"], columns
    assert len(stub.seen) == 2 * len(ALL_IDS), len(stub.seen)
    questions = set(q for _, _, q, _, _ in stub.seen)
    assert len(questions) == 2, questions


def check_limit_is_pushed_down_on_an_unfiltered_query():
    """Without pushdown this would verify every row of a 1.6M-row table."""
    sql = "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f FROM events LIMIT 3".format(
        QUESTION
    )
    with _StubbedVerification() as stub:
        rows, _, _ = _run(sql)
    assert len(rows) == 3, len(rows)
    assert len(stub.seen) == 3, len(stub.seen)


def check_limit_with_offset_still_returns_rows():
    """The LIMIT may only be pushed down when the rewritten query re-applying it
    is a no-op; with an OFFSET it is not."""
    sql = (
        "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f FROM events "
        "WHERE country = 'Ukraine' AND year = 2022 LIMIT 3 OFFSET 5"
    ).format(QUESTION)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert len(rows) == 3, len(rows)


def check_column_projected_under_its_own_name_resolves():
    """A FROM shape that is neither a single RangeVar nor a join still projects
    the text column under its own name."""
    sql = (
        "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f "
        "FROM (SELECT event_id_cnty, notes FROM events WHERE event_id_cnty IN ({})) sub"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    got = dict(rows)
    assert len(got) == len(ALL_IDS), got
    for i in POSITIVE_IDS:
        assert got[i] is True, (i, got)
    for i in NEGATIVE_IDS:
        assert got[i] is False, (i, got)


def check_computed_text_expression_in_a_projection():
    """The issue #50 text-argument support, on the projection side."""
    sql = (
        "SELECT event_id_cnty, "
        "answer(COALESCE(notes,'') || ' ' || COALESCE(tags,''), '{}') = 'Yes' AS f "
        "FROM events WHERE event_id_cnty IN ({}) ORDER BY event_id_cnty"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification() as stub:
        rows, _, _ = _run(sql)
    got = dict(rows)
    for i in POSITIVE_IDS:
        assert got[i] is True, (i, got)
    # the LLM must see the whole expression, not just `notes`
    assert stub.seen and all(
        isinstance(f, suql_module._ComputedTextField) for _, f, _, _, _ in stub.seen
    ), stub.seen


def check_free_text_where_plus_projection_predicate():
    """Both stages run: the WHERE clause filters via the retriever, then the
    surviving rows get their projection predicate verified."""
    sql = (
        "SELECT event_id_cnty, answer(notes, 'Did this happen in Ukraine?') = 'Yes' AS f "
        "FROM events WHERE event_id_cnty IN ({}) AND answer(notes, '{}') = 'Yes'"
    ).format(ID_LITERALS, QUESTION)
    with _StubbedVerification():
        rows, columns, _ = _run(sql)
    assert columns == ["event_id_cnty", "f"], columns
    assert sorted(r[0] for r in rows) == sorted(POSITIVE_IDS), rows


def check_plain_projection_answer_is_left_to_postgres():
    """A projection that wants prose must not be rewritten - the compiler has
    nothing to verify there."""
    node = _stmt(
        "SELECT event_id_cnty, answer(notes, 'which city?') FROM events "
        "WHERE country = 'Ukraine'"
    )
    assert not _extract_projection_predicates(node)


def check_projection_predicate_inside_a_cte():
    """CTE bodies go through the same visitor, so a predicate in one has to be
    verified before the outer query filters on it."""
    sql = (
        "WITH flagged AS ("
        "  SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f "
        "  FROM events WHERE event_id_cnty IN ({})"
        ") SELECT event_id_cnty FROM flagged WHERE f"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert sorted(r[0] for r in rows) == sorted(POSITIVE_IDS), rows


def check_cte_predicate_feeding_an_outer_aggregate():
    sql = (
        "WITH flagged AS ("
        "  SELECT country, CASE WHEN answer(notes, '{}') = 'Yes' THEN 1 ELSE 0 END AS f "
        "  FROM events WHERE event_id_cnty IN ({})"
        ") SELECT country, SUM(f) FROM flagged GROUP BY country"
    ).format(QUESTION, ID_LITERALS)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert rows == [("Ukraine", len(POSITIVE_IDS))], rows


def check_max_verification_workers_is_honored():
    sql = (
        "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f FROM events "
        "WHERE country = 'Ukraine' AND year = 2022 LIMIT 24"
    ).format(QUESTION)

    with _StubbedVerification(delay=0.2) as serial:
        start = time.time()
        rows, _, _ = _run(sql, max_verification_workers=1)
        serial_elapsed = time.time() - start
    assert len(rows) == 24, len(rows)
    assert serial.max_inflight == 1, serial.max_inflight

    with _StubbedVerification(delay=0.2) as parallel:
        start = time.time()
        rows, _, _ = _run(sql, max_verification_workers=12)
        parallel_elapsed = time.time() - start
    assert len(rows) == 24, len(rows)
    assert parallel.max_inflight > 1, parallel.max_inflight
    assert parallel.max_inflight <= 12, parallel.max_inflight
    assert parallel_elapsed < serial_elapsed / 2, (parallel_elapsed, serial_elapsed)


INTEGRATION_CHECKS = [
    check_reported_repro,
    check_verification_sees_the_question_and_the_literal,
    check_aggregate_is_not_undercounted,
    check_projection_is_boolean_not_prose,
    check_having_and_order_by,
    check_internal_column_does_not_leak_into_select_star,
    check_two_questions_in_one_query,
    check_limit_is_pushed_down_on_an_unfiltered_query,
    check_limit_with_offset_still_returns_rows,
    check_column_projected_under_its_own_name_resolves,
    check_computed_text_expression_in_a_projection,
    check_free_text_where_plus_projection_predicate,
    check_plain_projection_answer_is_left_to_postgres,
    check_projection_predicate_inside_a_cte,
    check_cte_predicate_feeding_an_outer_aggregate,
    check_max_verification_workers_is_honored,
]


# ---------- runner ------------------------------------------------------------

def main():
    failures = []
    print("--- unit ---")
    for check in UNIT_CHECKS:
        try:
            check()
            print("[PASS] {}".format(check.__name__))
        except Exception:
            print("[FAIL] {}".format(check.__name__))
            traceback.print_exc()
            failures.append(check.__name__)

    print("\n--- integration (real acled DB, stubbed LLM) ---")
    for check in INTEGRATION_CHECKS:
        try:
            check()
            print("[PASS] {}".format(check.__name__))
        except Exception:
            print("[FAIL] {}".format(check.__name__))
            traceback.print_exc()
            failures.append(check.__name__)

    if failures:
        print("\n{} failed: {}".format(len(failures), failures))
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
