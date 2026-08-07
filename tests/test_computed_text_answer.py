"""Tests for issue #50: `answer()` over a computed text expression.

Before this, the text argument of `answer()` had to be a bare `ColumnRef` -
anything computed (`COALESCE(...)`, `||`, `lower(...)`, ...) tripped
`assert len(field_lst) == 1` in `breakdown_unstructural_query`.

Now the compiler projects the expression under an internal alias in the
structural query, retrieves candidates using the indexed columns the expression
reads from, and verifies the expression's own value with the LLM.

Unit tests check the AST-level helpers in isolation.

Integration tests run the full pipeline against the live Postgres `acled` DB and
the live embedding server (default http://127.0.0.1:8505), with only the LLM
verification call stubbed out - so the SQL projection, the retriever call and
the document that would be handed to the LLM are all real.
"""

import os
import sys
import traceback

import requests as _requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pglast import parse_sql
from pglast.ast import ColumnRef

import suql.sql_free_text_support.execute_free_text_sql as suql_module
from suql.sql_free_text_support.execute_free_text_sql import (
    _INTERNAL_EXPR_COLUMN_PREFIX,
    _ComputedTextField,
    _computed_text_alias,
    _drop_internal_columns,
    _extract_all_free_text_fcns,
    _extract_column_refs,
    _extract_computed_text_args,
    _field_display_name,
    _free_text_fcn_args,
    suql_execute,
)

EMBEDDING_SERVER_ADDRESS = os.environ.get(
    "SUQL_EMBEDDING_SERVER", "http://127.0.0.1:8505"
)
QUESTION = "Does this event involve the Colombian military?"
CONCAT_TEXT = "COALESCE(e.notes, '') || ' | tags: ' || COALESCE(e.tags, '')"


def _where_clause(sql):
    return parse_sql(sql)[0].stmt.whereClause


def _answer_call(sql):
    """The `answer(...)` FuncCall of `WHERE answer(...) = 'Yes'`."""
    return _where_clause(sql).lexpr


def _embedding_server_reachable():
    try:
        r = _requests.post(
            EMBEDDING_SERVER_ADDRESS + "/search",
            json={
                "id_list": [],
                "field_query_list": [["events", "notes"]],
                "top": 1,
                "single_table": True,
            },
            timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False


# ---------- unit checks -------------------------------------------------------

def check_alias_is_deterministic_and_prefixed():
    a = _computed_text_alias("COALESCE(notes, '') || tags")
    b = _computed_text_alias("COALESCE(notes, '') || tags")
    c = _computed_text_alias("COALESCE(notes, '') || source")
    assert a == b, (a, b)
    assert a != c, (a, c)
    assert a.startswith(_INTERNAL_EXPR_COLUMN_PREFIX), a
    # must be usable as a bare SQL identifier
    assert a.replace("_", "").isalnum(), a


def check_free_text_fcn_args_positional():
    call = _answer_call("SELECT * FROM events e WHERE answer(e.notes, 'q') = 'Yes'")
    text_arg, query = _free_text_fcn_args(call)
    assert isinstance(text_arg, ColumnRef), text_arg
    assert query == "q", query


def check_free_text_fcn_args_computed_text():
    call = _answer_call(
        f"SELECT * FROM events e WHERE answer({CONCAT_TEXT}, 'q') = 'Yes'"
    )
    text_arg, query = _free_text_fcn_args(call)
    assert not isinstance(text_arg, ColumnRef), text_arg
    assert query == "q", query


def check_free_text_fcn_args_rejects_missing_question():
    call = _answer_call("SELECT * FROM events e WHERE answer(e.notes) = 'Yes'")
    try:
        _free_text_fcn_args(call)
    except ValueError as e:
        assert "question" in str(e), e
        return
    raise AssertionError("expected ValueError for a one-argument answer()")


def check_free_text_fcn_args_rejects_non_literal_question():
    call = _answer_call(
        "SELECT * FROM events e WHERE answer(e.notes, e.tags) = 'Yes'"
    )
    try:
        _free_text_fcn_args(call)
    except ValueError as e:
        assert "string literal" in str(e), e
        return
    raise AssertionError("expected ValueError for a non-literal question")


def check_extract_computed_text_args_finds_expression():
    where = _where_clause(
        f"SELECT * FROM events e WHERE answer({CONCAT_TEXT}, 'q') = 'Yes'"
    )
    res = _extract_computed_text_args(where)
    assert len(res) == 1, res
    alias = list(res)[0]
    assert alias.startswith(_INTERNAL_EXPR_COLUMN_PREFIX), alias


def check_extract_computed_text_args_ignores_bare_column():
    where = _where_clause("SELECT * FROM events e WHERE answer(e.notes, 'q') = 'Yes'")
    assert _extract_computed_text_args(where) == {}


def check_extract_computed_text_args_dedups_identical_expressions():
    # the same expression asked two different questions is projected once
    where = _where_clause(
        f"SELECT * FROM events e WHERE answer({CONCAT_TEXT}, 'q1') = 'Yes' "
        f"AND answer({CONCAT_TEXT}, 'q2') = 'Yes'"
    )
    assert len(_extract_computed_text_args(where)) == 1


def check_extract_computed_text_args_handles_function_call():
    where = _where_clause(
        "SELECT * FROM events e WHERE answer(lower(e.notes), 'q') = 'Yes'"
    )
    assert len(_extract_computed_text_args(where)) == 1


def check_extract_column_refs_finds_underlying_columns():
    call = _answer_call(
        f"SELECT * FROM events e WHERE answer({CONCAT_TEXT}, 'q') = 'Yes'"
    )
    text_arg, _ = _free_text_fcn_args(call)
    names = {
        tuple(f.sval for f in ref.fields) for ref in _extract_column_refs(text_arg)
    }
    assert names == {("e", "notes"), ("e", "tags")}, names


def check_computed_text_field_is_tuple_compatible():
    field = _ComputedTextField(
        table="events",
        column="_suql_expr_abc",
        expr_sql="notes || tags",
        source_columns=(("events", "notes"),),
    )
    assert field[0] == "events"
    assert field[1] == "_suql_expr_abc"
    # used as a dict key by the verification cache
    assert {field: 1}[field] == 1


def check_field_display_name():
    field = _ComputedTextField(
        table="events", column="_suql_expr_abc", expr_sql="notes || tags"
    )
    # the LLM is shown the expression, not the meaningless internal alias
    assert _field_display_name(field) == "notes || tags"
    assert _field_display_name(("events", "notes")) == "notes"


def check_drop_internal_columns():
    column_info = [
        ("event_id_cnty", "text"),
        (_INTERNAL_EXPR_COLUMN_PREFIX + "abc", "text"),
        ("notes", "text"),
    ]
    results = [("a", "computed a", "notes a"), ("b", "computed b", "notes b")]
    rows, columns = _drop_internal_columns(results, column_info)
    assert columns == [("event_id_cnty", "text"), ("notes", "text")], columns
    assert rows == [("a", "notes a"), ("b", "notes b")], rows


def check_drop_internal_columns_is_a_noop_without_them():
    column_info = [("event_id_cnty", "text")]
    results = [("a",)]
    rows, columns = _drop_internal_columns(results, column_info)
    assert rows is results and columns is column_info


def check_extract_all_free_text_fcns_handles_computed_text():
    res = _extract_all_free_text_fcns(
        f"SELECT * FROM events e WHERE answer({CONCAT_TEXT}, 'q') = 'Yes'"
    )
    assert len(res) == 1, res
    assert res[0][1] == "q", res


UNIT_CHECKS = [
    check_alias_is_deterministic_and_prefixed,
    check_free_text_fcn_args_positional,
    check_free_text_fcn_args_computed_text,
    check_free_text_fcn_args_rejects_missing_question,
    check_free_text_fcn_args_rejects_non_literal_question,
    check_extract_computed_text_args_finds_expression,
    check_extract_computed_text_args_ignores_bare_column,
    check_extract_computed_text_args_dedups_identical_expressions,
    check_extract_computed_text_args_handles_function_call,
    check_extract_column_refs_finds_underlying_columns,
    check_computed_text_field_is_tuple_compatible,
    check_field_display_name,
    check_drop_internal_columns,
    check_drop_internal_columns_is_a_noop_without_them,
    check_extract_all_free_text_fcns_handles_computed_text,
]


# ---------- integration checks (real DB + real retriever, stubbed LLM) --------

class _StubbedVerification:
    """Replaces the LLM verification call with a deterministic keyword match,
    recording every (document, field) pair it was asked about. Everything else -
    the structural SQL, the retriever call, the temp table - stays real."""

    def __init__(self, accept="military"):
        self.accept = accept
        self.seen = []
        self._original = None

    def __enter__(self):
        self._original = suql_module._verify

        def _fake_verify(document, field, query, operator, value, *args, **kwargs):
            self.seen.append((document, field))
            return self.accept in (document or "").lower()

        suql_module._verify = _fake_verify
        return self

    def __exit__(self, *exc):
        suql_module._verify = self._original
        return False

    @property
    def documents(self):
        return [doc for doc, _ in self.seen]


def _run(sql, **overrides):
    kwargs = dict(
        table_w_ids={"events": "event_id_cnty"},
        database="acled",
        select_username="select_user",
        select_userpswd="select_user",
        host="127.0.0.1",
        port="5432",
        embedding_server_address=EMBEDDING_SERVER_ADDRESS,
        llm_model_name="gpt-4o-mini",
        disable_try_catch=True,
        statement_timeout=120000,
    )
    kwargs.update(overrides)
    return suql_execute(sql, **kwargs)


def check_issue_50_repro():
    """The shape from the issue: answer() over a concatenation of two columns.
    Used to raise AssertionError before reaching the database."""
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer({CONCAT_TEXT}, '{QUESTION}') = 'Yes' LIMIT 3;"
    )
    with _StubbedVerification() as stub:
        rows, columns, _ = _run(sql)

    assert stub.seen, "no row ever reached verification"
    assert rows, "query returned no rows"
    assert columns == ["event_id_cnty"], columns


def check_verification_sees_the_computed_text():
    """The LLM must be handed the value of the whole expression, not one of the
    columns it reads from."""
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer({CONCAT_TEXT}, '{QUESTION}') = 'Yes' LIMIT 3;"
    )
    with _StubbedVerification() as stub:
        _run(sql)

    assert stub.documents, "no document reached verification"
    for document in stub.documents:
        assert " | tags: " in document, document[:200]
    field = stub.seen[0][1]
    assert isinstance(field, _ComputedTextField), field
    assert ("events", "notes") in field.source_columns, field


def check_retriever_prefilters_on_underlying_column():
    """`events.notes` is embedded, `events.tags` is not. Retrieval has to fall
    back to the indexed column rather than verifying all 24k Colombian rows."""
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer({CONCAT_TEXT}, '{QUESTION}') = 'Yes' LIMIT 3;"
    )
    with _StubbedVerification() as stub:
        _run(sql)

    colombia_rows = 20000  # the table holds far more Colombian events than this
    assert len(stub.seen) < colombia_rows, (
        f"{len(stub.seen)} verification calls - the retriever did not prefilter"
    )


def check_internal_column_does_not_leak_into_select_star():
    sql = (
        "SELECT * FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer({CONCAT_TEXT}, '{QUESTION}') = 'Yes' LIMIT 1;"
    )
    with _StubbedVerification():
        rows, columns, _ = _run(sql)

    leaked = [c for c in columns if c.startswith(_INTERNAL_EXPR_COLUMN_PREFIX)]
    assert not leaked, leaked
    if rows:
        assert len(rows[0]) == len(columns), (len(rows[0]), len(columns))


def check_single_argument_function_expression():
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer(lower(e.notes), '{QUESTION}') = 'Yes' LIMIT 2;"
    )
    with _StubbedVerification() as stub:
        rows, _, _ = _run(sql)

    assert stub.documents, "no document reached verification"
    for document in stub.documents:
        assert document == document.lower(), document[:200]


def check_computed_and_plain_answer_together():
    """AND of a computed-text predicate and a plain-column one."""
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer({CONCAT_TEXT}, '{QUESTION}') = 'Yes' "
        f"AND answer(e.notes, '{QUESTION}') = 'Yes' LIMIT 2;"
    )
    with _StubbedVerification() as stub:
        _run(sql)

    fields = {type(field) for _, field in stub.seen}
    assert _ComputedTextField in fields, fields
    assert tuple in fields, fields


def check_unretrievable_expression_over_large_table_is_refused():
    """An expression with no retrievable column would mean one LLM call per
    surviving row - the compiler must refuse rather than silently do it."""
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer('a' || 'b', '{QUESTION}') = 'Yes' LIMIT 1;"
    )
    with _StubbedVerification():
        try:
            _run(sql)
        except NotImplementedError as e:
            assert "disable_retriever" in str(e), e
            return
    raise AssertionError("expected NotImplementedError")


def check_plain_column_answer_still_works():
    """Regression: the bare-column path must be untouched."""
    sql = (
        "SELECT e.event_id_cnty FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer(e.notes, '{QUESTION}') = 'Yes' LIMIT 3;"
    )
    with _StubbedVerification() as stub:
        rows, columns, _ = _run(sql)

    assert rows, "query returned no rows"
    assert columns == ["event_id_cnty"], columns
    assert all(field == ("events", "notes") for _, field in stub.seen)


def check_qualified_projection_resolves_against_temp_table():
    """Independent of the computed-text work, but needed for the issue's own
    repro: a table-qualified projection must survive the FROM clause being
    swapped for the temp table holding the verified rows."""
    sql = (
        "SELECT e.event_id_cnty, e.notes FROM events e "
        "WHERE e.country = 'Colombia' "
        f"AND answer(e.notes, '{QUESTION}') = 'Yes' LIMIT 2;"
    )
    with _StubbedVerification():
        rows, columns, _ = _run(sql)

    assert rows, "query returned no rows"
    assert columns == ["event_id_cnty", "notes"], columns


def check_two_materialized_ctes_plus_outer_answer():
    """Two CTE bodies over the same table both become temp tables, and the
    outer query then runs its own answer(). The alias each temp table carries
    must not be mistaken for a user alias to rewrite."""
    sql = (
        "WITH a AS ("
        "  SELECT e.event_id_cnty, e.notes FROM events e"
        f"  WHERE e.country = 'Colombia' AND answer(e.notes, '{QUESTION}') = 'Yes'"
        "  LIMIT 5"
        "), b AS ("
        "  SELECT e.event_id_cnty, e.notes FROM events e"
        "  WHERE e.country = 'Colombia'"
        "  AND answer(e.notes, 'Does this event involve civilians?') = 'Yes'"
        "  LIMIT 5"
        ") "
        "SELECT event_id_cnty FROM a "
        f"WHERE answer(notes, '{QUESTION}') = 'Yes' "
        "AND event_id_cnty NOT IN (SELECT event_id_cnty FROM b) LIMIT 2;"
    )
    with _StubbedVerification(accept="colombia"):
        rows, columns, _ = _run(sql)

    assert columns == ["event_id_cnty"], columns


def check_computed_text_in_cte():
    """The issue's own workaround shape, plus a computed expression on top of
    the CTE's projection."""
    sql = (
        "WITH enriched AS ("
        "  SELECT e.event_id_cnty, e.notes, e.tags FROM events e"
        "  WHERE e.country = 'Colombia'"
        ") "
        "SELECT event_id_cnty FROM enriched "
        f"WHERE answer(COALESCE(notes, '') || ' | tags: ' || COALESCE(tags, ''), "
        f"'{QUESTION}') = 'Yes' LIMIT 2;"
    )
    with _StubbedVerification() as stub:
        rows, columns, _ = _run(sql)

    assert stub.documents, "no document reached verification"
    for document in stub.documents:
        assert " | tags: " in document, document[:200]
    assert columns == ["event_id_cnty"], columns


INTEGRATION_CHECKS = [
    check_issue_50_repro,
    check_verification_sees_the_computed_text,
    check_retriever_prefilters_on_underlying_column,
    check_internal_column_does_not_leak_into_select_star,
    check_single_argument_function_expression,
    check_computed_and_plain_answer_together,
    check_unretrievable_expression_over_large_table_is_refused,
    check_plain_column_answer_still_works,
    check_qualified_projection_resolves_against_temp_table,
    check_two_materialized_ctes_plus_outer_answer,
    check_computed_text_in_cte,
]


# ---------- runner ------------------------------------------------------------

def main():
    failures = []
    print("--- unit ---")
    for check in UNIT_CHECKS:
        try:
            check()
            print(f"[PASS] {check.__name__}")
        except Exception:
            print(f"[FAIL] {check.__name__}")
            traceback.print_exc()
            failures.append(check.__name__)

    print("\n--- integration (real acled DB + embedding server, stubbed LLM) ---")
    if not _embedding_server_reachable():
        print(
            f"[SKIP] embedding server at {EMBEDDING_SERVER_ADDRESS} not reachable — "
            f"skipping {len(INTEGRATION_CHECKS)} integration checks"
        )
    else:
        for check in INTEGRATION_CHECKS:
            try:
                check()
                print(f"[PASS] {check.__name__}")
            except Exception:
                print(f"[FAIL] {check.__name__}")
                traceback.print_exc()
                failures.append(check.__name__)

    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
