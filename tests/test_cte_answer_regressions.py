"""Regressions for CTE materialization around `answer()`.

Four failure modes, all from one reported pipeline run in which 12/12 SUQL
executions failed. Each is the same underlying mistake in a different place:
a CTE is materialized into a temp table, and something the rest of the query
needs is not carried across.

  A  `missing FROM-clause entry for table "b"`
     `FROM base b` ... `SELECT b.*`. Swapping `base` for `temp_table_xxx`
     dropped the name the body refers to the relation by.

  B  `missing FROM-clause entry for table "base"`
     Same, for an unaliased `FROM base` with `base.<col>` references.

  C  `KeyError: 'temp_table_xxx'`
     A CTE that does not project the source table's ID column cannot be
     registered in `table_w_ids`, so the pipeline could not identify its rows.

  D  `column "russian_yes" does not exist`
     Downstream references were redirected to a temp table that holds the CTE's
     *input* rather than its output, dropping every column the body computes.

Integration checks run against the live Postgres `acled` DB with only the LLM
verification call stubbed out, so the CTE machinery, temp tables and rewritten
SQL are all real.
"""

import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pglast import parse_sql
from pglast.ast import Alias, RangeVar
from pglast.stream import RawStream

import suql.sql_free_text_support.execute_free_text_sql as suql_module
from suql.sql_free_text_support.execute_free_text_sql import (
    _SelectVisitor,
    _if_body_is_bare_select_star,
    _single_rangevar_relname,
    suql_execute,
)

QUESTION = "Is this an attack on energy infrastructure?"
POSITIVE_IDS = ["UKR73236", "UKR69217", "UKR88237"]
NEGATIVE_IDS = ["UKR127789", "UKR73096"]
ALL_IDS = POSITIVE_IDS + NEGATIVE_IDS
ID_LITERALS = ", ".join("'{}'".format(i) for i in ALL_IDS)


def _stmt(sql):
    return parse_sql(sql)[0].stmt


# ---------- unit checks -------------------------------------------------------

def check_rewrite_cte_refs_keeps_the_referenced_name():
    """`FROM base` must become `FROM temp_table_x AS base`, or every
    `base.<col>` in the same body dangles."""
    body = _stmt("SELECT base.a, base.b FROM base")
    _SelectVisitor._rewrite_cte_refs(body, {"base": "temp_table_x"})
    out = RawStream()(body)
    assert "temp_table_x AS base" in out, out
    assert _single_rangevar_relname(body.fromClause) == "temp_table_x"


def check_rewrite_cte_refs_preserves_an_explicit_alias():
    """An explicit alias is already the name in use - keep it, don't replace
    it with the CTE name."""
    body = _stmt("SELECT b.* FROM base b")
    _SelectVisitor._rewrite_cte_refs(body, {"base": "temp_table_x"})
    out = RawStream()(body)
    assert "temp_table_x AS b" in out, out
    assert " AS base" not in out, out


def check_rewrite_cte_refs_leaves_unmapped_tables_alone():
    body = _stmt("SELECT * FROM events")
    _SelectVisitor._rewrite_cte_refs(body, {"base": "temp_table_x"})
    assert RawStream()(body) == "SELECT * FROM events", RawStream()(body)


def check_bare_select_star_detection():
    """A temp table can stand in for a CTE only when the body adds nothing."""
    assert _if_body_is_bare_select_star(_stmt("SELECT * FROM temp_table_x"))
    for sql in (
        "SELECT a, b FROM temp_table_x",
        "SELECT * FROM temp_table_x WHERE a > 1",
        "SELECT * FROM temp_table_x GROUP BY a",
        "SELECT DISTINCT * FROM temp_table_x",
        "SELECT * FROM temp_table_x LIMIT 5",
        "SELECT * FROM a JOIN b ON a.i = b.i",
        "SELECT *, 1 AS extra FROM temp_table_x",
    ):
        assert not _if_body_is_bare_select_star(_stmt(sql)), sql


def check_single_rangevar_relname():
    assert _single_rangevar_relname(_stmt("SELECT * FROM t").fromClause) == "t"
    assert _single_rangevar_relname(_stmt("SELECT 1").fromClause) is None
    assert _single_rangevar_relname(
        _stmt("SELECT * FROM a, b").fromClause
    ) is None
    assert _single_rangevar_relname(
        _stmt("SELECT * FROM a JOIN b ON a.i = b.i").fromClause
    ) is None


UNIT_CHECKS = [
    check_rewrite_cte_refs_keeps_the_referenced_name,
    check_rewrite_cte_refs_preserves_an_explicit_alias,
    check_rewrite_cte_refs_leaves_unmapped_tables_alone,
    check_bare_select_star_detection,
    check_single_rangevar_relname,
]


# ---------- integration -------------------------------------------------------

class _StubbedVerification:
    def __init__(self):
        self.calls = 0
        self._original = None

    def __enter__(self):
        self._original = suql_module._verify
        suql_module._verified_res = {}

        def _fake_verify(document, field, query, operator, value, *a, **k):
            self.calls += 1
            text = (document or "").lower()
            return "power station" in text or "pipeline" in text

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


_EXPECTED_FLAGS = [(i, 1 if i in POSITIVE_IDS else 0) for i in sorted(ALL_IDS)]


def check_group_a_cte_alias_and_star():
    """`FROM base b` + `SELECT b.*`."""
    sql = """
    WITH base AS (
      SELECT event_id_cnty, country, notes FROM events WHERE event_id_cnty IN ({ids})
    ), classified AS (
      SELECT b.*, CASE WHEN answer(b.notes, '{q}') = 'Yes' THEN 1 ELSE 0 END AS flag
      FROM base b
    )
    SELECT event_id_cnty, flag FROM classified ORDER BY event_id_cnty
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        rows, columns, _ = _run(sql)
    assert columns == ["event_id_cnty", "flag"], columns
    assert rows == _EXPECTED_FLAGS, rows


def check_group_b_cte_referenced_by_name():
    """`FROM base` + `base.<col>`."""
    sql = """
    WITH base AS (
      SELECT event_id_cnty, country, notes FROM events WHERE event_id_cnty IN ({ids})
    ), classified AS (
      SELECT base.event_id_cnty, base.country,
             CASE WHEN answer(base.notes, '{q}') = 'Yes' THEN 1 ELSE 0 END AS flag
      FROM base
    )
    SELECT event_id_cnty, flag FROM classified ORDER BY event_id_cnty
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        rows, columns, _ = _run(sql)
    assert columns == ["event_id_cnty", "flag"], columns
    assert rows == _EXPECTED_FLAGS, rows


def check_group_c_cte_drops_the_id_column():
    """The CTE never projects `event_id_cnty`; the compiler must add it back
    rather than failing to register the temp table."""
    sql = """
    WITH base AS (
      SELECT country, admin1, notes FROM events WHERE event_id_cnty IN ({ids})
    ), hits AS (
      SELECT * FROM base WHERE answer(notes, '{q}') = 'Yes'
    )
    SELECT country, COUNT(*) AS n FROM hits GROUP BY country
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert rows == [("Ukraine", len(POSITIVE_IDS))], rows


def check_group_c_does_not_leak_the_id_into_results():
    """Adding the ID back is internal - it must not change what the user's
    own projection returns."""
    sql = """
    WITH base AS (
      SELECT country, admin1, notes FROM events WHERE event_id_cnty IN ({ids})
    ), hits AS (
      SELECT country, admin1 FROM base WHERE answer(notes, '{q}') = 'Yes'
    )
    SELECT * FROM hits ORDER BY admin1
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        rows, columns, _ = _run(sql)
    assert columns == ["country", "admin1"], columns
    assert len(rows) == len(POSITIVE_IDS), rows


def check_group_d_derived_column_survives_materialization():
    """A column the CTE computes must be visible to the next CTE."""
    sql = """
    WITH base AS (
      SELECT event_id_cnty, country, notes FROM events WHERE event_id_cnty IN ({ids})
    ), validated AS (
      SELECT event_id_cnty, country,
             CASE WHEN TRUE THEN answer(notes, '{q}') = 'Yes' END AS is_energy
      FROM base
    ), tall AS (
      SELECT country, COUNT(*) FILTER (WHERE is_energy) AS n_yes, COUNT(*) AS n_total
      FROM validated GROUP BY country
    )
    SELECT country, n_yes, n_total FROM tall
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        rows, columns, _ = _run(sql)
    assert columns == ["country", "n_yes", "n_total"], columns
    assert rows == [("Ukraine", len(POSITIVE_IDS), len(ALL_IDS))], rows


def check_aggregating_cte_reports_clearly():
    """A CTE that aggregates has no row to identify. That cannot be fixed by
    projecting an ID, so it must produce an actionable error rather than a bare
    KeyError on an internal table name."""
    # `notes` is grouped by, not computed, so it keeps a traceable lineage to
    # events.notes - which isolates the missing-ID failure from the separate
    # (already actionable) "no traceable lineage" one.
    sql = """
    WITH per_country AS (
      SELECT country, notes
      FROM events WHERE event_id_cnty IN ({ids}) GROUP BY country, notes
    ), hits AS (
      SELECT * FROM per_country WHERE answer(notes, '{q}') = 'Yes'
    )
    SELECT country FROM hits
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        try:
            _run(sql)
        except NotImplementedError as e:
            assert "no known ID column" in str(e), str(e)
            return
        except KeyError as e:
            raise AssertionError("bare KeyError leaked: {}".format(e))
    raise AssertionError("expected NotImplementedError")


def check_plain_cte_pipeline_still_works():
    """The shapes #45 added must be unaffected."""
    sql = """
    WITH base AS (
      SELECT * FROM events WHERE event_id_cnty IN ({ids})
    )
    SELECT event_id_cnty FROM base WHERE answer(notes, '{q}') = 'Yes'
    ORDER BY event_id_cnty
    """.format(ids=ID_LITERALS, q=QUESTION)
    with _StubbedVerification():
        rows, _, _ = _run(sql)
    assert sorted(r[0] for r in rows) == sorted(POSITIVE_IDS), rows


INTEGRATION_CHECKS = [
    check_group_a_cte_alias_and_star,
    check_group_b_cte_referenced_by_name,
    check_group_c_cte_drops_the_id_column,
    check_group_c_does_not_leak_the_id_into_results,
    check_group_d_derived_column_survives_materialization,
    check_aggregating_cte_reports_clearly,
    check_plain_cte_pipeline_still_works,
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
