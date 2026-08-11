"""Tests for the per-query verification cost ceiling.

Background: a reported pipeline run spent 25 hours and $101.51 on 4,326,211
verification calls without a single query completing. The predicates were broad
free-text questions over a large candidate set. Parallelism made that faster,
not cheaper - nothing bounded the total.

`suql_execute` now takes `max_verification_cost` (default $1.00). It is enforced
in two places, and both matter:

  - before every verification, so a query can never run past the ceiling;
  - after a sample of calls, by extrapolating the mean over the remaining
    planned verifications - so a query that would cost $100 is refused after
    cents instead of after the full ceiling.

These tests stub `llm_generate` rather than `_verify`, so the real `_verify`
runs (including its budget check) and the tracker accumulates a known,
deterministic cost per call.
"""

import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import suql.sql_free_text_support.execute_free_text_sql as suql_module
from suql.sql_free_text_support.execute_free_text_sql import (
    _COST_PROJECTION_SAMPLE,
    _VerificationBudget,
    _resolve_verification_budget,
    SUQLCostLimitExceeded,
    suql_execute,
)

QUESTION = "Is this an attack on energy infrastructure?"


class _FakeTracker(dict):
    def __init__(self):
        import threading

        super().__init__(cost=0.0, calls=0, _lock=threading.Lock(), debug_log=None)


# ---------- unit checks -------------------------------------------------------

def check_defaults_and_overrides():
    assert _resolve_verification_budget(None, None, None).max_cost == 1.0
    assert _resolve_verification_budget(5.0, None, None).max_cost == 5.0
    # 0 or negative removes the ceiling
    assert _resolve_verification_budget(0, None, None).max_cost is None
    assert _resolve_verification_budget(-1, None, None).max_cost is None

    os.environ["SUQL_MAX_VERIFICATION_COST"] = "0.25"
    os.environ["SUQL_MAX_VERIFICATION_CALLS"] = "40"
    try:
        b = _resolve_verification_budget(None, None, None)
        assert b.max_cost == 0.25, b.max_cost
        assert b.max_calls == 40, b.max_calls
        # an explicit argument still wins over the environment
        assert _resolve_verification_budget(2.0, None, None).max_cost == 2.0
    finally:
        os.environ.pop("SUQL_MAX_VERIFICATION_COST")
        os.environ.pop("SUQL_MAX_VERIFICATION_CALLS")

    os.environ["SUQL_MAX_VERIFICATION_COST"] = "not-a-number"
    try:
        assert _resolve_verification_budget(None, None, None).max_cost == 1.0
    finally:
        os.environ.pop("SUQL_MAX_VERIFICATION_COST")


def check_ceiling_trips_on_accumulated_spend():
    """With nothing planned there is no projection to go on, so the ceiling is
    reached by accumulation. It must trip at the ceiling, not far past it."""
    tracker = _FakeTracker()
    budget = _VerificationBudget(max_cost=0.10, tracker=tracker)
    n = 0
    try:
        for _ in range(1000):
            budget.before_verification()
            tracker["cost"] += 0.01
            tracker["calls"] += 1
            n += 1
    except SUQLCostLimitExceeded as e:
        assert "cost limit reached" in str(e), str(e)
        # 10 calls covers the ceiling exactly; allow one more for float drift
        assert 10 <= n <= 11, n
        assert tracker["cost"] <= 0.11 + 1e-9, tracker["cost"]
        return
    raise AssertionError("expected SUQLCostLimitExceeded")


def check_call_ceiling():
    tracker = _FakeTracker()
    budget = _VerificationBudget(max_cost=None, max_calls=5, tracker=tracker)
    for _ in range(5):
        budget.before_verification()
        tracker["calls"] += 1
    try:
        budget.before_verification()
    except SUQLCostLimitExceeded as e:
        assert "call limit reached" in str(e), str(e)
        return
    raise AssertionError("expected SUQLCostLimitExceeded")


def check_projection_refuses_before_spending_the_ceiling():
    """The point of the whole exercise: a 4.3M-verification plan must be
    refused after the sample, not after $1 of spend."""
    tracker = _FakeTracker()
    budget = _VerificationBudget(max_cost=1.00, tracker=tracker)
    budget.register_planned(4_326_211)
    per_call = 0.0000235  # ~= $101.51 / 4.3M, the reported rate
    n = 0
    try:
        for _ in range(10_000):
            budget.before_verification()
            tracker["cost"] += per_call
            tracker["calls"] += 1
            n += 1
    except SUQLCostLimitExceeded as e:
        assert "projects to" in str(e), str(e)
        assert "4,326,211" in str(e), str(e)
        # refused right after the sample, having spent a fraction of a cent
        assert n <= _COST_PROJECTION_SAMPLE + 1, n
        assert tracker["cost"] < 0.01, tracker["cost"]
        return
    raise AssertionError("expected SUQLCostLimitExceeded")


def check_projection_allows_an_affordable_plan():
    tracker = _FakeTracker()
    budget = _VerificationBudget(max_cost=1.00, tracker=tracker)
    budget.register_planned(1000)
    for _ in range(200):
        budget.before_verification()          # 1000 * $0.0001 = $0.10, affordable
        tracker["cost"] += 0.0001
        tracker["calls"] += 1
    assert tracker["cost"] < 1.0


def check_no_ceiling_never_trips():
    tracker = _FakeTracker()
    budget = _VerificationBudget(max_cost=None, max_calls=None, tracker=tracker)
    budget.register_planned(10_000_000)
    for _ in range(100):
        budget.before_verification()
        tracker["cost"] += 1.0
        tracker["calls"] += 1


def check_cached_verifications_do_not_inflate_the_projection():
    """Cost is projected per verification *attempt*, so documents served from
    the memo cache (which cost nothing) don't make a cheap plan look expensive."""
    tracker = _FakeTracker()
    budget = _VerificationBudget(max_cost=1.00, tracker=tracker)
    budget.register_planned(10_000)
    # 1 real call in every 20 attempts; 10k attempts => 500 calls => $0.05
    for i in range(2000):
        budget.before_verification()
        if i % 20 == 0:
            tracker["cost"] += 0.0001
            tracker["calls"] += 1
    assert tracker["cost"] < 1.0


UNIT_CHECKS = [
    check_defaults_and_overrides,
    check_ceiling_trips_on_accumulated_spend,
    check_call_ceiling,
    check_projection_refuses_before_spending_the_ceiling,
    check_projection_allows_an_affordable_plan,
    check_no_ceiling_never_trips,
    check_cached_verifications_do_not_inflate_the_projection,
]


# ---------- integration -------------------------------------------------------

COST_PER_CALL = 0.002


class _MeteredLLM:
    """Replaces `llm_generate` so the real `_verify` runs - budget check
    included - while each call books a known cost against the query tracker."""

    def __init__(self, cost_per_call=COST_PER_CALL, verdict="the answer is correct."):
        self.cost_per_call = cost_per_call
        self.verdict = verdict
        self.calls = 0
        self._original = None

    def __enter__(self):
        self._original = suql_module.llm_generate
        suql_module._verified_res = {}

        def _fake(*args, **kwargs):
            tracker = suql_module._query_tracker.get()
            if tracker is not None:
                with tracker["_lock"]:
                    tracker["cost"] += self.cost_per_call
                    tracker["calls"] += 1
            self.calls += 1
            return [self.verdict]

        suql_module.llm_generate = _fake
        return self

    def __exit__(self, *exc):
        suql_module.llm_generate = self._original
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
        free_text_server_address=None,
    )
    kwargs.update(overrides)
    return suql_execute(sql, **kwargs)


_BROAD = (
    "SELECT event_id_cnty, answer(notes, '{}') = 'Yes' AS f FROM events "
    "WHERE country = 'Ukraine' AND year = 2022 LIMIT 400"
).format(QUESTION)


def check_runaway_query_is_refused_cheaply():
    """400 rows at $0.002 each would be $0.80; a $0.05 ceiling must stop it,
    and the projection must stop it near the sample rather than at the ceiling."""
    with _MeteredLLM() as llm:
        try:
            _run(_BROAD, max_verification_cost=0.05)
        except SUQLCostLimitExceeded as e:
            msg = str(e)
            assert "max_verification_cost" in msg, msg
            spent = llm.calls * COST_PER_CALL
            assert spent <= 0.05 + COST_PER_CALL, (llm.calls, spent)
            # the projection should fire around the sample size, well before
            # the ceiling would have been reached by accumulation alone
            assert llm.calls <= _COST_PROJECTION_SAMPLE + 2, llm.calls
            return
    raise AssertionError("expected SUQLCostLimitExceeded")


def check_affordable_query_completes_and_reports_spend():
    with _MeteredLLM(cost_per_call=0.00001) as llm:
        rows, columns, cache = _run(_BROAD, max_verification_cost=1.0)
    assert len(rows) == 400, len(rows)
    assert columns == ["event_id_cnty", "f"], columns
    stats = cache["_stats"]
    assert stats["verifications"] == 400, stats
    assert stats["max_verification_cost"] == 1.0, stats
    assert 0 < stats["cost"] < 1.0, stats


def check_ceiling_can_be_disabled():
    with _MeteredLLM(cost_per_call=0.05) as llm:
        rows, _, _ = _run(_BROAD, max_verification_cost=0)
    assert len(rows) == 400, len(rows)
    assert llm.calls == 400, llm.calls  # $20 spent, no ceiling to stop it


def check_call_ceiling_applies_end_to_end():
    with _MeteredLLM(cost_per_call=0.0) as llm:
        try:
            _run(_BROAD, max_verification_cost=0, max_verification_calls=30)
        except SUQLCostLimitExceeded as e:
            assert "call limit reached" in str(e), str(e)
            assert llm.calls <= 31, llm.calls
            return
    raise AssertionError("expected SUQLCostLimitExceeded")


def check_where_clause_verification_is_also_capped():
    """The ceiling must cover the retriever-backed WHERE path, not just
    projection predicates."""
    sql = (
        "SELECT event_id_cnty FROM events "
        "WHERE country = 'Ukraine' AND year = 2022 AND answer(notes, '{}') = 'Yes'"
    ).format(QUESTION)
    with _MeteredLLM(cost_per_call=0.01) as llm:
        try:
            _run(sql, max_verification_cost=0.05)
        except SUQLCostLimitExceeded:
            assert llm.calls * 0.01 <= 0.05 + 0.01, llm.calls
            return
    raise AssertionError("expected SUQLCostLimitExceeded")


def check_the_default_ceiling_is_active_without_being_asked():
    """A caller who passes nothing still gets a ceiling - that is the whole
    point, since the runaway happened with no ceiling configured."""
    with _MeteredLLM(cost_per_call=0.5) as llm:
        try:
            _run(_BROAD)
        except SUQLCostLimitExceeded as e:
            assert "1.00" in str(e), str(e)
            assert llm.calls <= 3, llm.calls
            return
    raise AssertionError("expected the default $1.00 ceiling to apply")


INTEGRATION_CHECKS = [
    check_runaway_query_is_refused_cheaply,
    check_affordable_query_completes_and_reports_spend,
    check_ceiling_can_be_disabled,
    check_call_ceiling_applies_end_to_end,
    check_where_clause_verification_is_also_capped,
    check_the_default_ceiling_is_active_without_being_asked,
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

    print("\n--- integration (real acled DB, metered fake LLM) ---")
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
