"""Offline tests: a scripted fake LLM drives the real graph, real tools and a temp copy
of the database. No API key needed.

Run:  python -m tests.test_graph        (or: pytest)
Needs data/rec_center.db first:  python data/generate_data.py
"""
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from agent import db
from agent.graph import build_graph
from agent.runner import ask, config_for, pending_interrupt, resume, stream_events
from agent.state import Evaluation, Plan


# ----------------------------------------------------------------- helpers
class ScriptedLLM:
    """Pops pre-written replies from one queue per kind of LLM call."""

    def __init__(self, plans=(), tool_msgs=(), evals=(), finals=()):
        self.q = {Plan: list(plans), Evaluation: list(evals), "tools": list(tool_msgs), "final": list(finals)}

    def with_structured_output(self, schema, **kw):
        return RunnableLambda(lambda _: self.q[schema].pop(0))

    def bind_tools(self, tools, **kw):
        return RunnableLambda(lambda _: self.q["tools"].pop(0))

    def invoke(self, _):
        return AIMessage(self.q["final"].pop(0))

    def all_used(self):
        return all(not v for v in self.q.values())


_ids = iter(range(10_000))


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call{next(_ids)}"}])


def temp_db():
    """Copy the generated DB so tests never change your real data."""
    src = db.DEFAULT_DB
    assert src.exists(), "Run `python data/generate_data.py` first."
    tmp = Path(tempfile.mkdtemp()) / "rec_center.db"
    shutil.copy(src, tmp)
    os.environ["REC_DB_PATH"] = str(tmp)
    os.environ["CHAOS_FAILURE_RATE"] = "0"
    return tmp


def run(graph, inputs, config):
    events = list(stream_events(graph, inputs, config))
    for kind, payload in events:
        text = str(payload).replace("\n", " | ")
        print(f"   {kind:14s} {text[:110]}")
    return events


def kinds(events):
    return [k for k, _ in events]


# ------------------------------------------------------------------- tests
def test_retry_with_feedback_and_sql_self_fix():
    temp_db()
    llm = ScriptedLLM(
        plans=[Plan(steps=["Find the busiest weekday hour"])],
        tool_msgs=[
            AIMessage("It's busy in the evening."),                    # attempt 1: vague
            call("run_sql", query="SELECT hour FROM checkins"),         # attempt 2: bad SQL
            call("run_sql", query="SELECT strftime('%H',checkin_time) h, COUNT(*) c FROM checkins "
                                  "GROUP BY h ORDER BY c DESC LIMIT 1"),
            AIMessage("Busiest hour is 18:00."),
        ],
        evals=[Evaluation(ok=False, feedback="No numbers."), Evaluation(ok=True, feedback="")],
        finals=["The busiest hour is 6pm."],
    )
    graph = build_graph(llm, checkpointer=InMemorySaver())
    ev = run(graph, ask("Busiest hour?"), config_for("t1"))
    assert kinds(ev).count("rejected") == 1
    assert any(k == "tool_result" and "ERROR" in p for k, p in ev)
    assert kinds(ev)[-1] == "answer" and llm.all_used()


def test_replan_after_repeated_failure():
    temp_db()
    llm = ScriptedLLM(
        plans=[Plan(steps=["Get revenue by facility"]),                  # impossible: no revenue data
               Plan(steps=["Count check-ins by facility instead"])],     # replanner's new approach
        tool_msgs=[
            AIMessage("There is no revenue table."),
            AIMessage("Still cannot find revenue."),
            call("run_sql", query="SELECT facility, COUNT(*) n FROM checkins GROUP BY facility ORDER BY n DESC"),
            AIMessage("Weight Room leads with the most check-ins."),
        ],
        evals=[Evaluation(ok=False, feedback="No revenue numbers."),
               Evaluation(ok=False, feedback="Still no numbers."),
               Evaluation(ok=True, feedback="")],
        finals=["Revenue isn't tracked; by check-ins, Weight Room is busiest."],
    )
    graph = build_graph(llm, checkpointer=InMemorySaver())
    ev = run(graph, ask("Which facility makes the most money?"), config_for("t2"))
    assert "replan_needed" in kinds(ev) and "replanned" in kinds(ev)
    assert kinds(ev)[-1] == "answer" and llm.all_used()


def test_staffing_proposal_interrupt_and_approval():
    tmp = temp_db()
    before = sqlite3.connect(tmp).execute("SELECT COUNT(*) FROM shifts").fetchone()[0]
    llm = ScriptedLLM(
        plans=[Plan(steps=["Find staffing gaps for the next 7 days", "Propose shifts to fill the gaps"])],
        tool_msgs=[
            call("find_staffing_gaps", days=7), AIMessage("Found understaffed evening hours."),
            call("propose_shifts", days=7), AIMessage("Drafted shifts, pending approval."),
        ],
        evals=[Evaluation(ok=True, feedback=""), Evaluation(ok=True, feedback="")],
        finals=["Evenings are short-staffed; I drafted shifts for your approval."],
    )
    graph = build_graph(llm, checkpointer=InMemorySaver())
    cfg = config_for("t3")
    ev = run(graph, ask("Are we staffed for next week?"), cfg)
    assert kinds(ev)[-1] == "interrupt"

    pending = pending_interrupt(graph, cfg)                  # the paused run is saved
    ids = [p["proposal_id"] for p in pending["proposals"]]
    assert len(ids) >= 2

    ev = run(graph, resume({"action": "approve", "approved_ids": ids[:2]}), cfg)
    assert kinds(ev) == ["approval_done"]
    after = sqlite3.connect(tmp).execute("SELECT COUNT(*) FROM shifts").fetchone()[0]
    assert after == before + 2
    assert pending_interrupt(graph, cfg) is None and llm.all_used()


def test_transient_db_errors_are_retried():
    temp_db()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise db.TransientDBError("simulated outage")
        return "ok"

    assert db.with_db_retries(flaky, base_delay=0.01) == ("ok", 2)


class Boom(Exception):
    pass


def _members_llm(fail_times):
    """Scripted LLM whose synthesizer call raises `fail_times` times first."""
    class Flaky(ScriptedLLM):
        failures = {"left": fail_times}

        def invoke(self, x):
            if self.failures["left"] > 0:
                self.failures["left"] -= 1
                raise Boom("API connection dropped")
            return super().invoke(x)

    return Flaky(
        plans=[Plan(steps=["Count members"])],
        tool_msgs=[call("run_sql", query="SELECT COUNT(*) FROM members"), AIMessage("1500 members.")],
        evals=[Evaluation(ok=True, feedback="")],
        finals=["There are 1,500 members."],
    )


def test_node_retry_policy_recovers_from_api_error():
    temp_db()
    llm = _members_llm(fail_times=1)            # fails once; RetryPolicy re-runs the node
    graph = build_graph(llm, checkpointer=InMemorySaver())
    ev = run(graph, ask("How many members?"), config_for("t4"))
    assert kinds(ev)[-1] == "answer" and llm.all_used()


def test_crash_resume_from_checkpoint():
    temp_db()
    llm = _members_llm(fail_times=3)            # more failures than RetryPolicy allows -> crash
    graph = build_graph(llm, checkpointer=InMemorySaver())
    cfg = config_for("t5")
    try:
        run(graph, ask("How many members?"), cfg)
        raise AssertionError("expected crash")
    except Boom:
        print("   (crashed)")
    assert graph.get_state(cfg).next == ("synthesizer",)    # progress was saved

    ev = run(graph, None, cfg)          # resume: no re-planning, no re-querying the DB
    assert kinds(ev) == ["answer"] and llm.all_used()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        print(f"\n{t.__name__}")
        t()
        print("   PASS")
    print(f"\nAll {len(tests)} tests passed.")
