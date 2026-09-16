"""Command-line version of the agent.

Run:
  python main.py                    new conversation
  python main.py --thread abc123    continue (or resume) a saved conversation
  python main.py --chaos 0.3        randomly fail 30% of DB calls to watch recovery
"""
import argparse
import logging
import os

from agent.graph import build_graph
from agent.llm import missing_key
from agent.memory import get_checkpointer
from agent.runner import (ask, config_for, new_thread_id, pending_interrupt, resume,
                          stream_events, unfinished)

ICONS = {"plan": "PLAN", "replanned": "RE-PLAN", "tool_call": "  -> tool", "tool_result": "  <- result",
         "draft": "  draft", "accepted": "  OK", "rejected": "  REJECTED (retrying)",
         "replan_needed": "  STEP FAILED (re-planning)", "step_failed": "  STEP FAILED (moving on)"}


def show(events):
    """Print events; return the interrupt payload if the graph paused."""
    for kind, payload in events:
        if kind in ("plan", "replanned"):
            print(f"\n[{ICONS[kind]}]")
            for i, step in enumerate(payload, 1):
                print(f"   {i}. {step}")
        elif kind == "tool_result":
            first = payload.splitlines()
            extra = f" ... (+{len(first) - 1} lines)" if len(first) > 1 else ""
            print(f"{ICONS[kind]} {first[0][:140]}{extra}")
            if "[recovered after" in payload:
                print("     (transient failure recovered by retry)")
        elif kind in ICONS:
            print(f"{ICONS[kind]}: {str(payload)[:160]}")
        elif kind in ("answer", "approval_done"):
            print(f"\nAgent: {payload}\n")
        elif kind == "interrupt":
            return payload
    return None


def approval_prompt(payload) -> dict:
    props = payload["proposals"]
    print("=" * 78)
    print(f"MANAGER APPROVAL NEEDED: {len(props)} draft shifts (batch {payload['batch_id']})")
    print("=" * 78)
    for p in props:
        print(f" #{p['proposal_id']:<4} {p['shift_date']} {p['start_hour']:>2}:00-{p['end_hour']}:00  "
              f"{p['name']:<16} {p['role']:<22} {p['reason']}")
    while True:
        choice = input("\n[a] approve all  [s] select ids  [r] reject all  [l] decide later: ").strip().lower()
        if choice == "a":
            return {"action": "approve"}
        if choice == "r":
            return {"action": "reject", "note": input("Reason (optional): ").strip()}
        if choice == "s":
            raw = input("Proposal ids to approve (comma-separated): ")
            valid = {p["proposal_id"] for p in props}
            ids = [int(x) for x in raw.replace(" ", "").split(",") if x.isdigit() and int(x) in valid]
            return {"action": "approve", "approved_ids": ids}
        if choice == "l":
            return {}


def run_until_done(graph, inputs, config):
    payload = show(stream_events(graph, inputs, config))
    while payload:
        decision = approval_prompt(payload)
        if not decision:
            print("Saved. Resume later with the same --thread id.\n")
            return
        payload = show(stream_events(graph, resume(decision), config))


def main():
    parser = argparse.ArgumentParser(description="Rec Center Ops Agent")
    parser.add_argument("--thread", help="conversation id to continue or resume")
    parser.add_argument("--chaos", type=float, default=None, help="simulated DB failure rate, e.g. 0.3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="     [retry] %(message)s")
    if args.chaos is not None:
        os.environ["CHAOS_FAILURE_RATE"] = str(args.chaos)
    if missing_key():
        raise SystemExit(f"{missing_key()} is missing. Copy .env.example to .env and add your key.")

    graph = build_graph(checkpointer=get_checkpointer())
    thread = args.thread or new_thread_id()
    config = config_for(thread)
    print(f"Rec Center Ops Agent  |  thread: {thread}  |  type 'quit' to exit")
    print(f"(Continue this conversation later with: python main.py --thread {thread})\n")

    # Recover anything left over from a previous session on this thread
    payload = pending_interrupt(graph, config)
    if payload:
        print("This conversation has shifts waiting for approval.")
        decision = approval_prompt(payload)
        if decision:
            run_until_done(graph, resume(decision), config)
    elif unfinished(graph, config):
        print(f"Found an unfinished run (next step: {unfinished(graph, config)[0]}).")
        if input("Resume it? [y/n]: ").strip().lower() == "y":
            run_until_done(graph, None, config)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", ""}:
            break
        try:
            run_until_done(graph, ask(question), config)
        except KeyboardInterrupt:
            print(f"\nStopped. Progress is saved; run `python main.py --thread {thread}` to resume.")
            break


if __name__ == "__main__":
    main()
