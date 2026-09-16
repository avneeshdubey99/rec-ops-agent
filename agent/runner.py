"""Helpers shared by the CLI and the Streamlit app: run the graph and turn LangGraph's
raw stream into simple, readable events."""
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command


def new_thread_id() -> str:
    return uuid.uuid4().hex[:8]


def config_for(thread_id: str) -> dict:
    # The thread_id tells the checkpointer which conversation to save/load
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


def ask(question: str) -> dict:
    return {"question": question, "messages": [HumanMessage(question)]}


def resume(decision: dict) -> Command:
    return Command(resume=decision)


def pending_interrupt(graph, config):
    """The value passed to interrupt() if the thread is paused, else None."""
    for task in graph.get_state(config).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def unfinished(graph, config) -> tuple[str, ...]:
    """Nodes still waiting to run (non-empty after a crash or while paused)."""
    return graph.get_state(config).next


def stream_events(graph, inputs, config):
    """Run the graph (inputs=None resumes a crashed run) and yield (kind, payload)."""
    for namespace, update in graph.stream(inputs, config, stream_mode="updates", subgraphs=True):
        for node, up in update.items():
            if node == "__interrupt__":
                yield "interrupt", up[0].value
                continue
            if not up:
                continue
            if namespace:  # events from inside the executor subgraph
                if node == "agent":
                    for tc in getattr(up["messages"][-1], "tool_calls", []) or []:
                        yield "tool_call", f"{tc['name']}({_fmt_args(tc['args'])})"
                elif node == "tools":
                    for m in up["messages"]:
                        yield "tool_result", f"{m.name}: {m.text}"
                continue
            if node == "planner":
                yield "plan", up["plan"]
            elif node == "executor":
                yield "draft", f"attempt {up['attempts']}: {up['draft']}"
            elif node == "evaluator":
                if up.get("needs_replan"):
                    yield "replan_needed", up["last_failure"]["feedback"]
                elif up.get("feedback"):
                    yield "rejected", up["feedback"]
                else:
                    last = up["step_results"][-1]
                    yield ("accepted" if last["ok"] else "step_failed"), last["step"]
            elif node == "replanner":
                yield "replanned", up["plan"]
            elif node == "synthesizer":
                yield "answer", up["final_answer"]
            elif node == "approval" and up.get("final_answer"):
                yield "approval_done", up["final_answer"]


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
