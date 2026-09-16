"""The full agent graph.

    START -> planner -> executor -> evaluator --(retry / next step)--> executor
                                      |  \\
                                      |   +--(step failed every attempt)--> replanner -> executor
                                      |
                                      +--(all steps done)--> synthesizer --(shifts drafted?)--> approval -> END
                                                                   |
                                                                   +--(no)--> END

Failure recovery, from small to large:
  1. Tool level:   transient DB errors retried with backoff; other errors returned as text.
  2. Executor:     LLM reads tool errors and fixes its own calls; recursion limit stops loops.
  3. Evaluator:    rejects weak results and retries the step with feedback.
  4. Replanner:    if a step keeps failing, rewrite the rest of the plan another way.
  5. Node level:   RetryPolicy re-runs a node on API errors (timeouts, rate limits).
  6. Graph level:  checkpointer lets a crashed run resume from the last completed node.
Human-in-the-loop: `interrupt()` pauses the graph until a manager approves the shifts.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from agent import staffing
from agent.executor import build_executor
from agent.llm import get_llm
from agent.state import AgentState, Evaluation, Plan
from agent.tools import get_schema

MAX_ATTEMPTS_PER_STEP = 2
MAX_REPLANS = 1
EXECUTOR_RECURSION_LIMIT = 16          # max agent<->tools hops for one step
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0)

PLANNER_PROMPT = """You plan analyses for the Texas A&M Rec Center operations team.
Break the manager's latest question into 1-4 concrete steps. Simple questions need 1 step.

Available tools for the analyst: run_sql (database queries), forecast_traffic,
find_staffing_gaps, propose_shifts (drafts shifts for manager approval).
For staffing questions a typical plan is: find staffing gaps -> propose shifts.
Do not add a "summarize" or "get approval" step; those happen automatically.
Use the recent conversation to resolve follow-ups like "what about next week?".

Schema:
{schema}"""

REPLAN_PROMPT = """A step in the analysis plan failed repeatedly. Write 1-3 replacement steps
that reach the same goal a DIFFERENT way (simpler query, different tool, smaller date range).
If the goal truly cannot be reached, return one step that gathers whatever partial
information is possible.

Schema:
{schema}"""

EVALUATOR_PROMPT = """You check an analyst's work. Given a step and the analyst's result,
decide if the result truly completes the step using real data. Reject vague answers,
unresolved ERROR messages, or results that answer a different question.
A result saying there is no data / no gaps is acceptable if it came from a tool."""

SYNTH_PROMPT = """You are replying to a Rec Center manager. Using ONLY the step results,
answer their question clearly and concisely with the key numbers. Use short bullet
points when listing several items. If a step FAILED, say what could not be determined.
If shifts were drafted, say they are pending the manager's approval (do not list every
shift; the approval screen shows them)."""


def build_graph(llm=None, checkpointer=None):
    llm = llm or get_llm()
    planner_llm = llm.with_structured_output(Plan)
    evaluator_llm = llm.with_structured_output(Evaluation)
    schema = get_schema.invoke({})
    executor = build_executor(llm, schema)

    # ------------------------------------------------------------------ nodes
    def planner(state: AgentState):
        recent = state.get("messages", [])[-7:-1]   # earlier turns, excluding this question
        history = "\n".join(f"{m.type}: {m.text[:300]}" for m in recent)
        prompt = f"Recent conversation:\n{history}\n\n" if history else ""
        plan = planner_llm.invoke([
            SystemMessage(PLANNER_PROMPT.format(schema=schema)),
            HumanMessage(prompt + f"Latest question: {state['question']}"),
        ])
        return {
            "plan": plan.steps or [state["question"]],
            "current_step": 0, "attempts": 0, "feedback": "", "draft": "",
            "step_results": [], "needs_replan": False, "last_failure": None,
            "replans": 0, "proposal_batch": None, "final_answer": "",
        }

    def execute(state: AgentState):
        step = state["plan"][state["current_step"]]
        done = "\n".join(f"- {r['step']}: {r['result']}" for r in state["step_results"] if r["ok"])
        task = f"Overall question: {state['question']}\n"
        if done:
            task += f"Results from earlier steps:\n{done}\n"
        task += f"\nYour step: {step}"
        if state.get("feedback"):
            task += f"\n\nYour previous attempt was rejected: {state['feedback']}\nFix this."

        batch = state.get("proposal_batch")
        try:
            out = executor.invoke({"messages": [HumanMessage(task)]},
                                  {"recursion_limit": EXECUTOR_RECURSION_LIMIT})
            draft = out["messages"][-1].text   # .text flattens Claude's content blocks
            # Did the executor draft shifts? The tool's artifact carries the batch id.
            for m in out["messages"]:
                if isinstance(m, ToolMessage) and m.name == "propose_shifts" and m.artifact:
                    batch = m.artifact.get("batch_id") or batch
        except GraphRecursionError:
            draft = "ERROR: gave up after too many tool calls without finishing the step."
        return {"draft": draft, "attempts": state["attempts"] + 1, "proposal_batch": batch}

    def evaluate(state: AgentState):
        step = state["plan"][state["current_step"]]
        verdict = evaluator_llm.invoke([
            SystemMessage(EVALUATOR_PROMPT),
            HumanMessage(f"Step: {step}\n\nResult: {state['draft']}"),
        ])
        if verdict.ok:
            return _advance(state, step, ok=True)
        feedback = verdict.feedback or "The result did not complete the step."
        if state["attempts"] < MAX_ATTEMPTS_PER_STEP:
            return {"feedback": feedback, "needs_replan": False}           # retry same step
        if state.get("replans", 0) < MAX_REPLANS:
            return {"feedback": "", "needs_replan": True,                   # re-plan
                    "last_failure": {"step": step, "result": state["draft"], "feedback": feedback}}
        return _advance(state, step, ok=False)                              # give up on step

    def replan(state: AgentState):
        f = state["last_failure"]
        done = "\n".join(f"- {r['step']}: {r['result']}" for r in state["step_results"])
        new = planner_llm.invoke([
            SystemMessage(REPLAN_PROMPT.format(schema=schema)),
            HumanMessage(
                f"Question: {state['question']}\nCompleted steps:\n{done or '(none)'}\n\n"
                f"Failed step: {f['step']}\nLast result: {f['result']}\nWhy it failed: {f['feedback']}"
            ),
        ])
        i = state["current_step"]
        return {
            "plan": state["plan"][:i] + (new.steps or []),
            "step_results": state["step_results"] + [
                {"step": f["step"], "result": f"FAILED ({f['feedback']}); replaced by a new approach", "ok": False}],
            "replans": state.get("replans", 0) + 1,
            "needs_replan": False, "attempts": 0, "feedback": "",
        }

    def synthesize(state: AgentState):
        results = "\n".join(
            f"- {r['step']} [{'ok' if r['ok'] else 'FAILED'}]: {r['result']}" for r in state["step_results"]
        )
        if state.get("proposal_batch"):
            results += "\n(Shift proposals were drafted and are pending manager approval.)"
        reply = llm.invoke([
            SystemMessage(SYNTH_PROMPT),
            HumanMessage(f"Question: {state['question']}\n\nStep results:\n{results}"),
        ])
        return {"final_answer": reply.text, "messages": [AIMessage(reply.text)]}

    def approval(state: AgentState):
        batch = state["proposal_batch"]
        proposals = staffing.load_proposals(batch)
        if not proposals:
            return {"proposal_batch": None}
        # Pause here. The run is saved by the checkpointer; it continues when someone calls
        # graph.invoke(Command(resume={...})) with the manager's decision.
        decision = interrupt({
            "type": "shift_approval",
            "batch_id": batch,
            "proposals": proposals,
            "question": "Approve these draft shifts? Approved shifts are added to the live schedule.",
        })
        action = (decision or {}).get("action", "reject")
        if action == "approve":
            ids = decision.get("approved_ids")
            ids = [p["proposal_id"] for p in proposals] if ids is None else ids
        else:
            ids = []
        published, rejected = staffing.apply_decision(batch, ids)
        text = f"Manager decision recorded: published {published} shift(s) to the schedule"
        text += f", rejected {rejected}." if rejected else "."
        if decision and decision.get("note"):
            text += f" Note: {decision['note']}"
        return {"proposal_batch": None, "messages": [AIMessage(text)], "final_answer": text}

    # ---------------------------------------------------------------- routing
    def route_after_evaluate(state: AgentState):
        if state.get("needs_replan"):
            return "replanner"
        if state.get("feedback"):
            return "executor"
        if state["current_step"] < len(state["plan"]):
            return "executor"
        return "synthesizer"

    def route_after_replan(state: AgentState):
        return "executor" if state["current_step"] < len(state["plan"]) else "synthesizer"

    def route_after_synthesize(state: AgentState):
        return "approval" if state.get("proposal_batch") else END

    # ------------------------------------------------------------------ wire
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner, retry_policy=LLM_RETRY)
    builder.add_node("executor", execute, retry_policy=LLM_RETRY)
    builder.add_node("evaluator", evaluate, retry_policy=LLM_RETRY)
    builder.add_node("replanner", replan, retry_policy=LLM_RETRY)
    builder.add_node("synthesizer", synthesize, retry_policy=LLM_RETRY)
    builder.add_node("approval", approval)          # no auto-retry: it writes to the schedule

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_conditional_edges("evaluator", route_after_evaluate, ["executor", "replanner", "synthesizer"])
    builder.add_conditional_edges("replanner", route_after_replan, ["executor", "synthesizer"])
    builder.add_conditional_edges("synthesizer", route_after_synthesize, ["approval", END])
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=checkpointer)


def _advance(state: AgentState, step: str, ok: bool):
    record = {"step": step, "result": state["draft"], "ok": ok}
    return {
        "step_results": state["step_results"] + [record],
        "current_step": state["current_step"] + 1,
        "attempts": 0, "feedback": "", "needs_replan": False,
    }


if __name__ == "__main__":
    # `python -m agent.graph` prints a Mermaid diagram (paste into https://mermaid.live)
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class _Stub(FakeListChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def with_structured_output(self, schema, **kw):
            return self

    print(build_graph(_Stub(responses=["x"])).get_graph().draw_mermaid())
