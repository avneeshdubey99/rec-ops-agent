"""Graph state: the shared "notebook" every node reads from and writes to.

Nodes return only the fields they change. Fields with a reducer (like `messages`)
are merged; all other fields are overwritten.
"""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class StepResult(TypedDict):
    step: str
    result: str
    ok: bool


class AgentState(TypedDict, total=False):
    # Conversation history, kept across turns by the checkpointer (appended, not replaced)
    messages: Annotated[list[AnyMessage], add_messages]

    # --- per-question working memory (reset by the planner each turn) ---
    question: str
    plan: list[str]              # steps written by the planner / replanner
    current_step: int            # index into plan
    attempts: int                # tries on the current step
    draft: str                   # executor's latest result for the current step
    feedback: str                # evaluator's reason for rejecting a draft ("" = none)
    step_results: list[StepResult]
    needs_replan: bool           # set when a step failed every attempt
    last_failure: dict[str, Any] | None
    replans: int                 # how many times we re-planned this turn
    proposal_batch: str | None   # pending shift proposals waiting for approval
    final_answer: str


# Pydantic models force the LLM to reply in an exact shape (structured output)
class Plan(BaseModel):
    steps: list[str] = Field(
        description="1-4 short, concrete steps. Each is done with the available tools."
    )


class Evaluation(BaseModel):
    ok: bool = Field(description="True if the result actually completes the step with real data")
    feedback: str = Field(description="If not ok: what is wrong or missing. If ok: empty string")
