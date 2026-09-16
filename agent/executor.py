"""The executor subgraph: an LLM <-> tools loop that completes ONE plan step.

    START -> agent --(wants a tool?)--> tools -> agent -> ... -> END
"""
from datetime import date

from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import TOOLS

EXECUTOR_PROMPT = """You are an operations analyst for the Texas A&M Rec Center (all data is fake).
Today is {today}. You complete ONE step of a larger plan.

Tools:
- run_sql: ad-hoc questions (counts, lists, history). Schema is below.
- forecast_traffic: expected check-ins for upcoming days.
- find_staffing_gaps: understaffed hours (forecast vs schedule).
- propose_shifts: draft extra shifts for gaps (saved as pending; a manager approves later).

Database schema:
{schema}

Rules:
- Use tools to get real numbers; never invent data.
- If a tool returns ERROR, read it and try again differently (fix SQL, change arguments).
- Never claim shifts are published; proposals await manager approval.
- Finish with a short factual result for your step, including key numbers."""


def build_executor(llm, schema: str):
    llm_with_tools = llm.bind_tools(TOOLS)
    system = SystemMessage(EXECUTOR_PROMPT.format(today=date.today().isoformat(), schema=schema))

    def agent(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")
    return builder.compile()
