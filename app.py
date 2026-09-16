"""Streamlit app: chat with the agent, watch its reasoning, approve shifts, explore data.

Run:  streamlit run app.py
"""
import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import staffing
from agent.db import connect
from agent.graph import build_graph
from agent.llm import missing_key
from agent.memory import get_checkpointer
from agent.runner import (ask, config_for, new_thread_id, pending_interrupt, resume,
                          stream_events, unfinished)

load_dotenv()
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Rec Center Ops Agent", page_icon="🏋️", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph(checkpointer=get_checkpointer())


# ------------------------------------------------------------------ sidebar
if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

with st.sidebar:
    st.title("🏋️ Rec Ops Agent")
    st.caption("Texas A&M Rec Center operations assistant. All data is fake.")
    thread = st.text_input("Conversation id", st.session_state.thread_id,
                           help="Paste an old id to reopen that conversation.")
    st.session_state.thread_id = thread.strip() or st.session_state.thread_id
    if st.button("➕ New conversation", width="stretch"):
        st.session_state.thread_id = new_thread_id()
        st.rerun()

    st.divider()
    st.subheader("Failure simulation")
    chaos = st.slider("DB failure rate", 0.0, 0.8, float(os.getenv("CHAOS_FAILURE_RATE", "0") or 0), 0.1,
                      help="Randomly fail database calls to watch the agent retry and recover.")
    os.environ["CHAOS_FAILURE_RATE"] = str(chaos)

    st.divider()
    st.markdown("**Try asking**")
    examples = [
        "Which hours next week will be understaffed? Propose shifts to fix it.",
        "What's the busiest hour on weekdays vs weekends?",
        "Which facility had the most check-ins in the last 7 days?",
        "How many lockers expire in the next 30 days?",
    ]
    for ex in examples:
        if st.button(ex, width="stretch"):
            st.session_state.queued_question = ex

if missing_key():
    st.error(f"{missing_key()} is missing. Copy `.env.example` to `.env`, add your key, and restart.")
    st.stop()

try:
    graph = get_graph()
except FileNotFoundError as err:
    st.error(f"{err}")
    st.stop()

config = config_for(st.session_state.thread_id)


# ----------------------------------------------------------------- helpers
LABELS = {
    "plan": "🗺️ **Plan**", "replanned": "🔁 **Re-planned**", "tool_call": "🔧 Tool call",
    "draft": "📝 Draft", "accepted": "✅ Step accepted", "rejected": "↩️ Rejected, retrying",
    "replan_needed": "⚠️ Step kept failing, re-planning", "step_failed": "❌ Step failed, moving on",
}


def run_and_render(inputs):
    """Stream the graph inside a status box. Returns the final answer text, if any."""
    answer = None
    with st.status("Working…", expanded=True) as status:
        for kind, payload in stream_events(graph, inputs, config):
            if kind in ("plan", "replanned"):
                st.markdown(LABELS[kind] + "\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(payload, 1)))
            elif kind == "tool_call":
                st.markdown(f"{LABELS[kind]}: `{payload}`")
            elif kind == "tool_result":
                name, _, body = payload.partition(": ")
                if "[recovered after" in body:
                    st.warning(f"Database hiccup in `{name}`, recovered by retrying.")
                with st.expander(f"Result from `{name}`" + (" (error)" if body.startswith("ERROR") else "")):
                    st.code(body[:3000])
            elif kind in ("draft",):
                st.caption(f"{LABELS[kind]} {payload[:300]}")
            elif kind in LABELS:
                st.markdown(f"{LABELS[kind]}: {payload}")
            elif kind in ("answer", "approval_done"):
                answer = payload
            elif kind == "interrupt":
                status.update(label="Waiting for manager approval", state="complete", expanded=False)
                return answer
        status.update(label="Done", state="complete", expanded=False)
    return answer


def safe_run(inputs):
    try:
        run_and_render(inputs)
    except Exception as err:  # noqa: BLE001
        st.error(f"The run stopped: {err}. Progress is saved; use **Resume** to continue.")


# -------------------------------------------------------------------- tabs
chat_tab, data_tab = st.tabs(["💬 Assistant", "📊 Staffing outlook"])

with chat_tab:
    snapshot = graph.get_state(config)
    for msg in snapshot.values.get("messages", []):
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.text)

    pending = pending_interrupt(graph, config)
    stuck = unfinished(graph, config)

    if pending:
        props = pd.DataFrame(pending["proposals"])
        with st.chat_message("assistant"):
            st.markdown(f"### 🗓️ {len(props)} draft shifts need your approval")
            st.caption(pending["question"])
            props.insert(0, "approve", True)
            props["shift"] = props["start_hour"].astype(str) + ":00–" + props["end_hour"].astype(str) + ":00"
            edited = st.data_editor(
                props[["approve", "proposal_id", "shift_date", "shift", "name", "role", "reason"]],
                hide_index=True, width="stretch", disabled=["proposal_id", "shift_date", "shift",
                                                                      "name", "role", "reason"],
                column_config={"approve": st.column_config.CheckboxColumn("Approve"),
                               "proposal_id": st.column_config.NumberColumn("ID")},
                key=f"editor-{pending['batch_id']}",
            )
            note = st.text_input("Note for the record (optional)", key=f"note-{pending['batch_id']}")
            chosen = edited.loc[edited["approve"], "proposal_id"].astype(int).tolist()
            c1, c2 = st.columns(2)
            if c1.button(f"✅ Publish {len(chosen)} selected", type="primary", width="stretch",
                         disabled=not chosen):
                safe_run(resume({"action": "approve", "approved_ids": chosen, "note": note}))
                st.rerun()
            if c2.button("🚫 Reject all", width="stretch"):
                safe_run(resume({"action": "reject", "note": note}))
                st.rerun()
    elif stuck:
        st.warning(f"An earlier run stopped before finishing (next step: `{stuck[0]}`).")
        if st.button("▶️ Resume from last checkpoint", type="primary"):
            safe_run(None)
            st.rerun()

    question = st.chat_input("Ask about traffic, staffing, members, lockers…", disabled=bool(pending))
    question = question or st.session_state.pop("queued_question", None)
    if question and not pending:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            safe_run(ask(question))
        st.rerun()

with data_tab:
    st.subheader("Next 7 days: demand vs. schedule")
    st.caption(f"Rule: 1 staff per {staffing.CHECKINS_PER_STAFF} expected check-ins per hour, "
               f"minimum {staffing.MIN_STAFF}, staffed hours 6:00–22:00.")
    try:
        cov = staffing.coverage(None, 7)
        cov["when"] = pd.to_datetime(cov["date"]) + pd.to_timedelta(cov["hour"], unit="h")
        gaps = cov[cov["shortfall"] > 0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected check-ins", f"{cov['expected_checkins'].sum():,.0f}")
        m2.metric("Understaffed hours", len(gaps))
        m3.metric("Staff-hours short", int(gaps["shortfall"].sum()))
        st.line_chart(cov.set_index("when")[["required_staff", "scheduled_staff"]])

        st.subheader("Understaffed hours")
        if gaps.empty:
            st.success("No gaps. Every open hour meets the staffing rule.")
        else:
            st.dataframe(gaps[["date", "hour", "expected_checkins", "required_staff", "scheduled_staff",
                               "shortfall"]], hide_index=True, width="stretch")

        st.subheader("Average check-ins by hour (history)")
        with connect() as con:
            hist = pd.read_sql_query(
                """SELECT CAST(strftime('%H', checkin_time) AS INTEGER) AS hour,
                          CASE WHEN strftime('%w', checkin_time) IN ('0','6') THEN 'Weekend' ELSE 'Weekday' END AS kind,
                          COUNT(*) * 1.0 / COUNT(DISTINCT date(checkin_time)) AS avg_checkins
                   FROM checkins GROUP BY hour, kind""", con)
        st.bar_chart(hist.pivot(index="hour", columns="kind", values="avg_checkins"))
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load data: {err}")
