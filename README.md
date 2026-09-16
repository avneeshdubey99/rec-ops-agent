# Rec Center Operations Agent — project plan

**Use case:** LangGraph agent for Texas A&M Rec Center operations (dummy data): answers ops questions, forecasts busy hours, finds staffing gaps, drafts shifts, publishes only manager-approved shifts.

**Decisions:** OpenAI (gpt-4o-mini default) · code in `C:\Users\avnee\Documents\rec-ops-agent` · Streamlit app + CLI. The computer's shell can't reach the folder (Windows update issue), so code is built and tested in the cloud and saved into the folder.

**Status (2026-09-15): v1 complete, all phases built; 6 offline tests pass. Not yet run against the real OpenAI API.**

## Architecture
planner -> executor (agent<->tools subgraph) -> evaluator -> [retry | replanner | next step | synthesizer] -> approval (interrupt) -> END

- Tools: run_sql (read-only), forecast_traffic, find_staffing_gaps, propose_shifts (content_and_artifact → batch_id), get_schema
- Staffing math in `agent/staffing.py`: forecast = same weekday+hour mean over last 4 weeks; 1 staff/30 check-ins, min 2, hours 6–22, 4h shifts, weekly limits, no double-booking; proposals in `shift_proposals` table (pending/approved/rejected/superseded)
- Recovery layers: DB transient retries with backoff (+ CHAOS_FAILURE_RATE) · tool errors as text · evaluator retry with feedback (2 tries) · replanner (1 per question) · node RetryPolicy (3) · SqliteSaver checkpointer (`data/checkpoints.sqlite`) for memory + crash resume
- HITL: `interrupt()` in approval node; resume with `Command(resume={"action": "approve"|"reject", "approved_ids": [...], "note": ...})`
- UI: `app.py` (chat, live reasoning status, approval data_editor, chaos slider, staffing outlook charts); `main.py --thread ID --chaos 0.3`
- Tests: `python -m tests.test_graph` (scripted fake LLM, temp DB copy)

## Possible next steps
- First real run with an OpenAI key; tune prompts
- LangSmith tracing; push to GitHub with screenshots/GIF
- v2: "Aggie Campus Agent" router with an Orientation Week sub-graph
