"""Tools the agent can call. Each docstring is what the LLM reads to decide when and
how to use the tool, so keep them clear.

Recovery built in:
- Transient DB failures are retried with backoff (see agent/db.py).
- Any other error is returned as "ERROR: ..." text instead of crashing, so the LLM can
  read it and try something different (fix the SQL, change the date, etc.).
"""
import functools
import sqlite3
from datetime import date

from langchain_core.tools import tool

from agent import staffing
from agent.db import connect, with_db_retries

MAX_ROWS = 50


def _recovered(retries: int) -> str:
    return f"\n[recovered after {retries} retr{'y' if retries == 1 else 'ies'}]" if retries else ""


def safe_tool(fn):
    """Turn exceptions into readable error text for the LLM."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as err:  # noqa: BLE001
            return f"ERROR: {type(err).__name__}: {err}"
    return wrapper


def _clamp_days(days: int) -> int:
    return max(1, min(int(days), 14))


@tool
@safe_tool
def get_schema() -> str:
    """Return the database tables and their columns."""
    def q():
        with connect() as con:
            return con.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
    rows, retries = with_db_retries(q)
    return "\n\n".join(r[0] for r in rows) + _recovered(retries)


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQLite SELECT query against the Rec Center database and return up
    to 50 rows. Use SQLite syntax, e.g. strftime('%H', checkin_time), date('now','+1 day')."""
    if not query.strip().lower().startswith(("select", "with")):
        return "ERROR: only SELECT queries are allowed."

    def q():
        with connect() as con:
            cur = con.execute(query)
            return [c[0] for c in cur.description], cur.fetchmany(MAX_ROWS + 1)
    try:
        (cols, rows), retries = with_db_retries(q)
    except sqlite3.Error as err:
        return f"ERROR: {err}. Check the schema and fix the query."
    except Exception as err:  # noqa: BLE001
        return f"ERROR: {err}"
    lines = [" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows[:MAX_ROWS]]
    if len(rows) > MAX_ROWS:
        lines.append(f"... (truncated to {MAX_ROWS} rows; aggregate or add LIMIT)")
    return "\n".join(lines) + _recovered(retries)


@tool
@safe_tool
def forecast_traffic(start_date: str | None = None, days: int = 7) -> str:
    """Forecast expected hourly check-ins for upcoming days from recent history.
    start_date: 'YYYY-MM-DD' (default: tomorrow). days: 1-14.
    Returns each day's total and peak hour, plus the busiest hours overall."""
    days = _clamp_days(days)
    fc, retries = with_db_retries(lambda: staffing.forecast_traffic(start_date, days))
    lines = ["date | weekday | expected_total | peak_hour | peak_checkins"]
    for d, grp in fc.groupby("date"):
        peak = grp.loc[grp["expected_checkins"].idxmax()]
        wd = date.fromisoformat(d).strftime("%a")
        lines.append(f"{d} | {wd} | {grp['expected_checkins'].sum():.0f} | "
                     f"{int(peak['hour'])}:00 | {peak['expected_checkins']:.0f}")
    top = fc.nlargest(5, "expected_checkins")
    lines.append("\nBusiest 5 hours: " + "; ".join(
        f"{r.date} {r.hour}:00 (~{r.expected_checkins:.0f})" for r in top.itertuples()))
    return "\n".join(lines) + _recovered(retries)


@tool
@safe_tool
def find_staffing_gaps(start_date: str | None = None, days: int = 7) -> str:
    """Compare forecast demand with the current shift schedule and list understaffed hours.
    Rule: 1 staff per 30 expected check-ins, minimum 2, staffed hours 6:00-22:00.
    start_date: 'YYYY-MM-DD' (default: tomorrow). days: 1-14."""
    days = _clamp_days(days)
    gaps, retries = with_db_retries(lambda: staffing.staffing_gaps(start_date, days))
    if gaps.empty:
        return "No staffing gaps: every open hour meets the staffing rule." + _recovered(retries)
    lines = [f"{len(gaps)} understaffed hours, total shortfall {int(gaps['shortfall'].sum())} staff-hours.",
             "date | hour | expected_checkins | required | scheduled | short"]
    for r in gaps.head(40).itertuples():
        lines.append(f"{r.date} | {r.hour}:00 | {r.expected_checkins:.0f} | "
                     f"{r.required_staff} | {r.scheduled_staff} | {r.shortfall}")
    if len(gaps) > 40:
        lines.append(f"... {len(gaps) - 40} more")
    return "\n".join(lines) + _recovered(retries)


@tool(response_format="content_and_artifact")
def propose_shifts(start_date: str | None = None, days: int = 7) -> tuple[str, dict]:
    """Draft extra 4-hour shifts that close staffing gaps, respecting weekly hour limits.
    The drafts are saved as PENDING and must be approved by a manager before they are
    published; you cannot publish them yourself.
    start_date: 'YYYY-MM-DD' (default: tomorrow). days: 1-14."""
    try:
        (batch_id, proposals, unfilled), retries = with_db_retries(
            lambda: staffing.propose_shifts(start_date, _clamp_days(days)))
    except Exception as err:  # noqa: BLE001
        return f"ERROR: {type(err).__name__}: {err}", {}
    if not proposals:
        msg = "No shifts proposed: no gaps found." if not unfilled else \
            f"Could not propose shifts: no available staff for {len(unfilled)} gap blocks."
        return msg + _recovered(retries), {}
    lines = [f"Drafted {len(proposals)} shifts (batch {batch_id}), PENDING manager approval:",
             "date | shift | name | role | reason"]
    for p in proposals:
        lines.append(f"{p['shift_date']} | {p['start_hour']}:00-{p['end_hour']}:00 | "
                     f"{p['name']} | {p['role']} | {p['reason']}")
    if unfilled:
        lines.append(f"Still unfilled (no available staff): " + "; ".join(
            f"{u['shift_date']} {u['start_hour']}:00 needs {u['missing']}" for u in unfilled))
    return "\n".join(lines) + _recovered(retries), {"batch_id": batch_id, "count": len(proposals)}


TOOLS = [get_schema, run_sql, forecast_traffic, find_staffing_gaps, propose_shifts]
