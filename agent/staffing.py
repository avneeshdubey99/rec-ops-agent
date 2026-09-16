"""Forecasting and staffing logic (plain Python, no LLM).

Keeping the math here, outside the agent, makes it testable and trustworthy:
the LLM decides WHAT to ask; this code computes the numbers.
"""
import math
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd

from agent.db import connect

OPEN_HOURS = range(6, 22)          # staffed hours: 6am-10pm
SHIFT_BLOCKS = (6, 10, 14, 18)     # 4-hour shift start times
SHIFT_LENGTH = 4
CHECKINS_PER_STAFF = 30            # one staff member per 30 check-ins/hour
MIN_STAFF = 2                      # never fewer than 2 people on the floor
ROLE_PRIORITY = {"Facility Supervisor": 0, "Front Desk": 1,
                 "Climbing Wall Attendant": 2, "Lifeguard": 3}


def parse_start(start_date: str | None) -> date:
    return date.fromisoformat(start_date) if start_date else date.today() + timedelta(days=1)


def forecast_traffic(start_date: str | None = None, days: int = 7, weeks_back: int = 4) -> pd.DataFrame:
    """Expected check-ins per (date, hour) = average of the same weekday+hour
    over the last `weeks_back` weeks of history."""
    start = parse_start(start_date)
    with connect() as con:
        hist = pd.read_sql_query(
            """SELECT date(checkin_time) AS d,
                      CAST(strftime('%H', checkin_time) AS INTEGER) AS hour,
                      COUNT(*) AS n
               FROM checkins WHERE date(checkin_time) < ?
               GROUP BY d, hour""",
            con, params=[start.isoformat()],
        )
    if hist.empty:
        raise ValueError(f"No check-in history before {start}.")
    hist["d"] = pd.to_datetime(hist["d"])
    recent = hist[hist["d"] >= pd.Timestamp(start) - pd.Timedelta(weeks=weeks_back)]
    if recent.empty:
        recent = hist
    profile = recent.assign(weekday=recent["d"].dt.weekday).groupby(["weekday", "hour"])["n"].mean()

    rows = []
    for i in range(days):
        day = start + timedelta(days=i)
        for h in OPEN_HOURS:
            rows.append({"date": day.isoformat(), "hour": h,
                         "expected_checkins": round(float(profile.get((day.weekday(), h), 0.0)), 1)})
    return pd.DataFrame(rows)


def coverage(start_date: str | None = None, days: int = 7) -> pd.DataFrame:
    """Forecast + required staff + currently scheduled staff for every open hour."""
    fc = forecast_traffic(start_date, days)
    start = parse_start(start_date)
    end = start + timedelta(days=days - 1)
    with connect() as con:
        shifts = con.execute(
            "SELECT shift_date, start_hour, end_hour FROM shifts WHERE shift_date BETWEEN ? AND ?",
            [start.isoformat(), end.isoformat()],
        ).fetchall()
    scheduled = defaultdict(int)
    for d, s, e in shifts:
        for h in range(s, e):
            scheduled[(d, h)] += 1
    fc["required_staff"] = fc["expected_checkins"].apply(
        lambda x: max(MIN_STAFF, math.ceil(x / CHECKINS_PER_STAFF)))
    fc["scheduled_staff"] = [scheduled[(d, h)] for d, h in zip(fc["date"], fc["hour"])]
    fc["shortfall"] = (fc["required_staff"] - fc["scheduled_staff"]).clip(lower=0)
    return fc


def staffing_gaps(start_date: str | None = None, days: int = 7) -> pd.DataFrame:
    cov = coverage(start_date, days)
    return cov[cov["shortfall"] > 0].reset_index(drop=True)


def _block_for(hour: int) -> int:
    return max(b for b in SHIFT_BLOCKS if b <= hour)


def propose_shifts(start_date: str | None = None, days: int = 7):
    """Draft extra 4-hour shifts that close the gaps.

    Respects each staff member's weekly hour limit and never double-books anyone.
    Proposals are saved with status 'pending'; nothing touches the live schedule
    until a manager approves them. Returns (batch_id, proposals, unfilled).
    """
    gaps = staffing_gaps(start_date, days)
    if gaps.empty:
        return None, [], []
    gaps["block"] = gaps["hour"].apply(_block_for)

    with connect(write=True) as con:
        staff = {sid: (name, role, max_h) for sid, name, role, max_h in
                 con.execute("SELECT staff_id, name, role, max_hours_per_week FROM staff")}
        week_hours, busy = defaultdict(int), set()
        for sid, d, s, e in con.execute("SELECT staff_id, shift_date, start_hour, end_hour FROM shifts"):
            week_hours[(sid, date.fromisoformat(d).isocalendar()[:2])] += e - s
            busy.update((sid, d, h) for h in range(s, e))

        proposals, unfilled = [], []
        for (d, block), grp in gaps.groupby(["date", "block"], sort=True):
            needed = int(grp["shortfall"].max())
            peak = grp.loc[grp["shortfall"].idxmax()]
            week = date.fromisoformat(d).isocalendar()[:2]
            hours = range(block, block + SHIFT_LENGTH)
            candidates = sorted(
                (sid for sid, (_, _, max_h) in staff.items()
                 if all((sid, d, h) not in busy for h in hours)
                 and week_hours[(sid, week)] + SHIFT_LENGTH <= max_h),
                key=lambda sid: (ROLE_PRIORITY.get(staff[sid][1], 9),
                                 week_hours[(sid, week)] - staff[sid][2]),
            )
            for sid in candidates[:needed]:
                week_hours[(sid, week)] += SHIFT_LENGTH
                busy.update((sid, d, h) for h in hours)
                proposals.append({
                    "staff_id": sid, "name": staff[sid][0], "role": staff[sid][1],
                    "shift_date": d, "start_hour": block, "end_hour": block + SHIFT_LENGTH,
                    "reason": f"{int(peak['shortfall'])} short at {int(peak['hour'])}:00 "
                              f"(~{peak['expected_checkins']:.0f} check-ins expected)",
                })
            if len(candidates) < needed:
                unfilled.append({"shift_date": d, "start_hour": block,
                                 "missing": needed - len(candidates)})

        if not proposals:
            return None, [], unfilled

        batch_id = uuid.uuid4().hex[:8]
        now = datetime.now().isoformat(timespec="seconds")
        con.execute("UPDATE shift_proposals SET status='superseded' WHERE status='pending'")
        for p in proposals:
            cur = con.execute(
                """INSERT INTO shift_proposals
                   (batch_id, staff_id, shift_date, start_hour, end_hour, reason, status, created_at)
                   VALUES (?,?,?,?,?,?, 'pending', ?)""",
                [batch_id, p["staff_id"], p["shift_date"], p["start_hour"], p["end_hour"], p["reason"], now],
            )
            p["proposal_id"] = cur.lastrowid
    return batch_id, proposals, unfilled


def load_proposals(batch_id: str, status: str = "pending") -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT p.proposal_id, s.name, s.role, p.shift_date, p.start_hour, p.end_hour, p.reason
               FROM shift_proposals p JOIN staff s USING (staff_id)
               WHERE p.batch_id = ? AND p.status = ?
               ORDER BY p.shift_date, p.start_hour, s.name""",
            [batch_id, status],
        ).fetchall()
    keys = ["proposal_id", "name", "role", "shift_date", "start_hour", "end_hour", "reason"]
    return [dict(zip(keys, r)) for r in rows]


def apply_decision(batch_id: str, approved_ids: list[int]) -> tuple[int, int]:
    """Publish approved proposals into `shifts`; reject the rest of the batch.
    Only touches rows still 'pending', so running it twice is harmless."""
    approved = set(approved_ids)
    published = rejected = 0
    with connect(write=True) as con:
        pending = con.execute(
            """SELECT proposal_id, staff_id, shift_date, start_hour, end_hour
               FROM shift_proposals WHERE batch_id = ? AND status = 'pending'""", [batch_id],
        ).fetchall()
        for pid, sid, d, s, e in pending:
            if pid in approved:
                con.execute("INSERT INTO shifts (staff_id, shift_date, start_hour, end_hour) VALUES (?,?,?,?)",
                            [sid, d, s, e])
                con.execute("UPDATE shift_proposals SET status='approved' WHERE proposal_id=?", [pid])
                published += 1
            else:
                con.execute("UPDATE shift_proposals SET status='rejected' WHERE proposal_id=?", [pid])
                rejected += 1
    return published, rejected
