"""Generate a dummy SQLite database for the Rec Center Operations agent.

Run:  python data/generate_data.py
Creates data/rec_center.db with realistic-looking (but fake) data.
"""
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

random.seed(42)
DB_PATH = Path(__file__).parent / "rec_center.db"

FACILITIES = ["Weight Room", "Cardio Deck", "Pool", "Basketball Courts", "Climbing Wall"]
# Relative traffic by facility
FACILITY_WEIGHT = [0.35, 0.25, 0.12, 0.20, 0.08]
MEMBERSHIP_TYPES = ["Student", "Faculty/Staff", "Spouse", "Alumni"]
ROLES = ["Front Desk", "Facility Supervisor", "Lifeguard", "Climbing Wall Attendant"]

# Relative busyness by hour (6am-11pm): morning bump, big evening peak
HOUR_WEIGHT = {6: 4, 7: 6, 8: 5, 9: 4, 10: 4, 11: 5, 12: 6, 13: 5, 14: 5,
               15: 7, 16: 9, 17: 11, 18: 12, 19: 10, 20: 8, 21: 6, 22: 3}
# Relative busyness by weekday (Mon=0 ... Sun=6)
DAY_WEIGHT = [1.2, 1.15, 1.1, 1.0, 0.8, 0.55, 0.65]

FIRST = ["Alex", "Jordan", "Taylor", "Chris", "Sam", "Priya", "Diego", "Mia", "Noah",
         "Aisha", "Ethan", "Sofia", "Liam", "Emma", "Ravi", "Grace", "Lucas", "Zoe"]
LAST = ["Garcia", "Nguyen", "Smith", "Patel", "Johnson", "Lee", "Brown", "Martinez",
        "Davis", "Kim", "Wilson", "Lopez", "Clark", "Shah", "Young", "Hall"]


def name():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def build(days: int = 60, n_members: int = 1500, n_staff: int = 40):
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
    CREATE TABLE members (
        member_id INTEGER PRIMARY KEY,
        name TEXT,
        membership_type TEXT,
        join_date DATE
    );
    CREATE TABLE checkins (
        checkin_id INTEGER PRIMARY KEY,
        member_id INTEGER REFERENCES members(member_id),
        facility TEXT,
        checkin_time DATETIME
    );
    CREATE TABLE staff (
        staff_id INTEGER PRIMARY KEY,
        name TEXT,
        role TEXT,
        max_hours_per_week INTEGER
    );
    CREATE TABLE shifts (
        shift_id INTEGER PRIMARY KEY,
        staff_id INTEGER REFERENCES staff(staff_id),
        shift_date DATE,
        start_hour INTEGER,
        end_hour INTEGER
    );
    CREATE TABLE shift_proposals (
        proposal_id INTEGER PRIMARY KEY,
        batch_id TEXT,
        staff_id INTEGER REFERENCES staff(staff_id),
        shift_date DATE,
        start_hour INTEGER,
        end_hour INTEGER,
        reason TEXT,
        status TEXT,          -- pending / approved / rejected / superseded
        created_at DATETIME
    );
    CREATE TABLE lockers (
        locker_id INTEGER PRIMARY KEY,
        location TEXT,
        member_id INTEGER REFERENCES members(member_id),
        expires_on DATE
    );
    """)

    today = date.today()
    start = today - timedelta(days=days)

    # Members
    members = []
    for mid in range(1, n_members + 1):
        mtype = random.choices(MEMBERSHIP_TYPES, weights=[80, 12, 4, 4])[0]
        joined = start - timedelta(days=random.randint(0, 700))
        members.append((mid, name(), mtype, joined.isoformat()))
    cur.executemany("INSERT INTO members VALUES (?,?,?,?)", members)

    # Check-ins
    checkins = []
    for d in range(days):
        day = start + timedelta(days=d)
        base = 900 * DAY_WEIGHT[day.weekday()]
        for hour, hw in HOUR_WEIGHT.items():
            n = int(random.gauss(base * hw / 95, 6))
            for _ in range(max(n, 0)):
                t = datetime(day.year, day.month, day.day, hour, random.randint(0, 59))
                fac = random.choices(FACILITIES, weights=FACILITY_WEIGHT)[0]
                checkins.append((random.randint(1, n_members), fac, t.isoformat(sep=" ")))
    cur.executemany(
        "INSERT INTO checkins (member_id, facility, checkin_time) VALUES (?,?,?)", checkins
    )

    # Staff
    staff = []
    for sid in range(1, n_staff + 1):
        role = random.choices(ROLES, weights=[40, 20, 25, 15])[0]
        staff.append((sid, name(), role, random.choice([10, 15, 20])))
    cur.executemany("INSERT INTO staff VALUES (?,?,?,?)", staff)

    # Shifts for the past period AND the next 14 days (4-hour blocks).
    # Nobody is scheduled past their weekly hour limit, and evenings are often
    # understaffed on purpose so the agent has real gaps to find.
    shifts = []
    week_hours = {}
    for d in range(days + 14):
        day = start + timedelta(days=d)
        week = day.isocalendar()[:2]
        for block_start in (6, 10, 14, 18):
            k = random.randint(2, 4)
            pool = [s for s in staff if week_hours.get((s[0], week), 0) + 4 <= s[3]]
            for s in random.sample(pool, k=min(k, len(pool))):
                week_hours[(s[0], week)] = week_hours.get((s[0], week), 0) + 4
                shifts.append((s[0], day.isoformat(), block_start, block_start + 4))
    cur.executemany(
        "INSERT INTO shifts (staff_id, shift_date, start_hour, end_hour) VALUES (?,?,?,?)",
        shifts,
    )

    # Lockers
    lockers = []
    for lid in range(1, 301):
        loc = random.choice(["Men's Locker Room", "Women's Locker Room", "All-Gender Locker Room"])
        if random.random() < 0.7:
            lockers.append((lid, loc, random.randint(1, n_members),
                            (today + timedelta(days=random.randint(-20, 120))).isoformat()))
        else:
            lockers.append((lid, loc, None, None))
    cur.executemany("INSERT INTO lockers VALUES (?,?,?,?)", lockers)

    con.commit()
    counts = {t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["members", "checkins", "staff", "shifts", "lockers", "shift_proposals"]}
    con.close()
    print(f"Created {DB_PATH}")
    for t, c in counts.items():
        print(f"  {t:10s} {c:>7,} rows")


if __name__ == "__main__":
    build()
