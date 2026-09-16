"""Database access + tool-level failure recovery.

- connect():          opens the SQLite DB (read-only unless write=True) and always closes it.
- with_db_retries():  retries *transient* failures with exponential backoff.
- Chaos mode:         set CHAOS_FAILURE_RATE=0.3 to randomly fail 30% of DB calls, so you
                      can watch the agent recover. Leave it at 0 for normal use.
"""
import logging
import os
import random
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("rec_ops.db")

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "rec_center.db"


def db_path() -> Path:
    return Path(os.getenv("REC_DB_PATH", str(DEFAULT_DB)))


class TransientDBError(Exception):
    """A temporary failure that is worth retrying."""


@contextmanager
def connect(write: bool = False):
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python data/generate_data.py")
    uri = path.resolve().as_uri() + ("" if write else "?mode=ro")   # works on Windows paths too
    con = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
    try:
        yield con
        if write:
            con.commit()
    finally:
        con.close()


def _maybe_chaos():
    rate = float(os.getenv("CHAOS_FAILURE_RATE", "0") or 0)
    if rate > 0 and random.random() < rate:
        raise TransientDBError("simulated outage: database temporarily unavailable")


def _is_transient(err: Exception) -> bool:
    if isinstance(err, TransientDBError):
        return True
    msg = str(err).lower()
    return isinstance(err, sqlite3.OperationalError) and any(
        s in msg for s in ("locked", "busy", "unable to open", "disk i/o")
    )


def with_db_retries(fn, attempts: int = 3, base_delay: float = 0.2):
    """Run fn(); retry transient errors with backoff (0.2s, 0.4s, ...).

    Returns (result, retries_used). Non-transient errors (e.g. bad SQL) are raised
    immediately, because retrying them would never help.
    """
    for attempt in range(1, attempts + 1):
        try:
            _maybe_chaos()
            return fn(), attempt - 1
        except Exception as err:  # noqa: BLE001
            if not _is_transient(err) or attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("Transient DB error (%s). Retry %d/%d in %.1fs", err, attempt, attempts - 1, delay)
            time.sleep(delay)
