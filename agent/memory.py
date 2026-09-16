"""Checkpointing: LangGraph saves the full state after every node.

That gives us:
- Conversation memory: follow-up questions in the same thread see earlier messages.
- Crash recovery: if the process dies mid-run, re-run with the same thread id and the
  graph resumes from the last completed node instead of starting over.
- Human-in-the-loop: a paused run (waiting for approval) survives restarts.
"""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite"


def get_checkpointer(path: Path = CHECKPOINT_DB) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
