"""Minimal SQLite helper."""

import sqlite3
from pathlib import Path


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and dict-like rows."""

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, "
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.commit()


def add_run(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute("INSERT INTO runs (name) VALUES (?)", (name,))
    conn.commit()
    return int(cur.lastrowid or 0)
