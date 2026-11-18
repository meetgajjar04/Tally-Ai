"""Local SQLite history of all imports and their Tally results."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DB_FILE
from .logger import get_logger

log = get_logger("history")


def _connect() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_file TEXT NOT NULL,
            status TEXT NOT NULL,
            voucher_count INTEGER DEFAULT 0,
            details TEXT DEFAULT ''
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS vouchers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            kind TEXT, date TEXT, number TEXT, party TEXT,
            total REAL, status TEXT, message TEXT, payload TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )"""
    )
    con.commit()
    return con


def create_job(source_file: str) -> int:
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO jobs(created_at, source_file, status) VALUES (?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), Path(source_file).name, "processing"),
        )
        return int(cur.lastrowid)


def finish_job(job_id: int, status: str, details: str = "", count: int = 0) -> None:
    with _connect() as con:
        con.execute(
            "UPDATE jobs SET status=?, details=?, voucher_count=? WHERE id=?",
            (status, details[:2000], count, job_id),
        )


def add_voucher(job_id: int, v: Any, status: str, message: str) -> None:
    with _connect() as con:
        con.execute(
            """INSERT INTO vouchers(job_id, kind, date, number, party, total, status, message, payload)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                job_id,
                getattr(v, "kind", ""),
                getattr(v, "date", ""),
                getattr(v, "number", ""),
                getattr(v, "party", ""),
                float(getattr(v, "total", 0.0)),
                status,
                message[:1000],
                json.dumps(v.model_dump(mode="json"), ensure_ascii=False),
            ),
        )


def recent_jobs(limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as con:
        rows = con.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def job_vouchers(job_id: int) -> list[dict[str, Any]]:
    with _connect() as con:
        rows = con.execute("SELECT * FROM vouchers WHERE job_id=? ORDER BY id", (job_id,)).fetchall()
        return [dict(r) for r in rows]


def clear_history() -> None:
    with _connect() as con:
        con.execute("DELETE FROM vouchers")
        con.execute("DELETE FROM jobs")
