"""Append-only SQLite log for bot alerts (companion to Telegram)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_DB_PATH = Path(__file__).resolve().parent.parent / "niftybot_alerts.sqlite3"


def db_path() -> Path:
    return _DB_PATH


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(_DB_PATH))


def init_alerts_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT,
                body TEXT NOT NULL,
                extra_json TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts (created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_kind ON alerts (kind)")


def record_alert(
    *,
    kind: str,
    body: str,
    symbol: str | None = None,
    extra: dict | None = None,
) -> None:
    """Insert one row; never raises (failures printed)."""
    try:
        init_alerts_db()
        ts = datetime.now(IST).isoformat(timespec="seconds")
        extra_s = json.dumps(extra, default=str) if extra else None
        with _conn() as c:
            c.execute(
                "INSERT INTO alerts (created_at, kind, symbol, body, extra_json) VALUES (?,?,?,?,?)",
                (ts, kind[:64], symbol, body, extra_s),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"niftybot alerts DB: insert failed: {exc!r}", flush=True)


def fetch_recent_alerts(*, limit: int = 200) -> list[dict]:
    if not _DB_PATH.is_file():
        return []
    try:
        init_alerts_db()
        with _conn() as c:
            cur = c.execute(
                """
                SELECT id, created_at, kind, symbol, body, extra_json
                FROM alerts
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 2000)),),
            )
            rows = cur.fetchall()
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "created_at": r[1],
                "kind": r[2],
                "symbol": r[3],
                "body": r[4],
                "extra_json": r[5],
            }
        )
    return out
