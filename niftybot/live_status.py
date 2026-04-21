"""JSON snapshot written by `python main.py live` for the Streamlit home / Strategy tab."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
STATUS_PATH = Path(__file__).resolve().parent.parent / "niftybot_live_status.json"


def write_live_status(payload: dict[str, Any]) -> None:
    """Merge into a single file (best-effort; never raises to callers)."""
    try:
        payload = {**payload, "updated_at_ist": datetime.now(IST).isoformat(timespec="seconds")}
        STATUS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except OSError as e:
        print(f"niftybot live_status: write failed: {e!r}", flush=True)


def read_live_status() -> dict[str, Any] | None:
    if not STATUS_PATH.is_file():
        return None
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
