"""Persisted preferences for the Streamlit UI (read by `main.py live`)."""

from __future__ import annotations

import json
from pathlib import Path

_PREFS_PATH = Path(__file__).resolve().parent.parent / "niftybot_ui_prefs.json"

_BOOL_KEYS = frozenset({"execute_orders", "bot_enabled", "telegram_alerts_enabled"})

_DEFAULTS: dict = {
    "execute_orders": False,
    "bot_enabled": True,
    "telegram_alerts_enabled": True,
    "orb_symbol": "",
    "fno_product": "",
    "trade_instrument": "",
    "option_ce_symbol": "",
    "option_pe_symbol": "",
    "option_chain_expiry": "",
}


def prefs_path() -> Path:
    return _PREFS_PATH


def load_ui_prefs() -> dict:
    out = dict(_DEFAULTS)
    if not _PREFS_PATH.is_file():
        return out
    try:
        raw = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(raw, dict):
        return out
    for k in _BOOL_KEYS:
        if k in raw:
            out[k] = bool(raw[k])
    if "orb_symbol" in raw and isinstance(raw["orb_symbol"], str):
        out["orb_symbol"] = raw["orb_symbol"].strip()
    if "fno_product" in raw and isinstance(raw["fno_product"], str):
        fp = raw["fno_product"].strip().upper()
        out["fno_product"] = fp if fp in ("MIS", "NRML") else ""
    if "trade_instrument" in raw and isinstance(raw["trade_instrument"], str):
        ti = raw["trade_instrument"].strip().lower()
        out["trade_instrument"] = ti if ti in ("futures", "options") else ""
    if "option_ce_symbol" in raw and isinstance(raw["option_ce_symbol"], str):
        out["option_ce_symbol"] = raw["option_ce_symbol"].strip()
    if "option_pe_symbol" in raw and isinstance(raw["option_pe_symbol"], str):
        out["option_pe_symbol"] = raw["option_pe_symbol"].strip()
    if "option_chain_expiry" in raw and isinstance(raw["option_chain_expiry"], str):
        out["option_chain_expiry"] = raw["option_chain_expiry"].strip()
    if "option_auto_strike" in raw:
        out["option_auto_strike"] = bool(raw["option_auto_strike"])
    return out


def save_ui_prefs(updates: dict) -> None:
    """Merge `updates` into the JSON file (keeps other keys)."""
    cur = load_ui_prefs()
    for k, v in updates.items():
        if k in _BOOL_KEYS:
            cur[k] = bool(v)
        elif k == "orb_symbol":
            cur[k] = v.strip() if isinstance(v, str) else ""
        elif k == "fno_product":
            s = v.strip().upper() if isinstance(v, str) else ""
            cur[k] = s if s in ("MIS", "NRML") else ""
        elif k == "trade_instrument":
            s = v.strip().lower() if isinstance(v, str) else ""
            cur[k] = s if s in ("futures", "options") else ""
        elif k == "option_ce_symbol":
            cur[k] = v.strip() if isinstance(v, str) else ""
        elif k == "option_pe_symbol":
            cur[k] = v.strip() if isinstance(v, str) else ""
        elif k == "option_chain_expiry":
            cur[k] = v.strip() if isinstance(v, str) else ""
        elif k == "option_auto_strike":
            cur[k] = bool(v)
    _PREFS_PATH.write_text(json.dumps(cur, indent=2) + "\n", encoding="utf-8")


def live_bot_enabled() -> bool:
    """When False, `live` skips each poll (no API, strategy, orders, or alerts)."""
    return bool(load_ui_prefs().get("bot_enabled", True))


def telegram_alerts_enabled() -> bool:
    """When False, alerts are DB-only (no Telegram send)."""
    return bool(load_ui_prefs().get("telegram_alerts_enabled", True))


def live_execute_enabled(*, cli_execute: bool, alerts_only: bool) -> bool:
    """Whether the live bot should send real broker orders."""
    if alerts_only:
        return False
    ui_on = load_ui_prefs().get("execute_orders", False)
    return bool(cli_execute or ui_on)
