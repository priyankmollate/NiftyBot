"""Pick CE/PE trading_symbol from Groww option chain (nearest OTM + volume in a strike window)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from growwapi import GrowwAPI


def infer_underlying_from_fno_symbol(trading_symbol: str) -> str | None:
    """
    Best-effort underlying for option chain, e.g. NIFTY26APRFUT -> NIFTY, BANKNIFTY26APRFUT -> BANKNIFTY.
    Expects month code + FUT suffix (Groww-style index future).
    """
    s = (trading_symbol or "").strip().upper()
    m = re.match(r"^([A-Z]+)\d{2}[A-Z]{3}FUT$", s)
    return m.group(1) if m else None


def _nearest_otm_ce_strike_grid(ltp: float, step: int) -> int:
    """Smallest strike strictly above spot on the strike grid (OTM call)."""
    s = int(math.ceil(ltp / step) * step)
    if s <= ltp:
        s += step
    return s


def _nearest_otm_pe_strike_grid(ltp: float, step: int) -> int:
    """Largest strike strictly below spot on the strike grid (OTM put)."""
    return int(math.floor((ltp - 1e-9) / step) * step)


def pick_otm_option_symbol(
    groww: GrowwAPI,
    *,
    exchange: str,
    underlying: str,
    expiry_date: str,
    signal: str,
    strike_step: int,
    window_steps: int,
) -> tuple[str | None, str]:
    """
    Returns (trading_symbol, human note). trading_symbol is None on failure.
    BUY -> CE OTM above spot; SELL -> PE OTM below spot.
    Among the next `window_steps` strikes in that direction, pick max volume; tie -> nearest center.
    """
    try:
        raw = groww.get_option_chain(
            exchange=exchange,
            underlying=underlying,
            expiry_date=expiry_date,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"get_option_chain failed: {e!r}"

    ltp = float(raw.get("underlying_ltp") or 0.0)
    strikes = raw.get("strikes") or {}
    if ltp <= 0 or not strikes:
        return None, "option_chain: missing underlying_ltp or strikes"

    window_steps = max(1, int(window_steps))
    strike_step = max(1, int(strike_step))

    if signal == "BUY":
        center = _nearest_otm_ce_strike_grid(ltp, strike_step)
        candidates: list[tuple[int, int, int, str]] = []
        for i in range(window_steps):
            k = center + i * strike_step
            row = strikes.get(str(k)) or strikes.get(k)
            if not row:
                continue
            ce = row.get("CE") or {}
            sym = ce.get("trading_symbol")
            if not sym:
                continue
            vol = int(ce.get("volume") or 0)
            dist = abs(k - center)
            candidates.append((vol, dist, k, str(sym)))
        if not candidates:
            return None, f"no CE row for strikes from {center} (ltp={ltp:.2f})"
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        vol, _d, k, sym = candidates[0]
        return sym, f"auto CE strike={k} vol={vol} ltp={ltp:.2f}"

    if signal == "SELL":
        center = _nearest_otm_pe_strike_grid(ltp, strike_step)
        candidates = []
        for i in range(window_steps):
            k = center - i * strike_step
            if k <= 0:
                break
            row = strikes.get(str(k)) or strikes.get(k)
            if not row:
                continue
            pe = row.get("PE") or {}
            sym = pe.get("trading_symbol")
            if not sym:
                continue
            vol = int(pe.get("volume") or 0)
            dist = abs(k - center)
            candidates.append((vol, dist, k, str(sym)))
        if not candidates:
            return None, f"no PE row for strikes from {center} (ltp={ltp:.2f})"
        candidates.sort(key=lambda x: (-x[0], x[1], -x[2]))
        vol, _d, k, sym = candidates[0]
        return sym, f"auto PE strike={k} vol={vol} ltp={ltp:.2f}"

    return None, f"unknown signal {signal!r}"


def resolve_option_leg_for_signal(
    groww: GrowwAPI,
    settings: Settings,
    signal: str,
    *,
    log_info: Callable[[str], None] | None = None,
) -> str:
    """Static CE/PE from settings, or auto chain when `settings.option_auto_strike` is True."""
    if settings.trade_instrument != "options":
        return settings.symbol
    if settings.option_auto_strike and settings.option_chain_expiry and settings.option_chain_underlying:
        sym, note = pick_otm_option_symbol(
            groww,
            exchange=groww.EXCHANGE_NSE,
            underlying=settings.option_chain_underlying,
            expiry_date=settings.option_chain_expiry,
            signal=signal,
            strike_step=settings.option_strike_step,
            window_steps=settings.option_otm_window_steps,
        )
        if sym:
            line = f"option_auto: {note}"
            print(line, flush=True)
            if log_info:
                log_info(line)
            return sym
        line = f"option_auto fallback ({note})"
        print(line, flush=True)
        if log_info:
            log_info(line)
    if signal == "BUY":
        return settings.option_ce_symbol or settings.symbol
    return settings.option_pe_symbol or settings.symbol
