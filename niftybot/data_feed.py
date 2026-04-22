from __future__ import annotations

"""
Historical FNO candles for ORB backtests.

Primary path follows Groww **Historical Data** docs (`get_historical_candles`) with
`groww_symbol` resolution from instruments:
https://groww.in/trade-api/docs/python-sdk/historical-data

`trading_symbol` (e.g. NIFTY26APRFUT or a fixed CE/PE) is resolved to `groww_symbol` via the instruments CSV.
Falls back to deprecated `get_historical_candle_data` if the V2 call fails.

Groww limits how much 5m history each request can return (often ~2 weeks). Long ranges are fetched in
multiple chunks and merged (see ``ORB_HIST_CHUNK_DAYS``).
"""

import os
import warnings
from zoneinfo import ZoneInfo

import pandas as pd
from growwapi import GrowwAPI
from growwapi.groww.exceptions import InstrumentNotFoundException

IST = ZoneInfo("Asia/Kolkata")


def _parse_range_to_ist(start_time: str, end_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(start_time.strip())
    b = pd.Timestamp(end_time.strip())
    if a.tzinfo is None:
        a = a.tz_localize(IST)
    else:
        a = a.tz_convert(IST)
    if b.tzinfo is None:
        b = b.tz_localize(IST)
    else:
        b = b.tz_convert(IST)
    return a, b


def candles_deprecated_to_dataframe(response: dict) -> pd.DataFrame:
    """Legacy `get_historical_candle_data`: candles as rows with epoch `timestamp`."""
    candles = response.get("candles") or []
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
    df = df.tz_localize("UTC").tz_convert("Asia/Kolkata")
    return df.sort_index()


def candles_v2_to_dataframe(response: dict) -> pd.DataFrame:
    """
    `get_historical_candles` V2 format: each candle is
    [timestamp_str, open, high, low, close, volume, open_interest?].
    """
    rows = response.get("candles") or []
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )

    parsed: list[dict] = []
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 6:
            continue
        ts, o, h, l, c, vol = r[0], r[1], r[2], r[3], r[4], r[5]
        # Groww can occasionally return partial rows with None in OHLC fields.
        # Skip those rows rather than failing the whole request/fallback path.
        if ts is None or o is None or h is None or l is None or c is None:
            continue
        try:
            o_f = float(o)
            h_f = float(h)
            l_f = float(l)
            c_f = float(c)
            vol_f = float(vol) if vol is not None else 0.0
        except (TypeError, ValueError):
            continue
        parsed.append(
            {
                "timestamp": ts,
                "open": o_f,
                "high": h_f,
                "low": l_f,
                "close": c_f,
                "volume": vol_f,
            }
        )
    if not parsed:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )

    df = pd.DataFrame(parsed)
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df.sort_index()


def _resolve_groww_symbol(groww: GrowwAPI, *, trading_symbol: str) -> str:
    row = groww.get_instrument_by_exchange_and_trading_symbol(
        groww.EXCHANGE_NSE,
        trading_symbol,
    )
    g = (row.get("groww_symbol") or "").strip()
    if not g:
        raise ValueError(f"No groww_symbol for trading_symbol={trading_symbol!r}")
    return g


def _fetch_historical_5m_once(
    groww: GrowwAPI,
    *,
    trading_symbol: str,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """Single API request (may be capped by Groww)."""
    try:
        gsym = _resolve_groww_symbol(groww, trading_symbol=trading_symbol)
        resp = groww.get_historical_candles(
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_FNO,
            groww_symbol=gsym,
            start_time=start_time,
            end_time=end_time,
            candle_interval=groww.CANDLE_INTERVAL_MIN_5,
        )
        return candles_v2_to_dataframe(resp)
    except InstrumentNotFoundException:
        raise
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"get_historical_candles failed ({exc!r}); falling back to get_historical_candle_data.",
            UserWarning,
            stacklevel=3,
        )
        response = groww.get_historical_candle_data(
            trading_symbol=trading_symbol,
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_FNO,
            start_time=start_time,
            end_time=end_time,
            interval_in_minutes=5,
        )
        return candles_deprecated_to_dataframe(response)


def fetch_historical_5m(
    groww: GrowwAPI,
    *,
    trading_symbol: str,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    Fetch 5-minute OHLCV for an FNO `trading_symbol` (future or option).

    Uses `get_historical_candles` + instruments `groww_symbol` per Groww backtesting docs.

    Long ``start_time`` … ``end_time`` ranges are split into chunks of at most
    ``ORB_HIST_CHUNK_DAYS`` calendar days (default **14**) so each request stays within
    Groww limits; results are concatenated and de-duplicated by bar timestamp.
    """
    start_ts, end_ts = _parse_range_to_ist(start_time, end_time)
    if start_ts >= end_ts:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )

    chunk_days = int(os.getenv("ORB_HIST_CHUNK_DAYS", "14") or 14)
    chunk_days = max(1, chunk_days)
    chunk_delta = pd.Timedelta(days=chunk_days)

    frames: list[pd.DataFrame] = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + chunk_delta, end_ts)
        s_str = cur.strftime("%Y-%m-%d %H:%M:%S")
        e_str = nxt.strftime("%Y-%m-%d %H:%M:%S")
        part = _fetch_historical_5m_once(
            groww,
            trading_symbol=trading_symbol,
            start_time=s_str,
            end_time=e_str,
        )
        if not part.empty:
            frames.append(part)
        cur = nxt

    if not frames:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    # Clip to requested window (chunk boundaries can include slight overfetch from API)
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    return out
