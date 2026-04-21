"""Replay Session ORB v2 on historical 5m bars (same engine as `backtest` and live)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from growwapi import GrowwAPI

from niftybot.config import Settings
from niftybot.data_feed import fetch_historical_5m
from niftybot.strategy import compute_indicators, run_orb_simulation


@dataclass
class SessionORBReplayResult:
    capital: float
    pnl_history: list[float]


def run_quote_orb_backtest(
    groww: GrowwAPI,
    settings: Settings,
    *,
    start_time: str,
    end_time: str,
    min_orb_ticks: int | None = None,
) -> SessionORBReplayResult:
    """
    Same path as `python main.py backtest`: 5m OHLC + `compute_indicators` + `run_orb_simulation`.

    `min_orb_ticks` is ignored (kept for CLI compatibility); ORB depth is `ORB_MIN_ORB_CANDLES` / settings.orb.
    """
    _ = min_orb_ticks
    df = fetch_historical_5m(
        groww,
        trading_symbol=settings.symbol,
        start_time=start_time,
        end_time=end_time,
    )
    if df.empty:
        raise ValueError("No historical rows returned for replay.")

    df = compute_indicators(
        df.sort_index(),
        choppy_range_mean_max=settings.orb.choppy_range_mean_max,
    )
    trades, capital = run_orb_simulation(
        df,
        initial_capital=settings.capital,
        risk_per_trade=settings.risk_per_trade,
        params=settings.orb,
        debug=False,
    )
    return SessionORBReplayResult(capital=capital, pnl_history=trades)


def print_quote_report(runner: SessionORBReplayResult) -> None:
    trades = pd.Series(runner.pnl_history) if runner.pnl_history else pd.Series(dtype=float)
    print("\n===== SESSION ORB v2 REPLAY (5m, same as backtest) =====")
    print(f"Trades: {len(trades)}")
    if len(trades):
        print(f"Win Rate: {(trades > 0).mean() * 100:.2f}%")
        print(f"PnL: {trades.sum():.2f}")
    print(f"Capital: {runner.capital:.2f}")
