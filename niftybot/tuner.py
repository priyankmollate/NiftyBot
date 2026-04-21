"""
Grid-search ORBBacktestParams on historical 5m data.

Train / holdout split by session day; objective blends PnL and win-rate vs target.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict

import numpy as np
import pandas as pd
from growwapi import GrowwAPI

from niftybot.data_feed import fetch_historical_5m
from niftybot.strategy import ORBBacktestParams, compute_indicators, run_orb_simulation


def _split_by_date(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df.index.date.unique())
    if len(dates) < 3:
        return df, df.iloc[0:0]
    k = max(int(len(dates) * train_frac), 1)
    if k >= len(dates):
        k = len(dates) - 1
    train_dates = set(dates[:k])
    test_dates = set(dates[k:])
    train = df[[d in train_dates for d in df.index.date]]
    test = df[[d in test_dates for d in df.index.date]]
    return train, test


def _score_config(
    trades: list[float],
    *,
    target_win_rate: float,
    min_trades: int,
) -> float:
    if len(trades) < min_trades:
        return -1e9
    t = np.array(trades, dtype=float)
    wr = float((t > 0).mean())
    pnl = float(t.sum())
    wr_pen = 0.0 if wr >= target_win_rate else (target_win_rate - wr) * 50_000.0
    return pnl - wr_pen + 200.0 * wr * len(trades)


def grid_search_orb(
    groww: GrowwAPI,
    *,
    symbol: str,
    start_time: str,
    end_time: str,
    initial_capital: float,
    risk_per_trade: float,
    train_frac: float = 0.65,
    target_win_rate: float = 0.60,
    min_trades_train: int = 3,
) -> tuple[ORBBacktestParams, dict, dict]:
    raw = fetch_historical_5m(
        groww,
        trading_symbol=symbol,
        start_time=start_time,
        end_time=end_time,
    )
    if raw.empty:
        raise ValueError("No data for tuning.")

    train_df, test_df = _split_by_date(raw, train_frac)

    grid = {
        "min_confirmations": [1, 2, 3],
        "rr": [1.6, 2.0, 2.4],
        "sl_atr_cap_mult": [1.15, 1.35, 1.55],
        "orb_sl_buffer_atr": [0.12],
        "require_ema_trend": [False, True],
        "skip_choppy": [False, True],
        "volume_vs_ma_mult": [1.05, 1.15],
        "max_trades_per_day": [2],
        "one_trade_per_day": [False, True],
        "choppy_range_mean_max": [0.0016],
    }

    keys = list(grid.keys())
    best: tuple[float, ORBBacktestParams | None] = (-np.inf, None)

    for values in itertools.product(*[grid[k] for k in keys]):
        kwargs = dict(zip(keys, values, strict=True))
        base = ORBBacktestParams()
        merged = {**asdict(base), **kwargs}
        merged["min_score"] = merged.get("min_confirmations", 2)
        p = ORBBacktestParams(**merged)

        tr_df = compute_indicators(
            train_df.copy(),
            choppy_range_mean_max=p.choppy_range_mean_max,
        )
        trades, _ = run_orb_simulation(
            tr_df,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            params=p,
            debug=False,
        )
        sc = _score_config(trades, target_win_rate=target_win_rate, min_trades=min_trades_train)
        if sc > best[0]:
            best = (sc, p)

    if best[1] is None:
        raise RuntimeError("Grid search failed to evaluate any configuration.")

    best_p = best[1]

    tr_ind = compute_indicators(
        train_df.copy(),
        choppy_range_mean_max=best_p.choppy_range_mean_max,
    )
    te_ind = compute_indicators(
        test_df.copy(),
        choppy_range_mean_max=best_p.choppy_range_mean_max,
    )

    train_trades, train_cap = run_orb_simulation(
        tr_ind,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        params=best_p,
        debug=False,
    )
    test_trades, test_cap = run_orb_simulation(
        te_ind,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        params=best_p,
        debug=False,
    )

    def stats(tr: list[float], cap: float) -> dict:
        if not tr:
            return {"n": 0, "win_rate": None, "pnl": 0.0, "capital": cap}
        a = np.array(tr)
        return {
            "n": len(a),
            "win_rate": float((a > 0).mean()),
            "pnl": float(a.sum()),
            "capital": cap,
        }

    return best_p, stats(train_trades, train_cap), stats(test_trades, test_cap)


def print_tune_report(
    best: ORBBacktestParams,
    train_s: dict,
    test_s: dict,
    *,
    target_win_rate: float,
) -> None:
    print("\n========== SESSION ORB v2 — GRID SEARCH ==========")
    print("Best params (train split):")
    for k, v in asdict(best).items():
        print(f"  {k}: {v}")
    print(f"\n--- Train --- n={train_s['n']}  WR={_fmt_wr(train_s['win_rate'])}  PnL={train_s['pnl']:.2f}")
    print(f"--- Holdout --- n={test_s['n']}  WR={_fmt_wr(test_s['win_rate'])}  PnL={test_s['pnl']:.2f}")
    print(
        f"\nTarget win rate: {target_win_rate*100:.0f}%  "
        "(Holdout is a rough OOS check; short samples are noisy.)"
    )


def _fmt_wr(wr: float | None) -> str:
    if wr is None:
        return "n/a"
    return f"{wr*100:.1f}%"
