from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from growwapi import GrowwAPI

from niftybot.data_feed import fetch_historical_5m


@dataclass
class ORBBacktestParams:
    """
    Session ORB (institutional-style): define opening range, trade breakouts with
    structural invalidation, capped ATR risk, and a small confirmation quorum.

    Defaults favour *participation* (not one ultra-filtered trade per fortnight)
    while keeping explicit risk geometry.
    """

    orb_start: str = "09:15"
    orb_end: str = "09:45"
    min_orb_candles: int = 6

    rr: float = 2.0
    # Stop anchored beyond opposite side of ORB, tightened vs a pure ATR stop:
    orb_sl_buffer_atr: float = 0.12
    sl_atr_cap_mult: float = 1.35

    # Setup: need `min_confirmations` of the three checks (volume, body, vwap/momentum)
    min_confirmations: int = 2
    volume_vs_ma_mult: float = 1.12
    body_ratio_min: float = 0.32
    require_vwap_side: bool = True

    require_ema_trend: bool = False
    skip_choppy: bool = False
    choppy_range_mean_max: float = 0.0016

    # Session cadence: prop desks cap re-entries; single-trade mode still available.
    max_trades_per_day: int = 2
    one_trade_per_day: bool = False

    # --- legacy tuner / env aliases (still read in config) ---
    min_score: int = 2  # maps to min_confirmations when loading old .env
    sl_atr_mult: float = 1.2  # legacy name; equals sl_atr_cap_mult if only one set
    require_trend: bool = False  # alias for require_ema_trend
    vol_spike_mult: float = 1.5  # unused in v2; kept for tuner compatibility


def compute_indicators(
    df: pd.DataFrame,
    *,
    choppy_range_mean_max: float = 0.0016,
) -> pd.DataFrame:
    out = df.copy()

    out["tp"] = (out["high"] + out["low"] + out["close"]) / 3
    out["cum_vol"] = out.groupby(out.index.date)["volume"].cumsum()
    out["cum_tp_vol"] = (out["tp"] * out["volume"]).groupby(out.index.date).cumsum()
    out["vwap"] = out["cum_tp_vol"] / out["cum_vol"].replace(0, np.nan)

    out["tr"] = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            abs(out["high"] - out["close"].shift(1)),
            abs(out["low"] - out["close"].shift(1)),
        ),
    )
    out["atr"] = out["tr"].rolling(14).mean()

    out["body"] = abs(out["close"] - out["open"])
    out["candle_range"] = out["high"] - out["low"]
    out["body_ratio"] = out["body"] / out["candle_range"].replace(0, np.nan)

    out["vol_avg"] = out["volume"].rolling(20).mean()

    out["ema20"] = out["close"].ewm(span=20).mean()
    out["ema50"] = out["close"].ewm(span=50).mean()
    out["trend_up"] = out["ema20"] > out["ema50"]
    out["trend_down"] = out["ema20"] < out["ema50"]

    out["momentum"] = out["close"] - out["close"].shift(3)

    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["choppy"] = out["range_pct"].rolling(20).mean() < choppy_range_mean_max

    return out


def _orb_end_time(p: ORBBacktestParams) -> object:
    return pd.to_datetime(p.orb_end).time()


def _confirmations(row: pd.Series, side: str, p: ORBBacktestParams) -> int:
    """Three checks; require `min_confirmations` hits (default 2)."""
    n = 0
    if pd.notna(row["vol_avg"]) and float(row["volume"]) >= p.volume_vs_ma_mult * float(row["vol_avg"]):
        n += 1
    if pd.notna(row["body_ratio"]) and float(row["body_ratio"]) >= p.body_ratio_min:
        n += 1
    if p.require_vwap_side:
        if side == "LONG" and pd.notna(row["vwap"]) and float(row["close"]) > float(row["vwap"]):
            n += 1
        if side == "SHORT" and pd.notna(row["vwap"]) and float(row["close"]) < float(row["vwap"]):
            n += 1
    else:
        if (
            pd.notna(row["momentum"])
            and pd.notna(row["atr"])
            and abs(float(row["momentum"])) >= 0.45 * float(row["atr"])
        ):
            n += 1
    return n


def _long_stop(entry: float, range_low: float, atr: float, p: ORBBacktestParams) -> float:
    structural = float(range_low) - p.orb_sl_buffer_atr * float(atr)
    capped = float(entry) - p.sl_atr_cap_mult * float(atr)
    sl = max(structural, capped)
    if sl >= entry:
        sl = entry * 0.9995
    return sl


def _short_stop(entry: float, range_high: float, atr: float, p: ORBBacktestParams) -> float:
    structural = float(range_high) + p.orb_sl_buffer_atr * float(atr)
    capped = float(entry) + p.sl_atr_cap_mult * float(atr)
    sl = min(structural, capped)
    if sl <= entry:
        sl = entry * 1.0005
    return sl


class SessionORBEngine:
    """
    Stateful Session ORB v2: same bar logic as historical backtest.
    Used by `run_orb_simulation` (full days) and live 5m closed-bar stepping.
    """

    def __init__(
        self,
        params: ORBBacktestParams,
        risk_per_trade: float,
        *,
        debug: bool = False,
    ):
        self.params = params
        self.risk_per_trade = risk_per_trade
        self.debug = debug
        self.capital = 0.0
        self.closed_trades: list[float] = []
        self._events: list[tuple[str, dict]] = []

        self.min_conf = max(1, min(3, int(params.min_confirmations)))
        self.ema_long = params.require_ema_trend or params.require_trend
        self.ema_short = params.require_ema_trend or params.require_trend
        self.max_day = 1 if params.one_trade_per_day else max(1, int(params.max_trades_per_day))
        self.orb_end_t = _orb_end_time(params)

        self.range_high: float | None = None
        self.range_low: float | None = None
        self.in_trade = False
        self.trade: dict = {}
        self.trades_today = 0
        # Last ORB / bar diagnostics for Streamlit `niftybot_live_status.json` (live loop only).
        self._orb_ui: dict = {}
        self._bar_ui: dict = {}

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def reset_position_state(self) -> None:
        self.in_trade = False
        self.trade = {}
        self.trades_today = 0
        self.range_high = None
        self.range_low = None

    def try_set_orb(self, day_df: pd.DataFrame) -> bool:
        orb = day_df.between_time(self.params.orb_start, self.params.orb_end)
        self._log(f"ORB {self.params.orb_start}-{self.params.orb_end} | size: {len(orb)}")
        last_t = orb.index.max().time() if not orb.index.empty else None
        self._orb_ui = {
            "window": f"{self.params.orb_start}–{self.params.orb_end}",
            "bars_in_window": int(len(orb)),
            "min_bars_required": int(self.params.min_orb_candles),
            "latest_orb_bar_time": str(last_t) if last_t is not None else None,
            "orb_end_time": str(self.orb_end_t),
            "locked": False,
        }
        if len(orb) < self.params.min_orb_candles:
            self._orb_ui["wait_reason"] = (
                f"Need ≥{self.params.min_orb_candles} completed 5m bars inside ORB window; have {len(orb)}."
            )
            return False
        # Do not lock on a partial opening range (live replays cumulative `cum` bar-by-bar).
        if orb.index.empty or orb.index.max().time() < self.orb_end_t:
            self._orb_ui["wait_reason"] = (
                "ORB range not locked yet: data must include a bar ending after ORB end "
                f"(latest in-window bar time {last_t}, need time ≥ {self.orb_end_t})."
            )
            return False
        self.range_high = float(orb["high"].max())
        self.range_low = float(orb["low"].min())
        self._orb_ui["locked"] = True
        self._orb_ui["range_high"] = self.range_high
        self._orb_ui["range_low"] = self.range_low
        self._orb_ui.pop("wait_reason", None)
        self._log(f"ORB High: {self.range_high:.2f} | ORB Low: {self.range_low:.2f}")
        return True

    def drain_events(self) -> list[tuple[str, dict]]:
        out = self._events[:]
        self._events.clear()
        return out

    def on_bar(self, row: pd.Series) -> None:
        close = float(row["close"])
        self._bar_ui = {
            "bar_start_ist": str(row.name),
            "close": close,
            "phase": "unknown",
        }

        if self.range_high is None or self.range_low is None:
            self._bar_ui["phase"] = "orb_not_locked"
            self._bar_ui["why"] = "ORB high/low not set; see ORB wait above."
            return

        if row.name.time() <= self.orb_end_t:
            self._bar_ui["phase"] = "still_inside_orb_window"
            self._bar_ui["why"] = (
                f"No entries on bars ≤ ORB end ({self.orb_end_t}); strategy waits for post-ORB bars only."
            )
            return

        if self.params.skip_choppy and bool(row["choppy"]):
            self._log(f"Choppy skipped @ {row.name}")
            self._bar_ui["phase"] = "skipped_choppy"
            self._bar_ui["why"] = "ORB_SKIP_CHOPPY=1 and this bar is flagged choppy (low average range)."
            return

        atr = row["atr"]
        if pd.isna(atr) or float(atr) <= 0:
            self._bar_ui["phase"] = "no_atr"
            self._bar_ui["why"] = "ATR missing or zero — indicators not ready on this bar."
            return

        c_long = _confirmations(row, "LONG", self.params)
        c_short = _confirmations(row, "SHORT", self.params)
        self._bar_ui["confirmations_long"] = int(c_long)
        self._bar_ui["confirmations_short"] = int(c_short)
        self._bar_ui["min_confirmations"] = int(self.min_conf)
        self._bar_ui["vwap"] = float(row["vwap"]) if pd.notna(row.get("vwap")) else None
        self._log(
            f"{row.name.time()} | C {row['close']:.2f} | "
            f"conf L/S {c_long}/{c_short} | VWAP {row['vwap']:.2f}"
        )

        if not self.in_trade:
            if self.trades_today >= self.max_day:
                self._bar_ui["phase"] = "flat_daily_cap"
                self._bar_ui["why"] = "Max trades for this session day reached (ORB_MAX_TRADES_PER_DAY / one-trade mode)."
                return

            rh, rl = float(self.range_high), float(self.range_low)
            if close > rh:
                self._log("Long breakout probe")
                ok_ema = bool(row["trend_up"]) if self.ema_long else True
                if ok_ema and c_long >= self.min_conf:
                    entry = close
                    sl = _long_stop(entry, self.range_low, float(atr), self.params)
                    risk_1 = entry - sl
                    if risk_1 <= 0:
                        self._log("LONG skip: non-positive risk")
                        self._bar_ui["phase"] = "long_breakout"
                        self._bar_ui["why"] = "Breakout above ORB high but stop geometry gave non-positive risk (skip)."
                        return
                    target = entry + self.params.rr * risk_1
                    risk_amt = self.capital * self.risk_per_trade
                    qty = max(int(risk_amt / risk_1), 1)
                    self.trade = {"type": "BUY", "entry": entry, "sl": sl, "target": target, "qty": qty}
                    self._log(f"ENTRY BUY @ {entry:.2f} SL {sl:.2f} T {target:.2f} qty {qty}")
                    self.in_trade = True
                    self._bar_ui["phase"] = "entry_buy"
                    self._bar_ui["why"] = "Close above ORB high with enough confirmations — ENTRY BUY."
                    self._bar_ui["trade"] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in self.trade.items()}
                    self._events.append(("ENTRY", dict(self.trade)))
                else:
                    parts = []
                    if not ok_ema:
                        parts.append("EMA trend filter: need trend_up for long")
                    if c_long < self.min_conf:
                        parts.append(f"confirmations {c_long}/{self.min_conf} (volume/body/VWAP side)")
                    self._bar_ui["phase"] = "long_breakout_blocked"
                    self._bar_ui["why"] = "Close above ORB high but entry blocked: " + "; ".join(parts) + "."

            elif close < rl:
                self._log("Short breakout probe")
                ok_ema = bool(row["trend_down"]) if self.ema_short else True
                if ok_ema and c_short >= self.min_conf:
                    entry = close
                    sl = _short_stop(entry, self.range_high, float(atr), self.params)
                    risk_1 = sl - entry
                    if risk_1 <= 0:
                        self._log("SHORT skip: non-positive risk")
                        self._bar_ui["phase"] = "short_breakout"
                        self._bar_ui["why"] = "Breakout below ORB low but stop geometry gave non-positive risk (skip)."
                        return
                    target = entry - self.params.rr * risk_1
                    risk_amt = self.capital * self.risk_per_trade
                    qty = max(int(risk_amt / risk_1), 1)
                    self.trade = {"type": "SELL", "entry": entry, "sl": sl, "target": target, "qty": qty}
                    self._log(f"ENTRY SELL @ {entry:.2f} SL {sl:.2f} T {target:.2f} qty {qty}")
                    self.in_trade = True
                    self._bar_ui["phase"] = "entry_sell"
                    self._bar_ui["why"] = "Close below ORB low with enough confirmations — ENTRY SELL (short)."
                    self._bar_ui["trade"] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in self.trade.items()}
                    self._events.append(("ENTRY", dict(self.trade)))
                else:
                    parts = []
                    if not ok_ema:
                        parts.append("EMA trend filter: need trend_down for short")
                    if c_short < self.min_conf:
                        parts.append(f"confirmations {c_short}/{self.min_conf}")
                    self._bar_ui["phase"] = "short_breakout_blocked"
                    self._bar_ui["why"] = "Close below ORB low but entry blocked: " + "; ".join(parts) + "."

            else:
                self._bar_ui["phase"] = "no_breakout"
                self._bar_ui["why"] = (
                    f"Close {close:.2f} inside ORB range [{rl:.2f}, {rh:.2f}] — wait for breakout or different bar."
                )

        else:
            exit_price: float | None = None
            trade = self.trade

            if trade["type"] == "BUY":
                if float(row["low"]) <= trade["sl"]:
                    exit_price = trade["sl"]
                elif float(row["high"]) >= trade["target"]:
                    exit_price = trade["target"]
            else:
                if float(row["high"]) >= trade["sl"]:
                    exit_price = trade["sl"]
                elif float(row["low"]) <= trade["target"]:
                    exit_price = trade["target"]

            if row.name.hour == 15 and row.name.minute >= 15:
                exit_price = float(row["close"])

            self._bar_ui["phase"] = "in_trade"
            self._bar_ui["open_trade"] = {
                "side": trade["type"],
                "entry": float(trade["entry"]),
                "sl": float(trade["sl"]),
                "target": float(trade["target"]),
                "qty": int(trade["qty"]),
            }
            if exit_price is None:
                self._bar_ui["why"] = (
                    "In position — exit when bar touches stop/target, or session square-off at 15:15+ IST."
                )
            else:
                self._bar_ui["why"] = f"Exit triggered at {exit_price:.2f} this bar."

            if exit_price is not None:
                if trade["type"] == "BUY":
                    pnl = (exit_price - trade["entry"]) * trade["qty"]
                else:
                    pnl = (trade["entry"] - exit_price) * trade["qty"]

                self.capital += pnl
                self.closed_trades.append(pnl)
                self.trades_today += 1
                self._log(
                    f"EXIT @ {exit_price:.2f} | PnL: {pnl:.2f} | day fills {self.trades_today}/{self.max_day}"
                )
                self._events.append(
                    ("EXIT", {"pnl": pnl, "exit_price": float(exit_price), "trade": dict(trade)})
                )
                self._bar_ui["phase"] = "exited"
                self._bar_ui["exit_price"] = float(exit_price)
                self._bar_ui["pnl"] = float(pnl)
                self._bar_ui["why"] = f"Exit at {exit_price:.2f} | PnL {pnl:+.2f} (flat now)."
                self._bar_ui.pop("open_trade", None)
                self.in_trade = False
                self.trade = {}


def run_orb_simulation(
    df: pd.DataFrame,
    *,
    initial_capital: float,
    risk_per_trade: float,
    params: ORBBacktestParams,
    debug: bool = False,
) -> tuple[list[float], float]:
    capital = initial_capital
    trades: list[float] = []

    def log(msg: str) -> None:
        if debug:
            print(msg)

    for date, day_df in df.groupby(df.index.date):
        log(f"\nDATE: {date} | candles: {len(day_df)}")

        eng = SessionORBEngine(params, risk_per_trade, debug=debug)
        eng.capital = capital
        if not eng.try_set_orb(day_df):
            log("Skipping day (not enough ORB candles)")
            continue

        for i in range(len(day_df)):
            eng.on_bar(day_df.iloc[i])

        capital = eng.capital
        trades.extend(eng.closed_trades)

    return trades, capital


class HighAccuracyORB:
    """Session ORB backtest on Groww 5m FNO candles (ORB v2 engine in `run_orb_simulation`)."""

    def __init__(
        self,
        groww: GrowwAPI,
        symbol: str = "NIFTY26APRFUT",
        capital: float = 100_000.0,
        risk_per_trade: float = 0.01,
        debug: bool = True,
        params: ORBBacktestParams | None = None,
    ):
        self.groww = groww
        self.symbol = symbol
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.debug = debug
        self.params = params or ORBBacktestParams()

        self.data: pd.DataFrame | None = None
        self.trades: list[float] = []

    @property
    def rr(self) -> float:
        return self.params.rr

    def log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def fetch_data(self, start_time: str, end_time: str) -> None:
        self.log("Fetching data from Groww...")
        self.data = fetch_historical_5m(
            self.groww,
            trading_symbol=self.symbol,
            start_time=start_time,
            end_time=end_time,
        )
        self.log(f"Final DF shape: {self.data.shape}")
        self.log(f"Date range: {self.data.index.min()} → {self.data.index.max()}")

    def add_indicators(self) -> None:
        if self.data is None:
            raise RuntimeError("Call fetch_data before add_indicators.")
        self.log("Adding indicators...")
        self.data = compute_indicators(
            self.data,
            choppy_range_mean_max=self.params.choppy_range_mean_max,
        )
        self.log("Indicators ready.")

    def run_backtest(self) -> None:
        df = self.data
        if df is None:
            raise RuntimeError("No data loaded.")

        self.log("Starting Session ORB backtest...")
        self.capital = self.initial_capital
        self.trades, self.capital = run_orb_simulation(
            df,
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            params=self.params,
            debug=self.debug,
        )

    def report(self) -> None:
        if not self.trades:
            print("No trades executed")
            return

        trades = pd.Series(self.trades)
        print("\n===== FINAL REPORT (Session ORB v2) =====")
        print(f"Trades: {len(trades)}")
        print(f"Win Rate: {(trades > 0).mean() * 100:.2f}%")
        print(f"PnL: {trades.sum():.2f}")
        print(f"Capital: {self.capital:.2f}")

    def run(self, start_time: str, end_time: str) -> None:
        self.fetch_data(start_time, end_time)
        self.add_indicators()
        self.run_backtest()
        self.report()
