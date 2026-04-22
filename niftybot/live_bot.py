from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from growwapi import GrowwAPI

from niftybot.alerts_store import record_alert
from niftybot.config import Settings
from niftybot.data_feed import fetch_historical_5m
from niftybot.strategy import SessionORBEngine, compute_indicators
from niftybot.live_status import write_live_status
from niftybot.option_auto import resolve_option_leg_for_signal
from niftybot.telegram_alerts import send_telegram_message
from niftybot.ui_prefs import live_bot_enabled, telegram_alerts_enabled

IST = ZoneInfo("Asia/Kolkata")


def _dispatch_alert(
    settings: Settings,
    text: str,
    log,
    *,
    kind: str,
    symbol: str | None = None,
    extra: dict | None = None,
) -> None:
    """Persist every alert to SQLite; send Telegram when UI + .env allow."""
    record_alert(kind=kind, body=text, symbol=symbol, extra=extra)
    if not telegram_alerts_enabled():
        return
    tok, chat = settings.telegram_bot_token, settings.telegram_chat_id
    if not tok or not chat:
        return
    ok, err = send_telegram_message(bot_token=tok, chat_id=chat, text=text)
    if not ok:
        print(f"Telegram failed: {err}", flush=True)
        log(f"Telegram failed: {err}")


def _telegram_bot_started(
    settings: Settings,
    log,
    holder: object,
    *,
    mode: str,
    dry_run: bool,
    alerts_only: bool,
    poll_seconds: float,
    symbol: str,
    trade_instrument: str,
    orders_enabled: bool,
    once: bool = False,
) -> None:
    """One startup alert per process: always DB + Telegram if configured (idempotent via `holder`)."""
    if getattr(holder, "_startup_telegram_sent", False):
        return
    run_mode = "single poll (--once)" if once else "continuous loop"
    body = (
        f"NiftyBot · BOT STARTED\n"
        f"{mode}\n"
        f"Run: {run_mode}\n"
        f"Symbol: {symbol}\n"
        f"Instrument: {trade_instrument}\n"
        f"Poll: {poll_seconds:g}s\n"
        f"dry_run={dry_run}  alerts_only={alerts_only}  broker_orders={'on' if orders_enabled else 'off'}\n"
        f"IST {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    _dispatch_alert(
        settings,
        body,
        log,
        kind="bot_started",
        symbol=symbol,
        extra={
            "mode": mode,
            "run_mode": run_mode,
            "dry_run": dry_run,
            "alerts_only": alerts_only,
            "orders_enabled": orders_enabled,
        },
    )
    setattr(holder, "_startup_telegram_sent", True)


def _safe(x, default=None):
    return default if x is None else x


def _market_open_ist(ts: pd.Timestamp) -> pd.Timestamp:
    ts = ts.tz_convert(IST)
    return pd.Timestamp(
        year=ts.year,
        month=ts.month,
        day=ts.day,
        hour=9,
        minute=15,
        second=0,
        microsecond=0,
        tzinfo=IST,
    )


@dataclass
class SessionORBLiveRunner:
    """
    Live Session ORB v2: same bar engine as `backtest` / `run_orb_simulation`.
    Polls Groww 5m history, steps only *closed* bars for the current IST session.
    """

    groww: GrowwAPI
    settings: Settings
    dry_run: bool = True
    poll_seconds: float = 15.0
    warmup_days: int = 14

    pnl_history: list[float] = field(default_factory=list)
    _last_bar_ts: pd.Timestamp | None = field(default=None, repr=False)
    _engine_date: date | None = field(default=None, repr=False)
    _engine: SessionORBEngine | None = field(default=None, repr=False)
    _startup_telegram_sent: bool = field(default=False, repr=False)
    _cached_hist_df: pd.DataFrame | None = field(default=None, repr=False)
    _cached_hist_closed_bar_ts: pd.Timestamp | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.capital = self.settings.capital
        self._product = (
            self.groww.PRODUCT_MIS
            if self.settings.fno_product == "MIS"
            else self.groww.PRODUCT_NRML
        )

    def _log(self, msg: str) -> None:
        if self.settings.debug:
            print(msg)

    def _rollover_if_new_day(self, d: date) -> None:
        if self._engine_date is not None and d != self._engine_date:
            if self._engine is not None:
                self.capital = self._engine.capital
            self._engine = None
            self._last_bar_ts = None
            self._log(f"{datetime.now(IST).strftime('%H:%M:%S')} | New session day — ORB v2 engine reset")
        self._engine_date = d
        if self._engine is None:
            self._engine = SessionORBEngine(
                self.settings.orb,
                self.settings.risk_per_trade,
                debug=self.settings.debug,
            )
            self._engine.capital = self.capital

    def _order_symbol_for_signal(self, signal: str) -> str:
        return resolve_option_leg_for_signal(
            self.groww,
            self.settings,
            signal,
            log_info=self._log,
        )

    def _entry_exit_orders(self, signal: str) -> tuple[str, str, str]:
        sym = self._order_symbol_for_signal(signal)
        if self.settings.trade_instrument == "options":
            return (
                self.groww.TRANSACTION_TYPE_BUY,
                self.groww.TRANSACTION_TYPE_SELL,
                sym,
            )
        if signal == "BUY":
            return (
                self.groww.TRANSACTION_TYPE_BUY,
                self.groww.TRANSACTION_TYPE_SELL,
                sym,
            )
        return (
            self.groww.TRANSACTION_TYPE_SELL,
            self.groww.TRANSACTION_TYPE_BUY,
            sym,
        )

    def _orders_enabled(self) -> bool:
        return (not self.dry_run) and (not self.settings.alerts_only)

    def _place_market(
        self,
        transaction_type: str,
        qty: int,
        *,
        trading_symbol: str | None = None,
    ) -> None:
        sym = trading_symbol or self.settings.symbol
        ref = f"NB{uuid.uuid4().hex[:16]}"
        if not self._orders_enabled():
            mode = "ORB_ALERTS_ONLY" if self.settings.alerts_only else "DRY-RUN"
            self._log(f"{mode} — skip order {transaction_type} qty={qty} {sym} ref={ref}")
            return
        resp = self.groww.place_order(
            validity=self.groww.VALIDITY_DAY,
            exchange=self.groww.EXCHANGE_NSE,
            order_type=self.groww.ORDER_TYPE_MARKET,
            product=self._product,
            quantity=qty,
            segment=self.groww.SEGMENT_FNO,
            trading_symbol=sym,
            transaction_type=transaction_type,
            order_reference_id=ref,
        )
        self._log(f"ORDER {transaction_type} qty={qty} {sym} -> {resp}")

    def _execute_events(self, events: list[tuple[str, dict]]) -> None:
        for kind, payload in events:
            if kind == "ENTRY":
                trade = payload
                sig = str(trade["type"])
                ent_tx, _xit_tx, osym = self._entry_exit_orders(sig)
                opt_leg = f" long {'CE' if sig == 'BUY' else 'PE'}" if self.settings.trade_instrument == "options" else ""
                self._log(
                    f"{datetime.now(IST).strftime('%H:%M:%S')} | {sig}{opt_leg} | "
                    f"Entry:{trade['entry']:.2f} SL:{trade['sl']:.2f} T:{trade['target']:.2f} "
                    f"Qty:{trade['qty']} sym:{osym}"
                )
                _dispatch_alert(
                    self.settings,
                    (
                        f"NiftyBot · ENTRY ({sig}{opt_leg})\n"
                        f"ORB: {self.settings.symbol}\n"
                        f"Order symbol: {osym}\n"
                        f"Entry {trade['entry']:.2f}  SL {trade['sl']:.2f}  T {trade['target']:.2f}\n"
                        f"Qty {trade['qty']}\n"
                        f"IST {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    self._log,
                    kind="entry",
                    symbol=self.settings.symbol,
                    extra={"side": sig, "order_symbol": osym, "trade": trade},
                )
                self._place_market(ent_tx, int(trade["qty"]), trading_symbol=osym)
                if self._engine is not None and getattr(self._engine, "trade", None):
                    self._engine.trade["order_symbol"] = osym
            elif kind == "EXIT":
                tr = payload["trade"]
                sig = str(tr["type"])
                _ent_tx, xit_tx, _ = self._entry_exit_orders(sig)
                osym = tr.get("order_symbol") or self._order_symbol_for_signal(sig)
                pnl = float(payload["pnl"])
                self.pnl_history.append(pnl)
                cap = self._engine.capital if self._engine else self.capital
                _dispatch_alert(
                    self.settings,
                    (
                        f"NiftyBot · EXIT ({sig})\n"
                        f"ORB: {self.settings.symbol}\n"
                        f"Order symbol: {osym}\n"
                        f"Exit {payload['exit_price']:.2f}  PnL {pnl:+.2f}\n"
                        f"Capital (sim) {cap:,.2f}\n"
                        f"IST {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    self._log,
                    kind="exit",
                    symbol=self.settings.symbol,
                    extra={
                        "side": sig,
                        "order_symbol": osym,
                        "pnl": pnl,
                        "exit_price": payload["exit_price"],
                        "capital_sim": cap,
                    },
                )
                self._place_market(xit_tx, int(tr["qty"]), trading_symbol=osym)
                self._log(
                    f"{datetime.now(IST).strftime('%H:%M:%S')} | EXIT | {sig} "
                    f"@{payload['exit_price']:.2f} PnL:{pnl:.2f} Capital:{cap:.2f}"
                )

    def _load_df_with_indicators(self) -> pd.DataFrame | None:
        now_ts = pd.Timestamp.now(tz=IST)
        latest_closed_bar_start = (now_ts.floor("5min") - pd.Timedelta(minutes=5)).tz_convert(IST)
        if (
            self._cached_hist_df is not None
            and not self._cached_hist_df.empty
            and self._cached_hist_closed_bar_ts is not None
            and latest_closed_bar_start <= self._cached_hist_closed_bar_ts
        ):
            return self._cached_hist_df

        now = datetime.now(IST)
        start = datetime.combine((now - timedelta(days=self.warmup_days)).date(), datetime.min.time()).replace(
            tzinfo=IST
        )
        df = fetch_historical_5m(
            self.groww,
            trading_symbol=self.settings.symbol,
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=now.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if df.empty:
            self._cached_hist_df = None
            return None
        out = compute_indicators(
            df,
            choppy_range_mean_max=self.settings.orb.choppy_range_mean_max,
        )
        self._cached_hist_df = out
        self._cached_hist_closed_bar_ts = latest_closed_bar_start
        return out

    def _live_status_base(self) -> dict:
        return {
            "mode": "session_orb_v2_5m",
            "symbol": self.settings.symbol,
            "instrument": self.settings.trade_instrument,
            "poll_seconds": self.poll_seconds,
            "dry_run": self.dry_run,
            "alerts_only": self.settings.alerts_only,
            "orders_enabled": self._orders_enabled(),
            "ui_bot_on": live_bot_enabled(),
            "now_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }

    def run_once(self, *, single_poll: bool = False) -> None:
        base = self._live_status_base()
        if not live_bot_enabled():
            self._startup_telegram_sent = False
            write_live_status(
                {
                    **base,
                    "status": "paused",
                    "headline": "Bot OFF in UI",
                    "detail": "Turn **Bot ON** in Streamlit sidebar (niftybot_ui_prefs.json) so the live process polls Groww.",
                }
            )
            return
        _telegram_bot_started(
            self.settings,
            self._log,
            self,
            mode="Session ORB v2 (5m closed bars)",
            dry_run=self.dry_run,
            alerts_only=self.settings.alerts_only,
            poll_seconds=self.poll_seconds,
            symbol=self.settings.symbol,
            trade_instrument=self.settings.trade_instrument,
            orders_enabled=self._orders_enabled(),
            once=single_poll,
        )
        df = self._load_df_with_indicators()
        if df is None or df.empty:
            self._log(f"{datetime.now(IST).strftime('%H:%M:%S')} | No candle data")
            write_live_status(
                {
                    **base,
                    "status": "no_candles",
                    "headline": "No candle data",
                    "detail": "Groww returned no 5m bars for the warmup window — check ORB symbol and API.",
                }
            )
            return

        now = pd.Timestamp.now(tz=IST)
        today = now.date()
        ix_ist = df.index.tz_convert(IST)
        day_mask = pd.Series([t.date() == today for t in ix_ist], index=df.index, dtype=bool)
        day_df = df.loc[day_mask].sort_index()
        if day_df.empty:
            write_live_status(
                {
                    **base,
                    "status": "no_session_bars",
                    "headline": "No bars for today",
                    "detail": "Historical series has no 5m rows for today’s IST date (holiday or symbol mismatch).",
                }
            )
            return

        bar_end = day_df.index + pd.Timedelta(minutes=5)
        completed = day_df.loc[bar_end <= now]
        if completed.empty:
            first_start = day_df.index.min()
            write_live_status(
                {
                    **base,
                    "status": "waiting_first_bar",
                    "headline": "Waiting for first closed 5m bar",
                    "detail": f"No bar has fully closed yet today (first bar starts {first_start}).",
                    "first_bar_start_ist": str(first_start),
                }
            )
            return

        if self._last_bar_ts is not None:
            pending = completed.loc[completed.index > self._last_bar_ts]
        else:
            pending = completed

        if pending.empty:
            last_ts = completed.index.max()
            next_edge = last_ts + pd.Timedelta(minutes=5)
            eng = self._engine
            orb_u = getattr(eng, "_orb_ui", {}) if eng is not None else {}
            bar_u = getattr(eng, "_bar_ui", {}) if eng is not None else {}
            write_live_status(
                {
                    **base,
                    "status": "caught_up",
                    "headline": "Caught up — waiting for next 5m close",
                    "detail": f"Strategy processed through bar starting {last_ts}. Next check after {next_edge} (or poll).",
                    "last_processed_bar_start_ist": str(last_ts),
                    "next_bar_close_after_ist": str(next_edge),
                    "orb": orb_u,
                    "bar": bar_u,
                    "engine": (
                        {
                            "capital": float(eng.capital),
                            "trades_today": int(eng.trades_today),
                            "max_trades_today": int(eng.max_day),
                            "in_trade": bool(eng.in_trade),
                        }
                        if eng is not None
                        else {}
                    ),
                }
            )
            return

        day_full = df.loc[day_mask].sort_index()

        for ts, row in pending.sort_index().iterrows():
            d = ts.tz_convert(IST).date()
            self._rollover_if_new_day(d)
            assert self._engine is not None
            cum = day_full.loc[:ts]
            self._engine.try_set_orb(cum)
            self._engine.on_bar(row)
            self._execute_events(self._engine.drain_events())
            self.capital = self._engine.capital
            self._last_bar_ts = ts

        eng = self._engine
        assert eng is not None
        orb_u = getattr(eng, "_orb_ui", {}) or {}
        bar_u = getattr(eng, "_bar_ui", {}) or {}
        last_ts = self._last_bar_ts
        phase = bar_u.get("phase", "—")
        why = bar_u.get("why", "—")
        headline = {
            "orb_not_locked": "ORB range not ready",
            "still_inside_orb_window": "Inside ORB window (no entries yet)",
            "skipped_choppy": "Bar skipped (choppy)",
            "no_atr": "Indicators warming up",
            "flat_daily_cap": "Daily trade limit reached",
            "long_breakout_blocked": "Breakout up — filters blocked entry",
            "short_breakout_blocked": "Breakout down — filters blocked entry",
            "no_breakout": "No breakout vs ORB range",
            "entry_buy": "ENTRY: long (BUY)",
            "entry_sell": "ENTRY: short (SELL)",
            "in_trade": "In open position",
            "exited": "Exit filled",
            "long_breakout": "Long breakout skipped (risk)",
            "short_breakout": "Short breakout skipped (risk)",
        }.get(str(phase), f"Phase: {phase}")

        write_live_status(
            {
                **base,
                "status": "ok",
                "headline": headline,
                "detail": why,
                "last_processed_bar_start_ist": str(last_ts) if last_ts is not None else None,
                "orb": orb_u,
                "bar": bar_u,
                "engine": {
                    "capital": float(eng.capital),
                    "trades_today": int(eng.trades_today),
                    "max_trades_today": int(eng.max_day),
                    "in_trade": bool(eng.in_trade),
                },
            }
        )

    def run_forever(self) -> None:
        self._log(
            f"LIVE Session ORB v2 symbol={self.settings.symbol} "
            f"orders={self.settings.trade_instrument} "
            f"dry_run={self.dry_run} alerts_only={self.settings.alerts_only} "
            f"ui_bot={'on' if live_bot_enabled() else 'off'} "
            f"ui_telegram={'on' if telegram_alerts_enabled() else 'off'} "
            f"telegram_creds={'on' if (self.settings.telegram_bot_token and self.settings.telegram_chat_id) else 'off'} "
            f"poll={self.poll_seconds}s"
        )
        while True:
            try:
                self.run_once(single_poll=False)
            except KeyboardInterrupt:
                self._log("Stopped by user.")
                break
            except Exception as exc:  # noqa: BLE001
                self._log(f"loop error: {exc!r}")
            time.sleep(self.poll_seconds)


@dataclass
class QuoteORBRunner:
    """
    Live ORB using Groww get_quote ticks: 9:15–9:45 IST range from buffered prices,
    then breakout signals vs VWAP / ATR-style volatility (your script logic).
    """

    groww: GrowwAPI
    settings: Settings
    dry_run: bool = True
    poll_seconds: float = 5.0
    max_buffer: int = 50_000
    # Live: many ticks in ORB window. 5m replay backtest: fewer bars — use 6.
    min_orb_ticks: int = 10

    data_buffer: list[dict] = field(default_factory=list)
    in_trade: bool = False
    trade: dict = field(default_factory=dict)
    range_high: float | None = None
    range_low: float | None = None
    orb_done: bool = False
    last_signal_time: pd.Timestamp | None = None
    _startup_telegram_sent: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.capital = self.settings.capital
        self.pnl_history: list[float] = []
        self._current_date: date | None = None
        self._product = (
            self.groww.PRODUCT_MIS
            if self.settings.fno_product == "MIS"
            else self.groww.PRODUCT_NRML
        )

    def _rollover_if_new_day(self, tick_time: pd.Timestamp) -> None:
        d = tick_time.tz_convert(IST).date()
        if self._current_date is not None and d != self._current_date:
            self.data_buffer.clear()
            self.orb_done = False
            self.range_high = None
            self.range_low = None
            self.last_signal_time = None
            self.in_trade = False
            self.trade = {}
            self._log(f"{datetime.now(IST).strftime('%H:%M:%S')} | New session day — state reset")
        self._current_date = d

    def _log(self, msg: str) -> None:
        if self.settings.debug:
            print(msg)

    def fetch_live(self) -> dict | None:
        """Quote always follows `ORB_SYMBOL` (future/index) so ORB matches underlying."""
        try:
            q = self.groww.get_quote(
                exchange=self.groww.EXCHANGE_NSE,
                segment=self.groww.SEGMENT_FNO,
                trading_symbol=self.settings.symbol,
            )
            price = _safe(q.get("last_price"))
            if price is None:
                for k in ("ltp", "last_traded_price", "close"):
                    v = q.get(k)
                    if v is not None:
                        price = float(v)
                        break
            if price is None:
                return None

            volume = float(_safe(q.get("volume"), 0) or 0)
            return {
                "time": pd.Timestamp.now(tz=IST),
                "price": float(price),
                "volume": volume,
            }
        except Exception as exc:  # noqa: BLE001
            self._log(f"{datetime.now(IST).strftime('%H:%M:%S')} | ERROR fetch {exc!r}")
            return None

    def build_df(self) -> pd.DataFrame | None:
        if len(self.data_buffer) < 20:
            return None

        df = pd.DataFrame(self.data_buffer)
        df["tp"] = df["price"]
        df["cum_vol"] = df["volume"].cumsum()
        df["cum_tp_vol"] = (df["tp"] * df["volume"]).cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
        df["tr"] = df["price"].diff().abs()
        df["atr"] = df["tr"].rolling(14).mean()
        return df

    def update_orb(self, df: pd.DataFrame) -> None:
        if self.orb_done:
            return

        last_ts = df["time"].iloc[-1]
        market_open = _market_open_ist(last_ts)
        cutoff = market_open + pd.Timedelta(minutes=30)
        orb_df = df[(df["time"] >= market_open) & (df["time"] <= cutoff)]

        if len(orb_df) >= self.min_orb_ticks:
            self.range_high = float(orb_df["price"].max())
            self.range_low = float(orb_df["price"].min())
            self.orb_done = True
            self._log(
                f"{datetime.now(IST).strftime('%H:%M:%S')} | ORB locked "
                f"High:{self.range_high:.2f} Low:{self.range_low:.2f}"
            )

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        if self.range_high is None or self.range_low is None:
            return None

        row = df.iloc[-1]
        price = row["price"]
        vwap = row["vwap"]
        atr = row["atr"]

        if pd.isna(vwap) or pd.isna(atr):
            return None

        if self.last_signal_time is not None and self.last_signal_time == row["time"]:
            return None

        if len(df) < 5:
            return None

        momentum = abs(df["price"].iloc[-1] - df["price"].iloc[-5])

        if price > self.range_high and price > vwap and momentum > atr:
            self.last_signal_time = row["time"]
            return "BUY"

        if price < self.range_low and price < vwap and momentum > atr:
            self.last_signal_time = row["time"]
            return "SELL"

        return None

    def _order_symbol_for_signal(self, signal: str) -> str:
        return resolve_option_leg_for_signal(
            self.groww,
            self.settings,
            signal,
            log_info=self._log,
        )

    def _entry_exit_orders(self, signal: str) -> tuple[str, str, str]:
        """
        Returns (entry_transaction_type, exit_transaction_type, trading_symbol).

        Futures: BUY → buy/sell underlying; SELL → short buyback.
        Options: both legs are long option — always BUY to open, SELL to close on CE or PE.
        """
        sym = self._order_symbol_for_signal(signal)
        if self.settings.trade_instrument == "options":
            return (
                self.groww.TRANSACTION_TYPE_BUY,
                self.groww.TRANSACTION_TYPE_SELL,
                sym,
            )
        if signal == "BUY":
            return (
                self.groww.TRANSACTION_TYPE_BUY,
                self.groww.TRANSACTION_TYPE_SELL,
                sym,
            )
        return (
            self.groww.TRANSACTION_TYPE_SELL,
            self.groww.TRANSACTION_TYPE_BUY,
            sym,
        )

    def _orders_enabled(self) -> bool:
        return (not self.dry_run) and (not self.settings.alerts_only)

    def _place_market(
        self,
        transaction_type: str,
        qty: int,
        *,
        trading_symbol: str | None = None,
    ) -> None:
        sym = trading_symbol or self.settings.symbol
        ref = f"NB{uuid.uuid4().hex[:16]}"
        if not self._orders_enabled():
            mode = "ORB_ALERTS_ONLY" if self.settings.alerts_only else "DRY-RUN"
            self._log(f"{mode} — skip order {transaction_type} qty={qty} {sym} ref={ref}")
            return
        resp = self.groww.place_order(
            validity=self.groww.VALIDITY_DAY,
            exchange=self.groww.EXCHANGE_NSE,
            order_type=self.groww.ORDER_TYPE_MARKET,
            product=self._product,
            quantity=qty,
            segment=self.groww.SEGMENT_FNO,
            trading_symbol=sym,
            transaction_type=transaction_type,
            order_reference_id=ref,
        )
        self._log(f"ORDER {transaction_type} qty={qty} {sym} -> {resp}")

    def handle_trade(self, signal: str | None, df: pd.DataFrame) -> None:
        row = df.iloc[-1]
        price = row["price"]
        atr = row["atr"]

        if pd.isna(atr):
            return

        if not self.in_trade and signal:
            if signal == "BUY":
                sl = price - atr
                target = price + self.settings.rr * (price - sl)
            else:
                sl = price + atr
                target = price - self.settings.rr * (sl - price)

            risk = abs(price - sl)
            qty = max(int((self.capital * self.settings.risk_per_trade) / risk), 1)

            ent_tx, xit_tx, osym = self._entry_exit_orders(signal)
            self.trade = {
                "type": signal,
                "entry": price,
                "sl": sl,
                "target": target,
                "qty": qty,
                "order_symbol": osym,
                "exit_tx": xit_tx,
            }
            opt_leg = f" long {'CE' if signal == 'BUY' else 'PE'}" if self.settings.trade_instrument == "options" else ""
            self._log(
                f"{datetime.now(IST).strftime('%H:%M:%S')} | {signal}{opt_leg} | "
                f"Entry:{price:.2f} SL:{sl:.2f} Target:{target:.2f} Qty:{qty} sym:{osym}"
            )
            _dispatch_alert(
                self.settings,
                (
                    f"NiftyBot · ENTRY quote-ORB ({signal}{opt_leg})\n"
                    f"Quote: {self.settings.symbol}\n"
                    f"Order symbol: {osym}\n"
                    f"Entry {price:.2f}  SL {sl:.2f}  T {target:.2f}\n"
                    f"Qty {qty}\n"
                    f"IST {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}"
                ),
                self._log,
                kind="entry_quote",
                symbol=self.settings.symbol,
                extra={"side": signal, "order_symbol": osym, "entry": price, "sl": sl, "target": target, "qty": qty},
            )
            self._place_market(ent_tx, qty, trading_symbol=osym)
            self.in_trade = True

        elif self.in_trade:
            exit_price: float | None = None
            t = self.trade["type"]
            if t == "BUY":
                if price <= self.trade["sl"]:
                    exit_price = self.trade["sl"]
                elif price >= self.trade["target"]:
                    exit_price = self.trade["target"]
            else:
                if price >= self.trade["sl"]:
                    exit_price = self.trade["sl"]
                elif price <= self.trade["target"]:
                    exit_price = self.trade["target"]

            if exit_price is not None:
                if t == "BUY":
                    pnl = (exit_price - self.trade["entry"]) * self.trade["qty"]
                else:
                    pnl = (self.trade["entry"] - exit_price) * self.trade["qty"]

                self.capital += pnl
                self.pnl_history.append(pnl)
                osym = self.trade.get("order_symbol", self.settings.symbol)
                _dispatch_alert(
                    self.settings,
                    (
                        f"NiftyBot · EXIT quote-ORB ({t})\n"
                        f"Quote: {self.settings.symbol}\n"
                        f"Order symbol: {osym}\n"
                        f"Exit {exit_price:.2f}  PnL {pnl:+.2f}\n"
                        f"Capital (sim) {self.capital:,.2f}\n"
                        f"IST {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    self._log,
                    kind="exit_quote",
                    symbol=self.settings.symbol,
                    extra={
                        "side": t,
                        "order_symbol": osym,
                        "pnl": pnl,
                        "exit_price": exit_price,
                        "capital_sim": self.capital,
                    },
                )
                self._place_market(self.trade["exit_tx"], self.trade["qty"], trading_symbol=osym)
                self._log(
                    f"{datetime.now(IST).strftime('%H:%M:%S')} | EXIT | {t} "
                    f"Exit:{exit_price:.2f} PnL:{pnl:.2f} Capital:{self.capital:.2f}"
                )
                self.in_trade = False
                self.trade = {}

    def feed_tick(self, tick: dict) -> None:
        """Append one {time, price, volume} sample and run ORB / signal / trade step."""
        self._rollover_if_new_day(tick["time"])
        self.data_buffer.append(tick)
        if len(self.data_buffer) > self.max_buffer:
            self.data_buffer = self.data_buffer[-self.max_buffer :]

        df = self.build_df()
        if df is None:
            return

        self.update_orb(df)
        if self.orb_done:
            sig = self.generate_signal(df)
            self.handle_trade(sig, df)

    def run_once(self, *, single_poll: bool = False) -> None:
        if not live_bot_enabled():
            self._startup_telegram_sent = False
            return
        _telegram_bot_started(
            self.settings,
            self._log,
            self,
            mode="Quote ORB (get_quote ticks)",
            dry_run=self.dry_run,
            alerts_only=self.settings.alerts_only,
            poll_seconds=self.poll_seconds,
            symbol=self.settings.symbol,
            trade_instrument=self.settings.trade_instrument,
            orders_enabled=self._orders_enabled(),
            once=single_poll,
        )
        tick = self.fetch_live()
        if not tick:
            return
        self.feed_tick(tick)

    def run_forever(self) -> None:
        self._log(
            f"LIVE quote ORB quote={self.settings.symbol} "
            f"orders={self.settings.trade_instrument} "
            f"dry_run={self.dry_run} alerts_only={self.settings.alerts_only} "
            f"ui_bot={'on' if live_bot_enabled() else 'off'} "
            f"ui_telegram={'on' if telegram_alerts_enabled() else 'off'} "
            f"telegram_creds={'on' if (self.settings.telegram_bot_token and self.settings.telegram_chat_id) else 'off'} "
            f"poll={self.poll_seconds}s"
        )
        while True:
            try:
                self.run_once(single_poll=False)
            except KeyboardInterrupt:
                self._log("Stopped by user.")
                break
            except Exception as exc:  # noqa: BLE001
                self._log(f"loop error: {exc!r}")
            time.sleep(self.poll_seconds)


# Default live bot uses Groww Live Data (`get_quote`) polling.
ORBLiveBot = QuoteORBRunner
