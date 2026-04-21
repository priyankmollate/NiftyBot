#!/usr/bin/env python3
"""
NiftyBot web UI: strategy snapshot, Session ORB v2 backtest, optional grid tune.

Run from repo root (loads `.env` like the CLI):

  streamlit run streamlit_app.py

Live trading stays on `python main.py live` — long-poll loop is not hosted here.
ORB symbol from the sidebar overrides `ORB_SYMBOL` in prefs JSON (same file as Bot ON/OFF).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from niftybot.alerts_store import db_path, fetch_recent_alerts, record_alert
from niftybot.live_status import read_live_status
from niftybot.config import Settings, load_settings
from niftybot.data_feed import fetch_historical_5m
from niftybot.session import build_groww_client
from niftybot.telegram_alerts import send_telegram_message, verify_telegram_bot_token
from niftybot.ui_prefs import load_ui_prefs, save_ui_prefs

_UI_IST = ZoneInfo("Asia/Kolkata")
from niftybot.strategy import HighAccuracyORB
from niftybot.tuner import grid_search_orb


def _mask_secret(s: str | None, *, keep: int = 4) -> str:
    if not s:
        return "—"
    if len(s) <= keep * 2:
        return "•" * min(len(s), 8)
    return f"{s[:keep]}…{s[-keep:]}"


def _load_settings_safe() -> Settings | None:
    try:
        s = load_settings()
        st.session_state.pop("settings_error", None)
        return s
    except ValueError as e:
        st.session_state["settings_error"] = str(e)
        return None


def _notify_bot_ui_toggle(settings: Settings, *, enabled: bool) -> None:
    """Telegram + SQLite when Streamlit Bot ON/OFF changes (uses .env Telegram creds)."""
    state = "ON" if enabled else "OFF"
    kind = "bot_ui_on" if enabled else "bot_ui_off"
    body = (
        f"NiftyBot · Bot {state}\n"
        f"Source: Streamlit UI (niftybot_ui_prefs.json)\n"
        f"ORB symbol: {settings.symbol}\n"
        f"IST {datetime.now(_UI_IST).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    record_alert(kind=kind, body=body, symbol=settings.symbol, extra={"enabled": enabled, "source": "streamlit"})
    tok, chat = settings.telegram_bot_token, settings.telegram_chat_id
    if not tok or not chat:
        st.info("Bot state saved. Add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env to receive this on Telegram.")
        return
    ok, err = send_telegram_message(bot_token=tok, chat_id=chat, text=body)
    if not ok:
        st.warning(f"Telegram send failed: {err[:300]}")


def _render_live_market_status() -> None:
    """Shows last JSON snapshot from `python main.py live` (Session ORB v2)."""
    st.subheader("Live market status")
    st.caption(
        "Updated by the **CLI live process** (`python main.py live`) into `niftybot_live_status.json`. "
        "The Streamlit app only **reads** this file; run `live` in a terminal. "
        "With Streamlit ≥1.33, this block auto-refreshes every 8s."
    )
    frag = getattr(st, "fragment", None)
    if frag is not None:
        try:

            @frag(run_every=timedelta(seconds=8))
            def _panel() -> None:
                _live_status_inner()

            _panel()
            return
        except (TypeError, ValueError):
            pass
    c0, c1 = st.columns([3, 1])
    with c0:
        _live_status_inner()
    with c1:
        if st.button("Refresh", key="nb_live_status_refresh"):
            st.rerun()


def _live_status_inner() -> None:
    live = read_live_status()
    if not live:
        st.warning(
            "No status file yet. Start **`python main.py live`** with **Bot ON** in the sidebar — "
            "after the first poll, ORB/engine details appear here."
        )
        return

    updated = live.get("updated_at_ist", "—")
    st.caption(f"Last live poll write: **{updated}** (IST)")

    hl = live.get("headline") or live.get("status", "—")
    det = live.get("detail", "")
    st.markdown(f"### {hl}")
    if det:
        st.info(det)

    row = live.get("bar") or {}
    orb = live.get("orb") or {}
    eng = live.get("engine") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ORB locked", "yes" if orb.get("locked") else "no")
    rh, rl = orb.get("range_high"), orb.get("range_low")
    m2.metric("ORB high", f"{rh:.2f}" if isinstance(rh, (int, float)) else "—")
    m3.metric("ORB low", f"{rl:.2f}" if isinstance(rl, (int, float)) else "—")
    phase = row.get("phase", "—")
    m4.metric("Bar phase", str(phase)[:24] + ("…" if len(str(phase)) > 24 else ""))

    c_long = row.get("confirmations_long")
    c_short = row.get("confirmations_short")
    mc = row.get("min_confirmations")
    if c_long is not None:
        st.caption(
            f"Confirmations (long / short): **{c_long}** / **{c_short}** · need **{mc}** · "
            f"last bar **{row.get('bar_start_ist', '—')}** · close **{row.get('close', '—')}**"
        )

    if eng.get("in_trade") and row.get("open_trade"):
        st.success(f"Open position: {row['open_trade']}")
    elif eng.get("trades_today") is not None:
        st.caption(f"Trades today: **{eng.get('trades_today')}** / **{eng.get('max_trades_today')}** · capital (sim) **₹{eng.get('capital', 0):,.0f}**")

    with st.expander("Raw ORB / bar / engine (debug)", expanded=False):
        st.json({"orb": orb, "bar": row, "engine": eng, "meta": {k: live[k] for k in live if k not in ("orb", "bar", "engine")}})


def _extract_live_events(*, symbol: str, today: datetime.date) -> pd.DataFrame:
    rows = fetch_recent_alerts(limit=500)
    out: list[dict] = []
    for r in rows:
        if (r.get("symbol") or "").strip() != symbol:
            continue
        kind = str(r.get("kind") or "")
        if kind not in {"entry", "exit", "entry_quote", "exit_quote"}:
            continue
        ts_raw = r.get("created_at")
        if not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize(_UI_IST)
            else:
                ts = ts.tz_convert(_UI_IST)
        except Exception:
            continue
        if ts.date() != today:
            continue
        extra = {}
        if r.get("extra_json"):
            try:
                extra = json.loads(r["extra_json"])
            except (TypeError, json.JSONDecodeError):
                extra = {}
        trade = extra.get("trade") if isinstance(extra, dict) else {}
        if not isinstance(trade, dict):
            trade = {}
        price = None
        if kind.startswith("entry"):
            price = trade.get("entry") or extra.get("entry")
            event = "ENTRY"
        else:
            price = extra.get("exit_price")
            event = "EXIT"
        try:
            p = float(price)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "time": ts,
                "candle_time": ts.floor("5min"),
                "price": p,
                "event": event,
                "side": str(extra.get("side") or trade.get("type") or ""),
                "order_symbol": str(extra.get("order_symbol") or trade.get("order_symbol") or ""),
                "leg": (
                    "CE"
                    if "CE" in str(extra.get("order_symbol") or trade.get("order_symbol") or "").upper()
                    else (
                        "PE"
                        if "PE" in str(extra.get("order_symbol") or trade.get("order_symbol") or "").upper()
                        else "FUT"
                    )
                ),
                "sl": float(trade["sl"]) if trade.get("sl") is not None else None,
                "target": float(trade["target"]) if trade.get("target") is not None else None,
                "qty": int(trade["qty"]) if trade.get("qty") is not None else None,
                "pnl": float(extra["pnl"]) if extra.get("pnl") is not None else None,
                "kind": kind,
            }
        )
    if not out:
        return pd.DataFrame(
            columns=[
                "time",
                "candle_time",
                "price",
                "event",
                "side",
                "order_symbol",
                "leg",
                "sl",
                "target",
                "qty",
                "pnl",
                "kind",
            ]
        )
    return pd.DataFrame(out).sort_values("time")


def _render_live_trading_chart(settings: Settings) -> None:
    """Plot a TradingView-style 5m chart with strategy overlays."""
    st.subheader("Live trading view (5m)")
    st.caption(
        "Trading-day view (09:15–15:30 IST): candlesticks, volume, ORB zone, "
        "Entry/SL/Target levels, and live ENTRY/EXIT markers."
    )
    c_left, c_right = st.columns([3, 1])
    with c_right:
        if st.button("Refresh chart", key="nb_live_chart_refresh"):
            st.rerun()
    now = datetime.now(_UI_IST)
    session_start = datetime.combine(now.date(), datetime.min.time()).replace(hour=9, minute=15, tzinfo=_UI_IST)
    session_end = datetime.combine(now.date(), datetime.min.time()).replace(hour=15, minute=30, tzinfo=_UI_IST)
    try:
        groww = build_groww_client(settings)
        raw = fetch_historical_5m(
            groww,
            trading_symbol=settings.symbol,
            start_time=session_start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=now.strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:  # noqa: BLE001
        with c_left:
            st.warning(f"Live chart data unavailable: {e}")
        return

    if raw.empty:
        with c_left:
            st.info("No 5m candle data available yet for today.")
        return

    idx_ist = raw.index.tz_convert(_UI_IST)
    day_mask = pd.Series([d.date() == now.date() for d in idx_ist], index=raw.index, dtype=bool)
    day_df = raw.loc[day_mask].copy().sort_index()
    if day_df.empty:
        with c_left:
            st.info("No bars found for today's session.")
        return

    orb = day_df.between_time(settings.orb.orb_start, settings.orb.orb_end)
    if not orb.empty:
        day_df["ORB High"] = float(orb["high"].max())
        day_df["ORB Low"] = float(orb["low"].min())
    else:
        day_df["ORB High"] = pd.NA
        day_df["ORB Low"] = pd.NA

    live = read_live_status() or {}
    row = live.get("bar") or {}
    open_trade = row.get("open_trade") if isinstance(row, dict) else None
    events = _extract_live_events(symbol=settings.symbol, today=now.date())

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )
    fig.add_trace(
        go.Candlestick(
            x=day_df.index,
            open=day_df["open"],
            high=day_df["high"],
            low=day_df["low"],
            close=day_df["close"],
            name="Price",
            increasing_line_color="#22c55e",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#ef4444",
            decreasing_fillcolor="#ef4444",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    vol_colors = ["#22c55e" if c >= o else "#ef4444" for o, c in zip(day_df["open"], day_df["close"], strict=True)]
    fig.add_trace(
        go.Bar(
            x=day_df.index,
            y=day_df["volume"],
            marker_color=vol_colors,
            opacity=0.55,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    level_rows: list[dict] = []
    if not orb.empty:
        level_rows.extend(
            [
                {"label": "ORB High", "value": float(orb["high"].max())},
                {"label": "ORB Low", "value": float(orb["low"].min())},
            ]
        )
    if isinstance(open_trade, dict):
        if open_trade.get("sl") is not None:
            level_rows.append({"label": "SL", "value": float(open_trade["sl"])})
        if open_trade.get("target") is not None:
            level_rows.append({"label": "Target", "value": float(open_trade["target"])})
        if open_trade.get("entry") is not None:
            level_rows.append({"label": "Entry", "value": float(open_trade["entry"])})

    level_colors = {
        "ORB High": "#3b82f6",
        "ORB Low": "#3b82f6",
        "Entry": "#22c55e",
        "SL": "#ef4444",
        "Target": "#f43f5e",
    }
    for lvl in level_rows:
        label = str(lvl["label"])
        value = float(lvl["value"])
        color = level_colors.get(label, "#9ca3af")
        fig.add_hline(
            y=value,
            line_color=color,
            line_width=1.2,
            line_dash="dash" if label.startswith("ORB") else "solid",
            annotation_text=label,
            annotation_position="left",
            annotation_font_color=color,
            row=1,
            col=1,
        )

    if not events.empty:
        evt = events.copy()
        evt["marker"] = evt["event"].map({"ENTRY": "triangle-up", "EXIT": "diamond"}).fillna("circle")
        evt["mcolor"] = evt["event"].map({"ENTRY": "#8b5cf6", "EXIT": "#f59e0b"}).fillna("#e5e7eb")
        evt["label"] = evt["event"]
        evt.loc[evt["event"] == "ENTRY", "label"] = (
            "ENTRY " + evt.loc[evt["event"] == "ENTRY", "side"].fillna("").astype(str)
        ).str.strip()
        evt.loc[evt["event"] == "ENTRY", "label"] = (
            evt.loc[evt["event"] == "ENTRY", "label"] + " · " + evt.loc[evt["event"] == "ENTRY", "leg"].fillna("")
        ).str.strip()
        evt.loc[(evt["event"] == "EXIT") & evt["pnl"].notna(), "label"] = evt.loc[
            (evt["event"] == "EXIT") & evt["pnl"].notna(),
            "pnl",
        ].map(lambda x: f"EXIT {x:+.1f}")
        fig.add_trace(
            go.Scatter(
                x=evt["candle_time"],
                y=evt["price"],
                mode="markers+text",
                text=evt["label"],
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(size=11, color=evt["mcolor"], symbol=evt["marker"], line=dict(width=1, color="#111827")),
                customdata=evt[["event", "side", "leg", "order_symbol", "time", "sl", "target", "qty", "pnl"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                    "Leg: %{customdata[2]}<br>"
                    "Strike/Symbol: %{customdata[3]}<br>"
                    "Alert: %{customdata[4]}<br>"
                    "Price: %{y:.2f}<br>"
                    "SL: %{customdata[5]:.2f}<br>"
                    "Target: %{customdata[6]:.2f}<br>"
                    "Qty: %{customdata[7]}<br>"
                    "PnL: %{customdata[8]:.2f}<extra></extra>"
                ),
                showlegend=False,
                name="Events",
            ),
            row=1,
            col=1,
        )

    orb_start = datetime.combine(now.date(), datetime.min.time()).replace(
        hour=int(settings.orb.orb_start.split(":")[0]),
        minute=int(settings.orb.orb_start.split(":")[1]),
        tzinfo=_UI_IST,
    )
    orb_end = datetime.combine(now.date(), datetime.min.time()).replace(
        hour=int(settings.orb.orb_end.split(":")[0]),
        minute=int(settings.orb.orb_end.split(":")[1]),
        tzinfo=_UI_IST,
    )
    fig.add_vrect(
        x0=orb_start,
        x1=orb_end,
        fillcolor="#1d4ed8",
        opacity=0.10,
        line_width=0,
        annotation_text="ORB",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=orb_start,
        x1=orb_end,
        fillcolor="#1d4ed8",
        opacity=0.08,
        line_width=0,
        row=2,
        col=1,
    )
    current_price = float(day_df["close"].iloc[-1])
    fig.add_hline(
        y=current_price,
        line_color="#22d3ee",
        line_width=1.1,
        line_dash="dot",
        annotation_text=f"Current {current_price:.2f}",
        annotation_position="right",
        annotation_font_color="#22d3ee",
        row=1,
        col=1,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0b1020",
        paper_bgcolor="#0b1020",
        font=dict(color="#d1d5db"),
    )
    fig.update_xaxes(
        range=[session_start, session_end],
        showgrid=True,
        gridcolor="#1f2937",
        tickformat="%H:%M",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        range=[session_start, session_end],
        showgrid=True,
        gridcolor="#1f2937",
        tickformat="%H:%M",
        row=2,
        col=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937", side="right", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937", side="right", row=2, col=1, title="Vol")

    with c_left:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        last_close = float(day_df["close"].iloc[-1])
        st.metric("Current price", f"{last_close:.2f}")
        st.caption(
            f"Last bar: {day_df.index[-1].tz_convert(_UI_IST).strftime('%H:%M')} IST | "
            f"Close {last_close:.2f} | Bars today {len(day_df)}"
        )
        if not events.empty:
            show = events[["time", "event", "side", "leg", "order_symbol", "price", "sl", "target", "qty", "pnl"]].copy()
            show["time"] = pd.to_datetime(show["time"]).dt.strftime("%H:%M:%S")
            st.dataframe(show.tail(8), width="stretch", hide_index=True)
        else:
            st.caption("No ENTRY/EXIT alerts yet today for this symbol.")


def _orb_params_table(settings: Settings) -> pd.DataFrame:
    """All `value` cells as strings so Streamlit/PyArrow does not infer a numeric dtype (e.g. orb times)."""
    d = asdict(settings.orb)
    return pd.DataFrame(
        [{"parameter": k, "value": "" if v is None else str(v)} for k, v in sorted(d.items())]
    )


def main() -> None:
    st.set_page_config(
        page_title="NiftyBot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("NiftyBot")
    st.caption("Session ORB v2 on Groww · same engine as `python main.py backtest`")

    settings = _load_settings_safe()
    if settings is None:
        err = st.session_state.get("settings_error", "")
        st.error("**Configuration**")
        st.code(err or "Set GROWW_* and ORB_* in `.env` (see `env.example`).")

        st.divider()
        st.subheader("Stuck on this screen?")
        opt_stuck = (
            "options" in err.lower()
            or "ORB_OPTION" in err
            or "call and put" in err.lower()
            or "option_chain" in err.lower()
            or "auto otm" in err.lower()
            or "ORB_OPTION_AUTO" in err
        )
        if st.button(
            "Switch to **Futures** (UI override)",
            key="nb_rec_fut",
            help="Writes trade_instrument=futures to niftybot_ui_prefs.json so it overrides ORB_TRADE_INSTRUMENT=options in .env.",
        ):
            save_ui_prefs(
                {
                    "trade_instrument": "futures",
                    "option_ce_symbol": "",
                    "option_pe_symbol": "",
                }
            )
            st.session_state.pop("settings_error", None)
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("nb_rec"):
                    st.session_state.pop(k, None)
            st.rerun()
        st.caption(
            "Use the button above if you enabled Options by mistake and want the app to load without CE/PE legs."
        )

        with st.expander("Set options CE & PE (Groww trading_symbol)", expanded=bool(opt_stuck)):
            rp = load_ui_prefs()
            ce_fix = st.text_input("Call (CE)", value=rp.get("option_ce_symbol") or "", key="nb_rec_ce")
            pe_fix = st.text_input("Put (PE)", value=rp.get("option_pe_symbol") or "", key="nb_rec_pe")
            if st.button("Save CE / PE and reload", type="primary", key="nb_rec_save"):
                ce_s, pe_s = ce_fix.strip(), pe_fix.strip()
                if not ce_s or not pe_s:
                    st.warning("Enter both CE and PE symbols, then save again.")
                else:
                    save_ui_prefs({"option_ce_symbol": ce_s, "option_pe_symbol": pe_s})
                    st.session_state.pop("settings_error", None)
                    for k in list(st.session_state.keys()):
                        if isinstance(k, str) and k.startswith("nb_rec"):
                            st.session_state.pop(k, None)
                    st.rerun()

        st.stop()

    with st.sidebar:
        st.subheader("Connection")
        prefs = load_ui_prefs()
        prefs_sym = prefs.get("orb_symbol") or ""
        st.caption(
            "ORB uses Groww **trading_symbol** (e.g. `NIFTY26APRFUT`). Empty = `ORB_SYMBOL` from `.env`."
        )
        sym_in = st.text_input(
            "Stock / FNO symbol",
            value=prefs_sym,
            key="nb_orb_symbol_input",
            help="Saved to niftybot_ui_prefs.json. Used by Streamlit backtest/tune and `python main.py live` / CLI backtest.",
        )
        c_sym_a, c_sym_b = st.columns(2)
        if c_sym_a.button("Save symbol", key="nb_save_sym"):
            save_ui_prefs({"orb_symbol": sym_in.strip()})
            st.session_state.pop("nb_orb_symbol_input", None)
            st.success("Saved. Rerun uses this symbol.")
            st.rerun()
        if c_sym_b.button("Clear", key="nb_clear_sym"):
            save_ui_prefs({"orb_symbol": ""})
            st.session_state.pop("nb_orb_symbol_input", None)
            st.success("Cleared — using ORB_SYMBOL from .env.")
            st.rerun()

        st.divider()
        st.caption(
            "**FNO only** — this bot always uses Groww `SEGMENT_FNO` (no cash / commodity). "
            "Pick **product** MIS or NRML; trades are futures-only."
        )
        prod_vals = ["", "MIS", "NRML"]
        pref_fp = (prefs.get("fno_product") or "").strip().upper()
        p_idx = prod_vals.index(pref_fp) if pref_fp in prod_vals else 0

        def _fmt_prod(v: str) -> str:
            return {"": "Use .env (FNO_PRODUCT)", "MIS": "MIS (intraday)", "NRML": "NRML (carry)"}[v]

        prod_pick = st.selectbox(
            "F&O product",
            prod_vals,
            index=p_idx,
            format_func=_fmt_prod,
            key="nb_fno_product",
        )
        c_seg_a, c_seg_b = st.columns(2)
        if c_seg_a.button("Save F&O choices", key="nb_save_seg"):
            upd: dict = {
                "fno_product": prod_pick,
                "trade_instrument": "futures",
                # Clear legacy options prefs so old values never leak into UI.
                "option_ce_symbol": "",
                "option_pe_symbol": "",
                "option_chain_expiry": "",
                "option_auto_strike": False,
            }
            save_ui_prefs(upd)
            st.session_state.pop("nb_fno_product", None)
            st.success("F&O choices saved.")
            st.rerun()
        if c_seg_b.button("Clear F&O choices", key="nb_clear_seg"):
            save_ui_prefs(
                {
                    "fno_product": "",
                    "trade_instrument": "futures",
                }
            )
            st.session_state.pop("nb_fno_product", None)
            st.success("Cleared — .env drives F&O product (futures only).")
            st.rerun()

        st.text(f"Effective ORB symbol: {settings.symbol}")
        st.text(f"Effective F&O product: {settings.fno_product}")
        st.text(f"Effective instrument: {settings.trade_instrument}")
        st.text(f"API key: {_mask_secret(settings.api_key)}")
        auth = "TOTP" if (settings.totp or settings.totp_secret) else "API secret"
        st.caption(f"Auth: {auth}")

        st.divider()
        st.subheader("Live bot (CLI)")
        t_bot = st.toggle(
            "Bot ON",
            value=prefs["bot_enabled"],
            help="When OFF, `python main.py live` does not poll Groww (strategy idle). File: niftybot_ui_prefs.json.",
            key="nb_pref_bot",
        )
        t_tg = st.toggle(
            "Telegram alerts",
            value=prefs["telegram_alerts_enabled"],
            help="When ON, sends ENTRY/EXIT/bot_started to Telegram if TELEGRAM_* are in .env. SQLite always logs.",
            key="nb_pref_tg",
        )
        t_ex = st.toggle(
            "Place broker orders",
            value=prefs["execute_orders"],
            help="When ON, live may place real orders (same as --execute). ORB_ALERTS_ONLY=1 always blocks.",
            key="nb_pref_ex",
        )
        snap = (prefs["bot_enabled"], prefs["telegram_alerts_enabled"], prefs["execute_orders"])
        if (t_bot, t_tg, t_ex) != snap:
            prev_bot = prefs["bot_enabled"]
            save_ui_prefs(
                {
                    "bot_enabled": t_bot,
                    "telegram_alerts_enabled": t_tg,
                    "execute_orders": t_ex,
                }
            )
            st.success("Saved.")
            if t_bot != prev_bot:
                _notify_bot_ui_toggle(settings, enabled=t_bot)
        if settings.alerts_only:
            st.warning("`ORB_ALERTS_ONLY=1` — broker orders never sent until you change .env.")
        else:
            st.caption("Keep Bot ON while the live process runs; use Bot OFF to pause without stopping the process.")

        st.divider()
        st.subheader("Telegram check")
        st.caption(
            "If sends fail with 401: token is wrong for Telegram, or `ORB_TELEGRAM_BOT_TOKEN` overrides "
            "`TELEGRAM_BOT_TOKEN` with an old value — keep only one set."
        )
        if st.button("Test token (getMe)", key="nb_tg_getme"):
            try:
                s2 = load_settings()
            except ValueError as e:
                st.error(str(e))
            else:
                ok, msg = verify_telegram_bot_token(s2.telegram_bot_token)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    tab_overview, tab_bt, tab_tune, tab_alerts = st.tabs(["Strategy", "Backtest", "Tune", "Alerts"])

    with tab_overview:
        _render_live_market_status()
        _render_live_trading_chart(settings)
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital", f"₹{settings.capital:,.0f}")
        c2.metric("Risk / trade", f"{settings.risk_per_trade * 100:.2f}%")
        c3.metric("R:R", f"{settings.orb.rr:g}")
        c4.metric("ORB window", f"{settings.orb.orb_start}–{settings.orb.orb_end}")

        st.subheader("ORB v2 parameters")
        st.dataframe(_orb_params_table(settings), width="stretch", hide_index=True)

        st.info("Instrument mode: **Futures-only**. Live entries/exits always use the ORB futures symbol.")

        tg_on = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        st.subheader("Live alerts")
        st.write(
            f"- **ORB_ALERTS_ONLY:** `{'yes' if settings.alerts_only else 'no'}` "
            "(when yes, `live --execute` never places orders)\n"
            f"- **Telegram:** `{'configured' if tg_on else 'not set'}` "
            f"(token `{_mask_secret(settings.telegram_bot_token)}`)"
        )

        st.subheader("Live prefs (saved JSON)")
        lp = load_ui_prefs()
        ui_sym = (lp.get("orb_symbol") or "").strip()
        sym_note = f"`{ui_sym}`" if ui_sym else "`—` (falls back to `.env` `ORB_SYMBOL`)"
        ui_fp = (lp.get("fno_product") or "").strip().upper()
        fp_note = f"`{ui_fp}`" if ui_fp in ("MIS", "NRML") else "`—` (`FNO_PRODUCT` from `.env`)"
        st.write(
            f"- **ORB symbol (sidebar):** {sym_note}\n"
            f"- **F&O product (sidebar):** {fp_note}\n"
            f"- **Instrument:** `futures` (fixed)\n"
            f"- **Bot ON:** `{'yes' if lp.get('bot_enabled', True) else 'no'}`\n"
            f"- **Telegram alerts:** `{'yes' if lp.get('telegram_alerts_enabled', True) else 'no'}` "
            "(needs `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env`)\n"
            f"- **Place broker orders:** `{'yes' if lp.get('execute_orders', False) else 'no'}` "
            "(effective with `python main.py live` + optional `--execute`; respect `ORB_ALERTS_ONLY`)"
        )

        st.divider()
        st.markdown(
            "**Live bot:** `python main.py live` — set the **sidebar** ORB symbol (saved in `niftybot_ui_prefs.json`), "
            "then use toggles for real orders or pass `--execute`. Alerts go to **Telegram** (if configured) and **SQLite** "
            f"(`{db_path().name}`); browse them in the **Alerts** tab."
        )

    with tab_bt:
        st.subheader("Session ORB backtest")
        st.caption(
            "Default: **5m candles for the ORB symbol** (`ORB_SYMBOL` / sidebar), via Groww "
            "[backtesting / historical candles](https://groww.in/trade-api/docs/python-sdk/backtesting) "
            "(`groww_symbol` from instruments). **Auto OTM** is live-only. Optional override: another FNO "
            "`trading_symbol` (e.g. one CE/PE). Groww allows **≤30 days** per 5m request — use a shorter range if "
            "the fetch fails. PnL is **(price diff)×qty** on that series (not full lot economics unless you add lot size)."
        )
        col_a, col_b = st.columns(2)
        default_end = datetime.now().strftime("%Y-%m-%d 15:30:00")
        default_start = "2026-04-02 09:15:00"
        start = col_a.text_input("Start (IST wall time)", value=default_start, key="bt_start")
        end = col_b.text_input("End", value=default_end, key="bt_end")
        bt_sym = st.text_input(
            "Historical symbol override (optional)",
            value="",
            key="bt_symbol_override",
            placeholder=f"Leave blank for {settings.symbol}",
            help="Groww FNO trading_symbol, e.g. a fixed NIFTY…CE / …PE for that expiry. Empty = ORB symbol.",
        )
        sym_bt = (bt_sym or "").strip() or settings.symbol

        if st.button("Run backtest", type="primary", key="run_bt"):
            with st.spinner("Fetching candles and running simulation…"):
                try:
                    groww = build_groww_client(settings)
                    bot = HighAccuracyORB(
                        groww,
                        symbol=sym_bt,
                        capital=settings.capital,
                        risk_per_trade=settings.risk_per_trade,
                        debug=False,
                        params=settings.orb,
                    )
                    bot.run(start, end)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Backtest failed: {e}")
                else:
                    st.session_state["last_bt"] = {
                        "trades": list(bot.trades),
                        "capital": float(bot.capital),
                        "symbol": sym_bt,
                        "range": f"{start} → {end}",
                    }

        last = st.session_state.get("last_bt")
        if last:
            trades = pd.Series(last["trades"], dtype=float, name="pnl")
            n = len(trades)
            wr = float((trades > 0).mean()) if n else 0.0
            pnl = float(trades.sum()) if n else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", n)
            m2.metric("Win rate", f"{wr * 100:.1f}%" if n else "—")
            m3.metric("PnL", f"₹{pnl:,.0f}")
            m4.metric("Ending capital", f"₹{last['capital']:,.0f}")

            st.caption(f"Data: {last.get('symbol', '')} · {last.get('range', '')}")

            if n:
                chart_df = pd.DataFrame({"Cumulative PnL": trades.cumsum()}, index=range(1, n + 1))
                st.line_chart(chart_df, height=280)
                st.caption("Each step is one closed trade (Session ORB v2); not intrabar equity.")
            else:
                st.warning("No trades in this window.")

    with tab_tune:
        st.subheader("Grid search (train / holdout)")
        st.caption("Can take several minutes — many parameter combinations.")

        tc1, tc2, tc3 = st.columns(3)
        t_start = tc1.text_input("Start", value=default_start, key="tune_start")
        t_end = tc2.text_input("End", value=default_end, key="tune_end")
        train_frac = tc3.slider("Train fraction", 0.5, 0.9, 0.65, 0.05)

        tc4, tc5 = st.columns(2)
        target_wr = tc4.slider("Target win rate", 0.45, 0.75, 0.60, 0.05)
        min_trades = tc5.number_input("Min trades (train)", min_value=1, max_value=50, value=3)

        if st.button("Run grid search", key="run_tune"):
            with st.spinner("Grid search in progress…"):
                try:
                    groww = build_groww_client(settings)
                    best_p, train_s, test_s = grid_search_orb(
                        groww,
                        symbol=settings.symbol,
                        start_time=t_start,
                        end_time=t_end,
                        initial_capital=settings.capital,
                        risk_per_trade=settings.risk_per_trade,
                        train_frac=train_frac,
                        target_win_rate=target_wr,
                        min_trades_train=int(min_trades),
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Tune failed: {e}")
                else:
                    st.session_state["last_tune"] = {
                        "best": asdict(best_p),
                        "train": train_s,
                        "test": test_s,
                        "target_wr": target_wr,
                    }

        lt = st.session_state.get("last_tune")
        if lt:
            st.success("Best configuration (copy relevant keys into `.env` if you adopt them)")
            st.json(lt["best"])

            tr, te = lt["train"], lt["test"]
            u1, u2, u3 = st.columns(3)
            u1.metric("Train trades", tr["n"])
            u2.metric(
                "Train win rate",
                f"{tr['win_rate'] * 100:.1f}%" if tr["win_rate"] is not None else "—",
            )
            u3.metric("Train PnL", f"₹{tr['pnl']:,.0f}")
            v1, v2, v3 = st.columns(3)
            v1.metric("Holdout trades", te["n"])
            v2.metric(
                "Holdout win rate",
                f"{te['win_rate'] * 100:.1f}%" if te["win_rate"] is not None else "—",
            )
            v3.metric("Holdout PnL", f"₹{te['pnl']:,.0f}")
            st.caption(
                f"Target win rate used in scoring: **{lt['target_wr']*100:.0f}%** — "
                "holdout is a rough OOS check; short samples are noisy."
            )

    with tab_alerts:
        st.subheader("Alert log (live bot)")
        st.caption(
            f"SQLite file (repo root): `{db_path()}` — written on **bot_started**, **entry**, **exit** "
            "(and quote-ORB variants). Same text as Telegram when Telegram is on."
        )
        lim = st.number_input("Rows to load", min_value=20, max_value=1000, value=200, step=20, key="alerts_limit")
        rows = fetch_recent_alerts(limit=int(lim))
        if not rows:
            st.info("No rows yet. Alerts appear after you run `python main.py live` and the bot emits signals.")
        else:
            adf = pd.DataFrame(rows)
            adf["body"] = adf["body"].astype(str).str.slice(0, 500)
            for c in adf.columns:
                adf[c] = adf[c].map(lambda x: "" if x is None else str(x))
            st.dataframe(adf, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
