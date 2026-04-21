#!/usr/bin/env python3
"""CLI: backtest ORB on history, or run live polling bot."""

from __future__ import annotations

import argparse
from datetime import datetime

from niftybot.config import load_settings
from niftybot.live_bot import ORBLiveBot
from niftybot.ui_prefs import live_bot_enabled, live_execute_enabled, load_ui_prefs
from niftybot.quote_backtest import print_quote_report, run_quote_orb_backtest
from niftybot.session import build_groww_client
from niftybot.strategy import HighAccuracyORB
from niftybot.tuner import grid_search_orb, print_tune_report


def cmd_backtest(args: argparse.Namespace) -> None:
    settings = load_settings()
    groww = build_groww_client(settings)
    bot = HighAccuracyORB(
        groww,
        symbol=settings.symbol,
        capital=settings.capital,
        risk_per_trade=settings.risk_per_trade,
        debug=settings.debug,
        params=settings.orb,
    )
    bot.run(args.start, args.end)


def cmd_tune(args: argparse.Namespace) -> None:
    settings = load_settings()
    groww = build_groww_client(settings)
    best, train_s, test_s = grid_search_orb(
        groww,
        symbol=settings.symbol,
        start_time=args.start,
        end_time=args.end,
        initial_capital=settings.capital,
        risk_per_trade=settings.risk_per_trade,
        train_frac=args.train_frac,
        target_win_rate=args.target_win_rate,
        min_trades_train=args.min_trades,
    )
    print_tune_report(best, train_s, test_s, target_win_rate=args.target_win_rate)


def cmd_backtest_quote(args: argparse.Namespace) -> None:
    settings = load_settings()
    groww = build_groww_client(settings)
    runner = run_quote_orb_backtest(
        groww,
        settings,
        start_time=args.start,
        end_time=args.end,
        min_orb_ticks=args.min_orb_ticks,
    )
    print_quote_report(runner)


def cmd_live(args: argparse.Namespace) -> None:
    settings = load_settings()
    prefs = load_ui_prefs()
    ui_orders = bool(prefs.get("execute_orders", False))
    execute_live = live_execute_enabled(cli_execute=args.execute, alerts_only=settings.alerts_only)

    if not prefs.get("bot_enabled", True):
        print("Live bot: Bot OFF (UI) — no Groww polling until ON in Streamlit sidebar.")
    if not prefs.get("telegram_alerts_enabled", True):
        print("Live bot: Telegram alerts OFF (UI) — SQLite alert log still writes.")
    elif not (settings.telegram_bot_token and settings.telegram_chat_id):
        print("Live bot: Telegram ON (UI) but set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to deliver.")

    if settings.alerts_only:
        print("ORB_ALERTS_ONLY=1 — broker orders disabled (Telegram/log only).")
    elif execute_live:
        parts = []
        if args.execute:
            parts.append("CLI --execute")
        if ui_orders:
            parts.append("UI: Place broker orders")
        print(f"Live bot: broker orders ON ({' + '.join(parts)}).")
    else:
        print(
            "Live bot: broker orders OFF (dry-run). "
            "Enable in UI (sidebar) and/or run with --execute; blocked while ORB_ALERTS_ONLY=1."
        )

    if args.once and not live_bot_enabled():
        return

    groww = build_groww_client(settings)
    runner = ORBLiveBot(
        groww,
        settings,
        dry_run=not execute_live,
        poll_seconds=args.poll,
    )
    if args.once:
        runner.run_once(single_poll=True)
    else:
        runner.run_forever()


def main() -> None:
    p = argparse.ArgumentParser(description="NiftyBot ORB on Groww")
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser(
        "backtest",
        help="Historical 5m Session ORB v2 (structural stops, confirmations, multi-fill/day cap)",
    )
    pb.add_argument(
        "--start",
        default="2026-04-02 09:15:00",
        help="Start time (IST wall string as returned by API)",
    )
    pb.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d 15:30:00"),
        help="End time",
    )
    pb.set_defaults(func=cmd_backtest)

    pq = sub.add_parser(
        "backtest-quote",
        help="Replay Session ORB v2 on historical 5m (same engine as backtest / live)",
    )
    pq.add_argument("--start", default="2026-04-02 09:15:00", help="Start time")
    pq.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d 15:30:00"),
        help="End time",
    )
    pq.add_argument(
        "--min-orb-ticks",
        type=int,
        default=6,
        dest="min_orb_ticks",
        help="Deprecated no-op; ORB length is ORB_MIN_ORB_CANDLES / settings.orb",
    )
    pq.set_defaults(func=cmd_backtest_quote)

    pt = sub.add_parser(
        "tune",
        help="Grid-search ORB params (1 trade/day); reports train + holdout win rate / PnL",
    )
    pt.add_argument("--start", default="2026-04-02 09:15:00", help="Start time")
    pt.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d 15:30:00"),
        help="End time",
    )
    pt.add_argument(
        "--train-frac",
        type=float,
        default=0.65,
        dest="train_frac",
        help="Fraction of session days used for training split (default 0.65)",
    )
    pt.add_argument(
        "--target-win-rate",
        type=float,
        default=0.60,
        dest="target_win_rate",
        help="Objective soft-target win rate (default 0.60)",
    )
    pt.add_argument(
        "--min-trades",
        type=int,
        default=3,
        dest="min_trades",
        help="Minimum completed trades on train split to score a config (default 3)",
    )
    pt.set_defaults(func=cmd_tune)

    pl = sub.add_parser(
        "live",
        help="Session ORB v2 on closed 5m candles (same engine as backtest); optional quote-ORB via code",
    )
    pl.add_argument(
        "--execute",
        action="store_true",
        help="Place real orders (also if enabled in Streamlit UI; never when ORB_ALERTS_ONLY=1)",
    )
    pl.add_argument(
        "--once",
        action="store_true",
        help="Single poll then exit (for cron/systemd)",
    )
    pl.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Seconds between quote polls (default: 5)",
    )
    pl.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
