import os
from dataclasses import dataclass

from dotenv import load_dotenv

from niftybot.strategy import ORBBacktestParams
from niftybot.telegram_alerts import sanitize_chat_id_for_api, sanitize_telegram_bot_token_for_api
from niftybot.ui_prefs import load_ui_prefs

load_dotenv()


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no")


def _env_bool_off(name: str, default: bool = False) -> bool:
    """Default-off flags (e.g. ORB_ALERTS_ONLY)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no")


@dataclass
class Settings:
    """Groww auth: either (api_key + api_secret) OR (api_key + totp / totp_secret)."""

    api_key: str
    api_secret: str | None
    totp: str | None
    totp_secret: str | None
    symbol: str
    """Trading symbol for quotes + ORB (use Nifty future or index FNO symbol)."""
    trade_instrument: str
    """Futures-only mode: orders are always placed on `symbol`."""
    option_ce_symbol: str | None
    option_pe_symbol: str | None
    """Deprecated in futures-only mode (kept for backward compatibility)."""
    option_auto_strike: bool
    """Deprecated in futures-only mode (kept for backward compatibility)."""
    option_chain_expiry: str | None
    """Deprecated in futures-only mode (kept for backward compatibility)."""
    option_chain_underlying: str | None
    """Deprecated in futures-only mode (kept for backward compatibility)."""
    option_strike_step: int
    """Deprecated in futures-only mode (kept for backward compatibility)."""
    option_otm_window_steps: int
    """Deprecated in futures-only mode (kept for backward compatibility)."""
    fno_product: str
    capital: float
    risk_per_trade: float
    rr: float
    debug: bool
    orb: ORBBacktestParams
    alerts_only: bool
    """If True, never place broker orders (overrides `live --execute`)."""
    telegram_bot_token: str | None
    telegram_chat_id: str | None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def load_settings() -> Settings:
    api_key = os.getenv("GROWW_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set GROWW_API_KEY in .env")

    api_secret = os.getenv("GROWW_API_SECRET", "").strip() or None
    totp = os.getenv("GROWW_TOTP", "").strip() or None
    totp_secret = os.getenv("GROWW_TOTP_SECRET", "").strip() or None

    if not api_secret and not totp and not totp_secret:
        raise ValueError(
            "Set one auth method:\n"
            "  • TOTP: GROWW_TOTP_SECRET (authenticator seed; recommended) and/or GROWW_TOTP (6-digit code), OR\n"
            "  • API key + secret: GROWW_API_SECRET (daily approval secret from Groww)."
        )

    prefs = load_ui_prefs()

    product = os.getenv("FNO_PRODUCT", "MIS").strip().upper()
    if product not in {"MIS", "NRML"}:
        raise ValueError("FNO_PRODUCT must be MIS or NRML.")
    ui_fp = (prefs.get("fno_product") or "").strip().upper()
    if ui_fp in ("MIS", "NRML"):
        product = ui_fp

    # Futures-only mode: ignore ORB_TRADE_INSTRUMENT and old UI instrument prefs.
    trade_instrument = "futures"

    option_ce = os.getenv("ORB_OPTION_CE", "").strip() or None
    option_pe = os.getenv("ORB_OPTION_PE", "").strip() or None
    ui_ce = (prefs.get("option_ce_symbol") or "").strip()
    ui_pe = (prefs.get("option_pe_symbol") or "").strip()
    if ui_ce:
        option_ce = ui_ce
    if ui_pe:
        option_pe = ui_pe

    option_chain_expiry = os.getenv("ORB_OPTION_CHAIN_EXPIRY", "").strip() or None
    ui_exp = (prefs.get("option_chain_expiry") or "").strip()
    if ui_exp:
        option_chain_expiry = ui_exp

    option_auto_strike = _env_bool_off("ORB_OPTION_AUTO_STRIKE", False)
    if "option_auto_strike" in prefs:
        option_auto_strike = bool(prefs["option_auto_strike"])

    option_strike_step = int(os.getenv("ORB_OPTION_STRIKE_STEP", "50") or 50)
    option_otm_window_steps = int(os.getenv("ORB_OPTION_OTM_WINDOW", "5") or 5)

    sym = os.getenv("ORB_SYMBOL", "NIFTY26APRFUT").strip()
    ui_sym = (prefs.get("orb_symbol") or "").strip()
    if ui_sym:
        sym = ui_sym

    option_chain_underlying = None
    option_ce = None
    option_pe = None
    option_auto_strike = False

    rr = _env_float("ORB_RR", 2.0)
    _mc_raw = os.getenv("ORB_MIN_CONFIRMATIONS", "").strip() or os.getenv("ORB_MIN_SCORE", "").strip() or "2"
    min_conf = int(_mc_raw)
    sl_cap = _env_float("ORB_SL_CAP_ATR", _env_float("ORB_SL_ATR", 1.35))
    orb = ORBBacktestParams(
        rr=rr,
        orb_start=os.getenv("ORB_ORB_START", "09:15").strip(),
        orb_end=os.getenv("ORB_ORB_END", "09:45").strip(),
        min_orb_candles=int(os.getenv("ORB_MIN_ORB_CANDLES", "6") or 6),
        orb_sl_buffer_atr=_env_float("ORB_SL_BUFFER_ATR", 0.12),
        sl_atr_cap_mult=sl_cap,
        sl_atr_mult=sl_cap,
        min_confirmations=min_conf,
        min_score=min_conf,
        volume_vs_ma_mult=_env_float("ORB_VOLUME_MA_MULT", 1.12),
        body_ratio_min=_env_float("ORB_BODY_RATIO", 0.32),
        require_vwap_side=_env_bool("ORB_REQUIRE_VWAP_SIDE", True),
        require_ema_trend=_env_bool("ORB_REQUIRE_EMA", False) or _env_bool("ORB_REQUIRE_TREND", False),
        require_trend=_env_bool("ORB_REQUIRE_TREND", False),
        skip_choppy=_env_bool("ORB_SKIP_CHOPPY", False),
        choppy_range_mean_max=_env_float("ORB_CHOPPY_MAX", 0.0016),
        max_trades_per_day=int(os.getenv("ORB_MAX_TRADES_PER_DAY", "2") or 2),
        one_trade_per_day=_env_bool("ORB_ONE_TRADE_PER_DAY", False),
        vol_spike_mult=_env_float("ORB_VOL_SPIKE_MULT", 1.5),
    )

    tg_raw = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("ORB_TELEGRAM_BOT_TOKEN", "").strip()
    tg_token = sanitize_telegram_bot_token_for_api(tg_raw or None)
    tg_chat_raw = os.getenv("TELEGRAM_CHAT_ID", "").strip() or os.getenv("ORB_TELEGRAM_CHAT_ID", "").strip()
    tg_chat = sanitize_chat_id_for_api(tg_chat_raw or None)

    return Settings(
        api_key=api_key,
        api_secret=api_secret,
        totp=totp,
        totp_secret=totp_secret,
        symbol=sym,
        trade_instrument=trade_instrument,
        option_ce_symbol=option_ce,
        option_pe_symbol=option_pe,
        option_auto_strike=option_auto_strike,
        option_chain_expiry=option_chain_expiry,
        option_chain_underlying=option_chain_underlying,
        option_strike_step=max(1, option_strike_step),
        option_otm_window_steps=max(1, option_otm_window_steps),
        fno_product=product,
        capital=_env_float("ORB_CAPITAL", 100_000.0),
        risk_per_trade=_env_float("ORB_RISK_PCT", 0.01),
        rr=rr,
        debug=os.getenv("ORB_DEBUG", "1").strip() not in {"0", "false", "False"},
        orb=orb,
        alerts_only=_env_bool_off("ORB_ALERTS_ONLY", False),
        telegram_bot_token=tg_token,
        telegram_chat_id=tg_chat,
    )
