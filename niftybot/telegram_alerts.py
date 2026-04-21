"""Telegram Bot API alerts (stdlib only; no extra pip deps)."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

# https://core.telegram.org/bots/api#sendmessage
_MAX_MESSAGE_LEN = 4096


def normalize_bot_token(token: str | None) -> str | None:
    """Strip whitespace and optional surrounding quotes from .env mistakes."""
    t = (token or "").strip()
    # Allow inline comments in .env, e.g. TOKEN=123:abc # from BotFather
    t = re.sub(r"\s+#.*$", "", t).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        t = t[1:-1].strip()
    return t or None


def normalize_chat_id(chat: str | None) -> str | None:
    """Same trimming as token; chat_id is often numeric or -100… for groups."""
    return normalize_bot_token(chat)


# Copy/paste from PDF, Slack, etc. often injects these; they break the API path.
_INVISIBLE = (
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\u00a0",
    "\u2028",
    "\u2029",
)


def sanitize_telegram_bot_token_for_api(token: str | None) -> str | None:
    """
    Normalize .env value for `https://api.telegram.org/bot<token>/…`.

    Removes invisible characters, internal spaces, and a mistaken leading `bot`
    before the numeric id (avoids `…/botbot123:SECRET/…` which always 401).
    """
    t = normalize_bot_token(token)
    if not t:
        return None
    for ch in _INVISIBLE:
        t = t.replace(ch, "")
    t = re.sub(r"[\s\u00a0]+", "", t)
    # URL is …/bot<token>/ — token from @BotFather is `123:SECRET`, not `bot123:SECRET`
    if len(t) > 4 and t[:3].lower() == "bot" and t[3].isdigit():
        t = t[3:]
    return t or None


def sanitize_chat_id_for_api(chat: str | None) -> str | None:
    """Trim quotes and remove invisible characters (keep minus sign for group ids)."""
    c = normalize_chat_id(chat)
    if not c:
        return None
    for ch in _INVISIBLE:
        c = c.replace(ch, "")
    c = c.strip()
    # Ignore accidental trailing inline comments and spaces in .env.
    c = re.sub(r"\s+#.*$", "", c).strip()
    return c or None


def _friendly_urllib_error(e: urllib.error.URLError) -> str | None:
    """Human text for DNS / connectivity failures (otherwise None)."""
    r = e.reason
    es = str(e).lower()
    if isinstance(r, socket.gaierror) or "nodename nor servname" in es or "name or service not known" in es:
        return (
            "Network/DNS: could not resolve api.telegram.org (hostname unknown). "
            "Check internet, VPN, corporate firewall, or DNS (try 1.1.1.1 / 8.8.8.8). "
            "Confirm with: ping api.telegram.org or curl -I https://api.telegram.org"
        )
    if isinstance(r, (socket.timeout, TimeoutError)):
        return "Network: request to Telegram timed out. Check connection or try again."
    if isinstance(r, (ConnectionRefusedError, ConnectionResetError)):
        return "Network: connection to Telegram was refused or reset — firewall or proxy may block HTTPS."
    if isinstance(r, OSError) and r.errno is not None:
        return f"Network error ({type(r).__name__}): {r}"
    return None


def verify_telegram_bot_token(bot_token: str | None) -> tuple[bool, str]:
    """
    GET getMe — validates token without sending a chat message.
    Returns (ok, message_for_user).
    """
    t = sanitize_telegram_bot_token_for_api(bot_token)
    if not t:
        return False, "No TELEGRAM_BOT_TOKEN in environment (or empty after cleanup)."
    if ":" not in t or len(t) < 12:
        return False, "Token should look like `123456789:AAH…` (digits, colon, secret). Check .env."
    url = f"https://api.telegram.org/bot{t}/getMe"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        if data.get("ok"):
            r = data.get("result") or {}
            un = r.get("username", "?")
            i = r.get("id", "?")
            return True, f"Token OK — bot id {i}, @{un}"
        return False, _friendly_api_error(data, http_code=None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")[:800]
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return False, _friendly_api_error(data, http_code=e.code)
        except json.JSONDecodeError:
            pass
        if e.code == 401:
            return (
                False,
                "HTTP 401: Telegram rejected this token. Regenerate in @BotFather, update .env, "
                "restart Streamlit. If you use two env files, confirm this app loads the one you edited.",
            )
        return False, raw or f"HTTP {e.code}"
    except urllib.error.URLError as e:
        msg = _friendly_urllib_error(e)
        return False, msg if msg else repr(e)
    except Exception as e:  # noqa: BLE001
        return False, repr(e)


def _friendly_api_error(payload: dict, *, http_code: int | None) -> str:
    code = payload.get("error_code")
    desc = str(payload.get("description", "Unknown error"))
    if code == 401 or http_code == 401:
        return (
            "Unauthorized (401): TELEGRAM_BOT_TOKEN is invalid or revoked. "
            "Open @BotFather → /mybots → your bot → API Token, copy the full `digits:secret` string "
            "into .env with no spaces or quotes. If you rotated the token, update .env and restart."
        )
    if code == 400 and "chat not found" in desc.lower():
        return (
            "Bad request: chat not found — check TELEGRAM_CHAT_ID. "
            "Message your bot once, then open https://api.telegram.org/bot<TOKEN>/getUpdates"
        )
    if code == 403 or http_code == 403:
        d = desc.lower()
        if "blocked" in d:
            return (
                "Forbidden (403): you blocked this bot. In Telegram: Settings → Privacy → Blocked users → "
                "unblock the bot, then open the bot and tap Start."
            )
        if "kicked" in d or "not a member" in d:
            return (
                "Forbidden (403): the bot was removed from this group/supergroup or is not a member. "
                "Re-add the bot to the group (as member or admin) and try again."
            )
        if "channel" in d or "write" in d or "rights" in d:
            return (
                "Forbidden (403): bot cannot post here. For a channel, add the bot as an admin with "
                "“Post messages”. For a group, ensure the bot was not restricted and can send messages."
            )
        if "initiate" in d or "deactivated" in d:
            return (
                "Forbidden (403): open the bot in Telegram, send /start once, then retry. "
                "If the account is deleted/deactivated, use a different TELEGRAM_CHAT_ID."
            )
        return (
            "Forbidden (403): Telegram will not deliver to this TELEGRAM_CHAT_ID — "
            "wrong id, bot blocked, user never /start, bot kicked from group, or channel without admin rights. "
            "Confirm id via getUpdates after you message the bot or add it to the group."
        )
    return desc if desc else repr(payload)


def send_telegram_message(*, bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    """
    POST sendMessage. Returns (success, error_detail).
    On success error_detail is "".
    """
    bot_token = sanitize_telegram_bot_token_for_api(bot_token) or ""
    chat_id = sanitize_chat_id_for_api(chat_id) or ""
    if not bot_token or not chat_id:
        return True, ""
    body = text if len(text) <= _MAX_MESSAGE_LEN else text[: _MAX_MESSAGE_LEN - 24] + "\n…(truncated)"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": body,
            "disable_web_page_preview": "true",
        }
    ).encode()
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode()
        data = json.loads(raw)
        if not data.get("ok"):
            return False, _friendly_api_error(data, http_code=None)
        return True, ""
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")[:800]
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and not data.get("ok", True):
                return False, _friendly_api_error(data, http_code=e.code)
        except json.JSONDecodeError:
            pass
        if e.code == 401:
            return (
                False,
                "HTTP 401 Unauthorized: TELEGRAM_BOT_TOKEN rejected by Telegram. "
                "Regenerate token in @BotFather and update .env.",
            )
        return False, raw or f"HTTP {e.code}"
    except urllib.error.URLError as e:
        msg = _friendly_urllib_error(e)
        return False, msg if msg else repr(e)
    except Exception as e:  # noqa: BLE001
        return False, repr(e)
