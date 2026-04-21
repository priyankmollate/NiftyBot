from __future__ import annotations

import pyotp
from growwapi import GrowwAPI
from growwapi.groww.exceptions import GrowwAPIException

from niftybot.config import Settings

_AUTH_HINT = (
    "\n\nIf you use Groww's *TOTP* keys: GROWW_API_KEY must be the **TOTP API token** "
    "from the portal (Generate TOTP token), not the regular 'Generate API key' string. "
    "Use GROWW_TOTP_SECRET = authenticator seed (pyotp) and/or a fresh GROWW_TOTP. "
    "If you use *API key + daily secret*: set GROWW_API_SECRET instead of TOTP fields. "
    "See: https://groww.in/trade-api/docs/python-sdk (Authentication)."
)


def build_groww_client(settings: Settings) -> GrowwAPI:
    try:
        if settings.api_secret:
            access_token = GrowwAPI.get_access_token(
                api_key=settings.api_key,
                secret=settings.api_secret,
            )
        else:
            if settings.totp_secret:
                totp = pyotp.TOTP(settings.totp_secret.strip()).now()
            elif settings.totp:
                totp = settings.totp.strip()
            else:
                raise ValueError("TOTP flow needs GROWW_TOTP and/or GROWW_TOTP_SECRET.")

            access_token = GrowwAPI.get_access_token(
                api_key=settings.api_key,
                totp=totp,
            )
        return GrowwAPI(access_token)
    except GrowwAPIException as exc:
        raise GrowwAPIException(
            msg=exc.msg + _AUTH_HINT,
            code=exc.code,
        ) from exc
