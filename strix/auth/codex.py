"""ChatGPT (Codex) subscription OAuth: login, token refresh, and the OpenAI
client that routes inference through the ChatGPT backend.

This lets a user run Strix on their ChatGPT Plus/Pro subscription instead of a
metered OpenAI API key. The mechanism mirrors OpenAI's own Codex CLI: an OAuth
2.0 authorization-code flow with PKCE against ``auth.openai.com``, whose access
token is then sent as a ``Bearer`` token to ``chatgpt.com/backend-api/codex``
(the subscription endpoint) rather than to the metered ``api.openai.com``.

Terms of service: using a ChatGPT subscription outside OpenAI's own products is
not officially supported by OpenAI and may be against its terms. Strix identifies
as the Codex client (``originator: codex_cli_rs``) because the backend requires
it. This path exists as an option; the user chooses it knowingly.

The OAuth constants (client id, endpoints, headers, model names) are copied from
OpenAI's Codex CLI and must be kept in step with it if OpenAI changes them.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import AsyncOpenAI


from strix.auth import store


PROVIDER = "codex"

# --- OAuth client (from openai/codex) --------------------------------------
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105 - URL, not a secret
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPE = "openid profile email offline_access"

# --- inference (ChatGPT subscription backend) ------------------------------
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
ORIGINATOR = "codex_cli_rs"
_ACCOUNT_CLAIM = "https://api.openai.com/auth"

# Setting STRIX_LLM to this value runs inference on the user's authenticated
# ChatGPT subscription instead of a metered API key. It's the single switch —
# there is no separate auth-mode flag.
SUBSCRIPTION_MODEL = "openai/subscription"

# The actual model sent to the ChatGPT backend for a subscription run. gpt-5.4 is
# used because gpt-5.5 and newer apply stricter content moderation that
# interferes with security-testing prompts. (The subscription only exposes the
# general GPT-5.x models, not the ``*-codex`` API variants.)
DEFAULT_CODEX_MODEL = "gpt-5.4"

_TOKEN_TIMEOUT = 30
# Refresh a little before the token actually expires so a request never goes out
# with a token that lapses in flight.
_EXPIRY_SKEW_S = 300

_refresh_lock = threading.Lock()


@contextlib.contextmanager
def _refresh_guard() -> Iterator[None]:
    """Serialize a token refresh within and across Strix processes.

    The in-process lock covers concurrent agents in one process; a best-effort
    file lock (``flock``) covers concurrent Strix processes sharing one login, so
    two of them can't both spend the single-use refresh token and leave one with
    ``invalid_grant``. Degrades to the in-process lock alone where ``flock`` is
    unavailable (e.g. Windows).
    """
    with _refresh_lock:
        try:
            import fcntl

            lock_path = store.AUTH_PATH.with_suffix(".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("w")
        except (ImportError, OSError):
            yield
            return
        try:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


class CodexAuthError(Exception):
    """A Codex auth step failed. ``code`` is a stable, machine-readable reason."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


# --- PKCE + authorization ---------------------------------------------------


def _b64url(raw: bytes) -> str:
    """Unpadded base64url (RFC 7636). Padding here breaks the challenge check."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` for a fresh PKCE exchange."""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def create_state() -> str:
    """Return a random ``state`` value for CSRF protection on the callback."""
    return secrets.token_hex(16)


def build_authorize_url(challenge: str, state: str) -> str:
    """Build the browser URL the user visits to approve the sign-in."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def parse_redirect_input(value: str) -> tuple[str | None, str | None]:
    """Extract ``(code, state)`` from whatever the user pastes back.

    Accepts a full redirect URL, a ``code#state`` fragment, a bare query string,
    or just the code — so the manual-paste fallback is forgiving.
    """
    value = (value or "").strip()
    if not value:
        return None, None
    with contextlib.suppress(ValueError):
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme and parsed.query:
            query = urllib.parse.parse_qs(parsed.query)
            return _first(query, "code"), _first(query, "state")
    if "#" in value:
        code, _, state = value.partition("#")
        return code or None, state or None
    if "code=" in value:
        query = urllib.parse.parse_qs(value)
        return _first(query, "code"), _first(query, "state")
    return value, None


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


# --- token exchange / refresh ----------------------------------------------


def _post_form(payload: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode("ascii")
    request = urllib.request.Request(  # noqa: S310 - fixed https OAuth endpoint
        TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TOKEN_TIMEOUT) as response:  # noqa: S310
            data = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise CodexAuthError("token_http_error", f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CodexAuthError("unavailable", str(exc)) from exc
    if not isinstance(data, dict):
        raise CodexAuthError("bad_response", "token endpoint returned non-object")
    return data


def _record_from_token_response(
    data: dict[str, Any], refresh_fallback: str | None = None
) -> dict[str, Any]:
    access = data.get("access_token")
    # A refresh response may omit refresh_token when it isn't rotated; keep the old one.
    refresh = data.get("refresh_token") or refresh_fallback
    expires_in = data.get("expires_in")
    if not isinstance(access, str) or not access:
        raise CodexAuthError("bad_response", "token response missing access_token")
    if not isinstance(refresh, str) or not refresh:
        raise CodexAuthError("bad_response", "token response missing refresh_token")
    account_id = _account_id_from_jwt(access) or _account_id_from_jwt(
        data.get("id_token") if isinstance(data.get("id_token"), str) else ""
    )
    if not account_id:
        raise CodexAuthError("no_account_id", "could not read chatgpt_account_id from token")
    ttl = expires_in if isinstance(expires_in, (int, float)) else 3600
    return {
        "type": "oauth",
        "provider": PROVIDER,
        "access": access,
        "refresh": refresh,
        "account_id": account_id,
        "expires_at": time.time() + ttl,
    }


def exchange_code(code: str, verifier: str) -> dict[str, Any]:
    """Exchange an authorization code + PKCE verifier for a token record."""
    data = _post_form(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        }
    )
    return _record_from_token_response(data)


def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new token record."""
    data = _post_form(
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        }
    )
    return _record_from_token_response(data, refresh_fallback=refresh_token)


def _account_id_from_jwt(token: str | None) -> str | None:
    """Read the ChatGPT account id out of a JWT payload without verifying it.

    The token's authenticity is enforced by the server on use; here we only need
    to read the account id claim so we can send the ``chatgpt-account-id`` header.
    """
    if not token or token.count(".") != 2:
        return None
    payload_b64 = token.split(".")[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    auth = payload.get(_ACCOUNT_CLAIM)
    if isinstance(auth, dict):
        account_id = auth.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    organizations = payload.get("organizations")
    if isinstance(organizations, list) and organizations and isinstance(organizations[0], dict):
        org_id = organizations[0].get("id")
        if isinstance(org_id, str) and org_id:
            return org_id
    return None


# --- stored credential access ----------------------------------------------


def read_record() -> dict[str, Any] | None:
    """Return the stored Codex token record if it is complete, else None."""
    record = store.read_provider(PROVIDER)
    if not record or record.get("type") != "oauth":
        return None
    if not (record.get("access") and record.get("refresh") and record.get("account_id")):
        return None
    return record


def is_authenticated() -> bool:
    """True when a usable Codex token record exists on disk."""
    return read_record() is not None


def save_record(record: dict[str, Any]) -> None:
    store.write_provider(PROVIDER, record)


def logout() -> None:
    store.forget(PROVIDER)


def _near_expiry(record: dict[str, Any]) -> bool:
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return True
    return expires_at - _EXPIRY_SKEW_S <= time.time()


def get_valid_token() -> tuple[str, str]:
    """Return ``(access_token, account_id)``, refreshing if near expiry.

    Refreshes under a within- and cross-process lock and re-reads the store after
    acquiring it, so that when many agents (or parallel Strix processes sharing
    one login) fire at once only one refresh happens — OpenAI invalidates a
    refresh token as soon as it is used, so a concurrent stampede would fail. The
    re-read means a caller that loses the race picks up the token the winner just
    rotated instead of exchanging the now-dead one.
    """
    record = read_record()
    if record is None:
        raise CodexAuthError("not_authenticated", "not signed in; run: strix auth login")
    if not _near_expiry(record):
        return record["access"], record["account_id"]
    with _refresh_guard():
        record = read_record()
        if record is None:
            raise CodexAuthError("not_authenticated", "not signed in; run: strix auth login")
        if not _near_expiry(record):
            return record["access"], record["account_id"]
        refreshed = refresh_tokens(record["refresh"])
        save_record(refreshed)
        return refreshed["access"], refreshed["account_id"]


# --- inference client -------------------------------------------------------


def build_openai_client() -> AsyncOpenAI:
    """Build an ``AsyncOpenAI`` that routes inference through the ChatGPT backend.

    A per-request hook refreshes the token if needed and stamps the auth headers,
    so a long scan that outlives a single access token keeps working. The
    ``api_key`` passed to the SDK is a placeholder — the hook always overwrites
    the ``Authorization`` header with the current token.
    """
    import asyncio

    import httpx
    from openai import AsyncOpenAI

    # Validate up front so sign-in problems surface at configure time, not
    # mid-scan, and to fail fast if the stored refresh token is dead.
    get_valid_token()

    async def _auth_hook(request: httpx.Request) -> None:
        # Refresh-aware auth stamped per request so a scan that outlives one
        # access token keeps working. The account id can only change with the
        # token, so it is set here alongside the bearer rather than as a static
        # default header.
        access, account_id = await asyncio.to_thread(get_valid_token)
        request.headers["Authorization"] = f"Bearer {access}"
        request.headers["chatgpt-account-id"] = account_id

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        event_hooks={"request": [_auth_hook]},
    )
    return AsyncOpenAI(
        api_key="strix-codex-oauth",
        base_url=CODEX_BASE_URL,
        http_client=http_client,
        default_headers={
            "OpenAI-Beta": "responses=experimental",
            "originator": ORIGINATOR,
        },
    )


_subscription_client: AsyncOpenAI | None = None


def get_subscription_client() -> AsyncOpenAI:
    """Return the process-wide subscription client, building it once.

    One client is reused across every agent in a run: its per-request auth hook
    refreshes the token itself, so there's no reason to rebuild it, and building
    one per agent would leak httpx clients.
    """
    global _subscription_client  # noqa: PLW0603
    if _subscription_client is None:
        _subscription_client = build_openai_client()
    return _subscription_client


def is_subscription(model_name: str | None) -> bool:
    """Whether ``STRIX_LLM`` selects the authenticated ChatGPT subscription."""
    return (model_name or "").strip().lower() == SUBSCRIPTION_MODEL


def resolve_subscription_model() -> str:
    """The concrete model sent to the ChatGPT backend for a subscription run."""
    return DEFAULT_CODEX_MODEL


def auth_mode_label(model_name: str | None) -> str:
    """How a run authenticated, for recording in run.json / telemetry / the viewer."""
    return "subscription" if is_subscription(model_name) else "api_key"


__all__ = [
    "DEFAULT_CODEX_MODEL",
    "PROVIDER",
    "SUBSCRIPTION_MODEL",
    "CodexAuthError",
    "auth_mode_label",
    "build_authorize_url",
    "build_openai_client",
    "create_state",
    "exchange_code",
    "generate_pkce",
    "get_subscription_client",
    "get_valid_token",
    "is_authenticated",
    "is_subscription",
    "logout",
    "parse_redirect_input",
    "read_record",
    "refresh_tokens",
    "resolve_subscription_model",
    "save_record",
]
