"""Single-user local SIS authentication, sessions, and CSRF protection."""

from __future__ import annotations

import base64
import argparse
import getpass
import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Iterable
from urllib.parse import parse_qs, urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response


SESSION_COOKIE = "sis_session"
LOGIN_CSRF_COOKIE = "sis_login_csrf"
SESSION_TTL_SECONDS = 8 * 60 * 60
LOGIN_CSRF_TTL_SECONDS = 10 * 60
MAX_CSRF_BODY_BYTES = 8 * 1024 * 1024
_PBKDF2_ITERATIONS = 310_000


def _csv_env(name: str, default: Iterable[str]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return tuple(default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _normalize_host(host: str) -> str:
    return host.strip().lower().strip("[]")


def _normalize_origin(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return None
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    except ValueError:
        return None


@dataclass(frozen=True)
class AuthSettings:
    username: str
    password_hash: str
    session_secret: bytes
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    session_ttl_seconds: int = SESSION_TTL_SECONDS
    secure_cookies: bool = False

    @property
    def ready(self) -> bool:
        return bool(
            self.username
            and self.password_hash
            and len(self.session_secret) >= 32
            and self.allowed_hosts
            and self.allowed_origins
            and _valid_password_hash(self.password_hash)
        )


def load_auth_settings() -> AuthSettings:
    port = os.environ.get("SIS_PORT", "8000")
    defaults_hosts = ("localhost", "127.0.0.1", "::1")
    hosts = tuple(_normalize_host(item) for item in _csv_env("SIS_ALLOWED_HOSTS", defaults_hosts))
    default_origins = (
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"http://[::1]:{port}",
    )
    origins = tuple(
        origin
        for item in _csv_env("SIS_ALLOWED_ORIGINS", default_origins)
        if (origin := _normalize_origin(item)) is not None
    )
    secret = os.environ.get("SIS_SESSION_SECRET", "").encode("utf-8")
    return AuthSettings(
        username=os.environ.get("SIS_AUTH_USERNAME", ""),
        password_hash=os.environ.get("SIS_AUTH_PASSWORD_HASH", ""),
        session_secret=secret,
        allowed_hosts=hosts,
        allowed_origins=origins,
        secure_cookies=os.environ.get("SIS_COOKIE_SECURE", "").lower()
        in {"1", "true", "yes"},
    )


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Create a PBKDF2-SHA256 password verifier for SIS_AUTH_PASSWORD_HASH."""
    if len(password) < 12:
        raise ValueError("Local SIS passwords must contain at least 12 characters")
    if len(password) > 1024:
        raise ValueError("Local SIS passwords cannot exceed 1024 characters")
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode().rstrip("="),
        base64.urlsafe_b64encode(derived).decode().rstrip("="),
    )


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _valid_password_hash(value: str) -> bool:
    try:
        algorithm, iteration_text, salt, digest = value.split("$", 3)
        iterations = int(iteration_text)
        return (
            algorithm == "pbkdf2_sha256"
            and 100_000 <= iterations <= 2_000_000
            and len(_b64decode(salt)) >= 16
            and len(_b64decode(digest)) == 32
        )
    except (ValueError, TypeError):
        return False


def verify_password(password: str, encoded: str) -> bool:
    if not _valid_password_hash(encoded):
        return False
    algorithm, iteration_text, salt, expected = encoded.split("$", 3)
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), _b64decode(salt), int(iteration_text)
    )
    return hmac.compare_digest(actual, _b64decode(expected))


@dataclass
class _Session:
    username: str
    csrf_token: str
    created_at: float
    expires_at: float


class SessionStore:
    """Process-local sessions; restart invalidates sessions rather than replaying them."""

    def __init__(self, settings: AuthSettings):
        self.settings = settings
        self._sessions: dict[str, _Session] = {}
        self._login_attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _session_key(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(self, username: str) -> tuple[str, _Session]:
        now = time.time()
        token = secrets.token_urlsafe(48)
        session = _Session(
            username=username,
            csrf_token=secrets.token_urlsafe(32),
            created_at=now,
            expires_at=now + self.settings.session_ttl_seconds,
        )
        with self._lock:
            self._sessions[self._session_key(token)] = session
        return token, session

    def get(self, token: str | None) -> _Session | None:
        if not token:
            return None
        key = self._session_key(token)
        now = time.time()
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if session.expires_at <= now:
                self._sessions.pop(key, None)
                return None
            return session

    def revoke(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._sessions.pop(self._session_key(token), None)

    def make_login_csrf(self) -> str:
        nonce = secrets.token_urlsafe(32)
        expires = str(int(time.time()) + LOGIN_CSRF_TTL_SECONDS)
        payload = f"{nonce}.{expires}"
        signature = hmac.new(
            self.settings.session_secret, f"login:{payload}".encode(), hashlib.sha256
        ).hexdigest()
        return f"{payload}.{signature}"

    def verify_login_csrf(self, cookie: str | None, submitted: str | None) -> bool:
        if not cookie or not submitted or not hmac.compare_digest(cookie, submitted):
            return False
        try:
            nonce, expires, signature = cookie.split(".", 2)
            if int(expires) <= time.time():
                return False
        except (ValueError, TypeError):
            return False
        payload = f"{nonce}.{expires}"
        expected = hmac.new(
            self.settings.session_secret, f"login:{payload}".encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def authenticate(self, username: str, password: str) -> bool:
        username_matches = hmac.compare_digest(username, self.settings.username)
        password_matches = verify_password(password, self.settings.password_hash)
        return self.settings.ready and username_matches and password_matches

    def attempt_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            attempts = [stamp for stamp in self._login_attempts.get(client_ip, []) if now - stamp < 300]
            self._login_attempts[client_ip] = attempts
            return len(attempts) < 5

    def record_failed_attempt(self, client_ip: str) -> None:
        with self._lock:
            self._login_attempts.setdefault(client_ip, []).append(time.time())

    def clear_failed_attempts(self, client_ip: str) -> None:
        with self._lock:
            self._login_attempts.pop(client_ip, None)


def _cookie_header(scope) -> dict[str, str]:
    headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
    cookie = SimpleCookie()
    try:
        cookie.load(headers.get("cookie", ""))
    except Exception:
        return {}
    return {key: morsel.value for key, morsel in cookie.items()}


def _origin_from_headers(headers) -> str | None:
    origin = headers.get("origin")
    if origin:
        return _normalize_origin(origin)
    referer = headers.get("referer")
    if referer:
        parsed = urlsplit(referer)
        if parsed.scheme and parsed.netloc:
            return _normalize_origin(f"{parsed.scheme}://{parsed.netloc}")
    return None


def _same_origin(request: Request, origin: str | None) -> bool:
    if origin is None:
        return False
    expected = _normalize_origin(f"{request.url.scheme}://{request.headers.get('host', '')}")
    return bool(expected and hmac.compare_digest(origin, expected))


def _submitted_csrf(body: bytes, content_type: str) -> str | None:
    if "application/x-www-form-urlencoded" in content_type:
        values = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
        return values.get("_csrf", [None])[0]
    return None


class SISAuthMiddleware:
    """Protect every SIS HTTP route and reject cross-origin websocket upgrades."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await self._websocket(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        settings: AuthSettings = scope.get("app").state.auth_settings
        sessions: SessionStore = scope.get("app").state.auth_sessions
        try:
            host = request.url.hostname
        except ValueError:
            host = None
        path = request.url.path
        public = path in {"/login", "/healthz"} or path.startswith("/static/")

        async def immediate(response):
            response.headers.setdefault("Cache-Control", "no-store")
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
            response.headers.setdefault("Referrer-Policy", "same-origin")
            await response(scope, receive, send)

        if not host or _normalize_host(host) not in settings.allowed_hosts:
            await immediate(JSONResponse({"detail": "Untrusted Host"}, status_code=400))
            return
        if _normalize_host(host) not in {"localhost", "127.0.0.1", "::1"} and request.url.scheme != "https":
            await immediate(JSONResponse({"detail": "Remote SIS access requires HTTPS"}, status_code=400))
            return

        if not settings.ready and not public:
            await immediate(JSONResponse({"detail": "SIS authentication is not configured"}, status_code=503))
            return

        if not public:
            token = _cookie_header(scope).get(SESSION_COOKIE)
            session = sessions.get(token)
            if session is None:
                if request.method in {"GET", "HEAD"} and "text/html" in request.headers.get("accept", ""):
                    await immediate(RedirectResponse("/login", status_code=303))
                else:
                    await immediate(JSONResponse({"detail": "Authentication required"}, status_code=401))
                return
            scope.setdefault("state", {})["username"] = session.username
            scope["state"]["csrf_token"] = session.csrf_token
            scope["state"]["session"] = session

        body = None
        replay_receive = receive
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = _origin_from_headers(request.headers)
            if (
                origin is None
                or origin not in settings.allowed_origins
                or not _same_origin(request, origin)
            ):
                await immediate(JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403))
                return

            body = await request.body()
            if len(body) > MAX_CSRF_BODY_BYTES:
                await immediate(JSONResponse({"detail": "Request body too large"}, status_code=413))
                return
            content_type = request.headers.get("content-type", "").lower()
            submitted = request.headers.get("x-csrf-token") or _submitted_csrf(body, content_type)
            cookies = _cookie_header(scope)
            if path == "/login":
                csrf_ok = sessions.verify_login_csrf(cookies.get(LOGIN_CSRF_COOKIE), submitted)
            else:
                session = scope.get("state", {}).get("session")
                csrf_ok = bool(
                    session
                    and submitted
                    and hmac.compare_digest(session.csrf_token, submitted)
                )
            if not csrf_ok:
                await immediate(JSONResponse({"detail": "CSRF validation failed"}, status_code=403))
                return

            delivered = False

            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body or b"", "more_body": False}
                return await receive()

            replay_receive = replay_receive

        async def secured_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {key.lower() for key, _value in headers}
                defaults = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"SAMEORIGIN"),
                    (b"referrer-policy", b"same-origin"),
                ]
                if not path.startswith("/static/"):
                    defaults.append((b"cache-control", b"no-store"))
                if request.url.scheme == "https":
                    defaults.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                headers.extend((key, value) for key, value in defaults if key not in present)
                message["headers"] = headers
            await send(message)

        await self.app(scope, replay_receive, secured_send)

    async def _websocket(self, scope, receive, send):
        app = scope.get("app")
        settings: AuthSettings = app.state.auth_settings
        sessions: SessionStore = app.state.auth_sessions
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        cookies = _cookie_header(scope)
        host_header = headers.get("host", "")
        try:
            host = urlsplit(f"//{host_header}").hostname or ""
        except ValueError:
            host = ""
        host = _normalize_host(host)
        origin = _origin_from_headers(headers)
        ws_scheme = "https" if scope.get("scheme") == "wss" else "http"
        expected_origin = _normalize_origin(f"{ws_scheme}://{host_header}")
        session = sessions.get(cookies.get(SESSION_COOKIE))
        if (
            not settings.ready
            or host not in settings.allowed_hosts
            or (host not in {"localhost", "127.0.0.1", "::1"} and scope.get("scheme") != "wss")
            or session is None
            or origin not in settings.allowed_origins
            or not expected_origin
            or not hmac.compare_digest(origin or "", expected_origin)
        ):
            await send({"type": "websocket.close", "code": 4401})
            return
        scope.setdefault("state", {})["username"] = session.username
        await self.app(scope, receive, send)


def secure_cookie(request: Request) -> bool:
    return request.url.scheme == "https" or request.app.state.auth_settings.secure_cookies


def safe_local_path(value: str | None, default: str = "/") -> str:
    if not value:
        return default
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return default
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Local SIS authentication utilities")
    parser.add_argument(
        "--hash-password",
        action="store_true",
        help="Prompt for a password and print a PBKDF2 verifier for SIS_AUTH_PASSWORD_HASH",
    )
    args = parser.parse_args()
    if not args.hash_password:
        parser.print_help()
        return 0
    password = getpass.getpass("SIS local password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if not password or not hmac.compare_digest(password, confirmation):
        raise SystemExit("Passwords were empty or did not match")
    print(hash_password(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
