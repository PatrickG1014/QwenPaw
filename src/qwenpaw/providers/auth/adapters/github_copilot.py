# -*- coding: utf-8 -*-
"""GitHub Copilot provider auth adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import httpx
from pydantic import BaseModel

from qwenpaw.__version__ import __version__ as QWENPAW_VERSION

from ..adapter import ProviderAuthAdapter
from ..credential_store import OAuthCredentialStore
from ..models import (
    AuthStartRequest,
    AuthStartResult,
    AuthStatusResult,
    OAuthCredential,
    ProviderAuthFlowType,
    ProviderAuthStatus,
    ProviderAuthType,
)

if TYPE_CHECKING:
    from ...provider import Provider

logger = logging.getLogger(__name__)

DEFAULT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
GRANT_TYPE_DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"
DEFAULT_COPILOT_API_BASE_URL = "https://api.githubcopilot.com"

DEFAULT_EDITOR_VERSION = "vscode/1.95.0"
DEFAULT_PLUGIN_VERSION = f"qwenpaw/{QWENPAW_VERSION}"
DEFAULT_USER_AGENT = f"QwenPaw/{QWENPAW_VERSION}"


class ProviderAuthRequiredError(Exception):
    """Raised when Copilot credentials are missing or revoked."""


class ProviderAuthTemporaryError(Exception):
    """Raised when a temporary remote/network auth error occurs."""


class DeviceSession(BaseModel):
    """In-memory GitHub device-code flow state."""

    flow_id: str
    device_code: str
    user_code: str
    verification_uri: str
    expires_at: int
    interval: int
    last_poll_at: int = 0


class CopilotApiToken(BaseModel):
    """Short-lived Copilot runtime token. Kept in memory only."""

    token: str
    expires_at: int = 0
    refresh_in: int = 0
    api_endpoint: str = DEFAULT_COPILOT_API_BASE_URL
    chat_enabled: bool | None = None
    sku: str = ""

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Return True if the token expires within ``buffer_seconds``."""
        if not self.expires_at:
            return False
        return self.expires_at <= int(time.time()) + max(0, buffer_seconds)


class CopilotBearerAuth(httpx.Auth):
    """Inject a fresh Copilot runtime bearer token per request."""

    requires_request_body = False
    requires_response_body = False

    def __init__(
        self,
        adapter: "GitHubCopilotOAuthAdapter",
        provider_id: str,
    ) -> None:
        self.adapter = adapter
        self.provider_id = provider_id

    async def async_auth_flow(  # type: ignore[override]
        self,
        request: httpx.Request,
    ):
        token = await self.adapter.get_copilot_token(self.provider_id)
        request.headers["Authorization"] = f"Bearer {token.token}"
        response = yield request
        if response.status_code != 401:
            return
        token = await self.adapter.get_copilot_token(
            self.provider_id,
            force_refresh=True,
        )
        request.headers["Authorization"] = f"Bearer {token.token}"
        yield request


@dataclass
class _AdapterConfig:
    client_id: str
    editor_version: str
    plugin_version: str
    user_agent: str


class GitHubCopilotOAuthAdapter(ProviderAuthAdapter):
    """GitHub Device Code auth and Copilot runtime-token adapter."""

    provider_id = "github-copilot"
    auth_type = ProviderAuthType.OAUTH_DEVICE_CODE

    def __init__(
        self,
        credential_store: OAuthCredentialStore | None = None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        token_refresh_buffer: int = 300,
    ) -> None:
        self.credential_store = credential_store or OAuthCredentialStore()
        self._http_client_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=30.0)
        )
        self._token_refresh_buffer = token_refresh_buffer
        self._device_sessions: dict[str, DeviceSession] = {}
        self._runtime_tokens: dict[str, CopilotApiToken] = {}
        self._runtime_locks: dict[str, asyncio.Lock] = {}
        self._http_clients: dict[str, httpx.AsyncClient] = {}

    async def start(
        self,
        provider: "Provider",
        request: AuthStartRequest,  # pylint: disable=unused-argument
    ) -> AuthStartResult:
        """Start GitHub's OAuth device-code flow."""
        self._cleanup_expired_sessions()
        config = self._config_for(provider)
        async with self._http_client_factory() as client:
            response = await client.post(
                GITHUB_DEVICE_CODE_URL,
                headers=self._github_headers(config),
                data={
                    "client_id": config.client_id,
                    "scope": "read:user",
                },
            )
            response.raise_for_status()
            data = response.json() or {}

        device_code = str(data.get("device_code") or "")
        user_code = str(data.get("user_code") or "")
        verification_uri = str(
            data.get("verification_uri") or "https://github.com/login/device",
        )
        expires_in = int(data.get("expires_in") or 900)
        interval = int(data.get("interval") or 5)
        if not device_code or not user_code:
            raise ValueError(
                "GitHub device-code response was missing required fields",
            )

        now = int(time.time())
        flow_id = uuid.uuid4().hex
        self._device_sessions[flow_id] = DeviceSession(
            flow_id=flow_id,
            device_code=device_code,
            user_code=user_code,
            verification_uri=verification_uri,
            expires_at=now + expires_in,
            interval=interval,
        )
        return AuthStartResult(
            flow_id=flow_id,
            flow_type=ProviderAuthFlowType.DEVICE_CODE,
            user_code=user_code,
            verification_uri=verification_uri,
            expires_at=now + expires_in,
            interval=interval,
            message="Open the verification URL and enter the user code.",
        )

    async def poll(
        self,
        provider: "Provider",
        flow_id: str,
    ) -> AuthStatusResult:
        """Poll GitHub once for a device-code flow."""
        session = self._device_sessions.get(flow_id)
        if session is None:
            return AuthStatusResult(
                status=ProviderAuthStatus.ERROR,
                message="Authentication flow was not found or has expired.",
            )
        now = int(time.time())
        if session.expires_at <= now:
            self._device_sessions.pop(flow_id, None)
            return AuthStatusResult(
                status=ProviderAuthStatus.EXPIRED,
                message="GitHub device code expired before authorization.",
            )
        if session.last_poll_at and (
            now - session.last_poll_at < max(1, session.interval)
        ):
            return AuthStatusResult(
                status=ProviderAuthStatus.PENDING,
                message="Waiting for GitHub authorization.",
            )

        session.last_poll_at = now
        config = self._config_for(provider)
        async with self._http_client_factory() as client:
            response = await client.post(
                GITHUB_TOKEN_URL,
                headers=self._github_headers(config),
                data={
                    "client_id": config.client_id,
                    "device_code": session.device_code,
                    "grant_type": GRANT_TYPE_DEVICE_CODE,
                },
            )
            data = response.json() if response.content else {}

        if "error" in data:
            return self._handle_poll_error(flow_id, session, data)

        access_token = str(data.get("access_token") or "")
        if not access_token:
            return AuthStatusResult(
                status=ProviderAuthStatus.PENDING,
                message="Waiting for GitHub authorization.",
            )

        account_label = await self._fetch_github_login(access_token, config)
        now = int(time.time())
        credential = OAuthCredential(
            provider_id=provider.id,
            token_type=str(data.get("token_type") or "token"),
            access_token=access_token,
            account_label=account_label,
            expires_at=None,
            scopes=["read:user"],
            metadata={"github_login": account_label} if account_label else {},
            created_at=now,
            updated_at=now,
        )
        self.credential_store.save(credential)
        self._device_sessions.pop(flow_id, None)
        return AuthStatusResult(
            status=ProviderAuthStatus.AUTHENTICATED,
            account_label=credential.account_label,
            expires_at=credential.expires_at,
            scopes=credential.scopes,
        )

    async def logout(
        self,
        provider: "Provider",
        credential: OAuthCredential | None,  # pylint: disable=unused-argument
    ) -> None:
        """Clear local Copilot auth state."""
        self.credential_store.delete(provider.id)
        self._runtime_tokens.pop(provider.id, None)
        self._runtime_locks.pop(provider.id, None)
        self._device_sessions.clear()
        client = self._http_clients.pop(provider.id, None)
        if client is not None and not client.is_closed:
            await client.aclose()

    async def get_status(
        self,
        provider: "Provider",  # pylint: disable=unused-argument
        credential: OAuthCredential | None,
    ) -> AuthStatusResult:
        """Return credential-store backed GitHub Copilot auth status."""
        if credential and credential.access_token:
            return AuthStatusResult(
                status=ProviderAuthStatus.AUTHENTICATED,
                account_label=credential.account_label,
                expires_at=credential.expires_at,
                scopes=credential.scopes,
            )
        return AuthStatusResult(
            status=ProviderAuthStatus.NOT_CONFIGURED,
            message="GitHub Copilot is not authenticated.",
        )

    def has_valid_credential(self, provider_id: str) -> bool:
        """Return whether a GitHub OAuth credential exists locally."""
        credential = self.credential_store.load(provider_id)
        return bool(credential and credential.access_token)

    async def get_copilot_token(
        self,
        provider_id: str,
        *,
        force_refresh: bool = False,
    ) -> CopilotApiToken:
        """Return a cached or freshly exchanged Copilot runtime token."""
        cached = self._runtime_tokens.get(provider_id)
        if (
            not force_refresh
            and cached is not None
            and not cached.is_expired(self._token_refresh_buffer)
        ):
            return cached

        lock = self._runtime_locks.setdefault(provider_id, asyncio.Lock())
        async with lock:
            cached = self._runtime_tokens.get(provider_id)
            if (
                not force_refresh
                and cached is not None
                and not cached.is_expired(self._token_refresh_buffer)
            ):
                return cached
            credential = self.credential_store.load(provider_id)
            if not credential or not credential.access_token:
                raise ProviderAuthRequiredError(
                    "GitHub Copilot is not authenticated.",
                )
            token = await self._exchange_copilot_token(credential)
            self._runtime_tokens[provider_id] = token
            return token

    def get_cached_copilot_endpoint(self, provider_id: str) -> str | None:
        """Return the cached Copilot API endpoint, if a token is loaded."""
        token = self._runtime_tokens.get(provider_id)
        return token.api_endpoint if token else None

    def get_or_create_http_client(self, provider_id: str) -> httpx.AsyncClient:
        """Return a shared httpx client with Copilot auth headers."""
        client = self._http_clients.get(provider_id)
        if client is not None and not client.is_closed:
            return client
        config = _AdapterConfig(
            client_id=self._client_id(),
            editor_version=DEFAULT_EDITOR_VERSION,
            plugin_version=DEFAULT_PLUGIN_VERSION,
            user_agent=DEFAULT_USER_AGENT,
        )
        client = httpx.AsyncClient(
            auth=CopilotBearerAuth(self, provider_id),
            headers=self.chat_headers(config),
            timeout=httpx.Timeout(60.0, read=300.0),
        )
        self._http_clients[provider_id] = client
        return client

    def chat_headers(
        self,
        config: _AdapterConfig | None = None,
    ) -> dict[str, str]:
        """Return headers expected by Copilot chat/model endpoints."""
        cfg = config or _AdapterConfig(
            client_id=self._client_id(),
            editor_version=DEFAULT_EDITOR_VERSION,
            plugin_version=DEFAULT_PLUGIN_VERSION,
            user_agent=DEFAULT_USER_AGENT,
        )
        return {
            "Editor-Version": cfg.editor_version,
            "Editor-Plugin-Version": cfg.plugin_version,
            "Copilot-Integration-Id": "vscode-chat",
            "Openai-Intent": "conversation-panel",
            "X-Github-Api-Version": "2025-04-01",
            "User-Agent": cfg.user_agent,
        }

    def _handle_poll_error(
        self,
        flow_id: str,
        session: DeviceSession,
        data: dict,
    ) -> AuthStatusResult:
        error = str(data.get("error") or "")
        if error == "authorization_pending":
            return AuthStatusResult(
                status=ProviderAuthStatus.PENDING,
                message="Waiting for GitHub authorization.",
            )
        if error == "slow_down":
            session.interval += 5
            return AuthStatusResult(
                status=ProviderAuthStatus.PENDING,
                message="GitHub requested slower polling.",
            )
        self._device_sessions.pop(flow_id, None)
        status = (
            ProviderAuthStatus.EXPIRED
            if error == "expired_token"
            else ProviderAuthStatus.ERROR
        )
        return AuthStatusResult(
            status=status,
            message=f"GitHub OAuth error: {error or 'unknown_error'}",
        )

    async def _fetch_github_login(
        self,
        access_token: str,
        config: _AdapterConfig,
    ) -> str:
        try:
            async with self._http_client_factory() as client:
                response = await client.get(
                    GITHUB_USER_URL,
                    headers={
                        **self._github_headers(config),
                        "Authorization": f"token {access_token}",
                    },
                )
                if response.status_code != 200:
                    return ""
                data = response.json() or {}
                return str(data.get("login") or "")
        except httpx.HTTPError:
            logger.debug(
                "Failed to fetch GitHub login for Copilot credential",
                exc_info=True,
            )
            return ""

    async def _exchange_copilot_token(
        self,
        credential: OAuthCredential,
    ) -> CopilotApiToken:
        config = _AdapterConfig(
            client_id=self._client_id(),
            editor_version=DEFAULT_EDITOR_VERSION,
            plugin_version=DEFAULT_PLUGIN_VERSION,
            user_agent=DEFAULT_USER_AGENT,
        )
        try:
            async with self._http_client_factory() as client:
                response = await client.get(
                    COPILOT_TOKEN_URL,
                    headers={
                        **self._github_headers(config),
                        "Authorization": f"token {credential.access_token}",
                    },
                )
                if response.status_code == 401:
                    self.credential_store.delete(credential.provider_id)
                    self._runtime_tokens.pop(credential.provider_id, None)
                    client_obj = self._http_clients.pop(
                        credential.provider_id,
                        None,
                    )
                    if client_obj is not None and not client_obj.is_closed:
                        await client_obj.aclose()
                    raise ProviderAuthRequiredError(
                        "GitHub Copilot authorization expired. "
                        "Please sign in again.",
                    )
                response.raise_for_status()
                data = response.json() or {}
        except ProviderAuthRequiredError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderAuthTemporaryError(
                "Failed to reach GitHub Copilot. Please try again later.",
            ) from exc

        token = str(data.get("token") or "")
        if not token:
            raise ProviderAuthTemporaryError(
                "GitHub Copilot token response did not include a token.",
            )
        endpoints = data.get("endpoints") or {}
        api_endpoint = str(
            endpoints.get("api") or DEFAULT_COPILOT_API_BASE_URL,
        )
        chat_enabled = data.get("chat_enabled")
        return CopilotApiToken(
            token=token,
            expires_at=int(data.get("expires_at") or 0),
            refresh_in=int(data.get("refresh_in") or 0),
            api_endpoint=api_endpoint,
            chat_enabled=(
                bool(chat_enabled) if chat_enabled is not None else None
            ),
            sku=str(data.get("sku") or ""),
        )

    def _cleanup_expired_sessions(self) -> None:
        now = int(time.time())
        expired = [
            flow_id
            for flow_id, session in self._device_sessions.items()
            if session.expires_at <= now
        ]
        for flow_id in expired:
            self._device_sessions.pop(flow_id, None)

    def _config_for(self, provider: "Provider") -> _AdapterConfig:
        meta = provider.meta or {}
        return _AdapterConfig(
            client_id=self._client_id(meta),
            editor_version=self._meta_string(
                meta,
                "editor_version",
                DEFAULT_EDITOR_VERSION,
            ),
            plugin_version=self._meta_string(
                meta,
                "plugin_version",
                DEFAULT_PLUGIN_VERSION,
            ),
            user_agent=self._meta_string(
                meta,
                "user_agent",
                DEFAULT_USER_AGENT,
            ),
        )

    @staticmethod
    def _github_headers(config: _AdapterConfig) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": config.user_agent,
            "Editor-Version": config.editor_version,
            "Editor-Plugin-Version": config.plugin_version,
        }

    @staticmethod
    def _meta_string(
        meta: dict,
        key: str,
        default: str,
    ) -> str:
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
        return default

    def _client_id(self, meta: dict | None = None) -> str:
        env_value = os.environ.get("QWENPAW_GITHUB_COPILOT_CLIENT_ID")
        if env_value:
            return env_value
        if meta:
            value = meta.get("client_id")
            if isinstance(value, str) and value:
                return value
        return DEFAULT_CLIENT_ID
