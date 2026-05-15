# -*- coding: utf-8 -*-
# pylint: disable=protected-access,redefined-outer-name,unused-argument
"""Tests for GitHub Copilot provider auth adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import httpx
import pytest

from qwenpaw.providers.auth.adapters.github_copilot import (
    COPILOT_TOKEN_URL,
    GITHUB_DEVICE_CODE_URL,
    GITHUB_TOKEN_URL,
    GITHUB_USER_URL,
    CopilotApiToken,
    DeviceSession,
    GitHubCopilotOAuthAdapter,
    ProviderAuthRequiredError,
    ProviderAuthTemporaryError,
)
from qwenpaw.providers.auth.credential_store import OAuthCredentialStore
from qwenpaw.providers.auth.models import (
    AuthStartRequest,
    OAuthCredential,
    ProviderAuthStatus,
    ProviderAuthType,
)
from qwenpaw.providers.openai_provider import OpenAIProvider


@pytest.fixture(autouse=True)
def _isolate_master_key(tmp_path: Path, monkeypatch):
    import qwenpaw.security.secret_store as mod

    test_key = bytes.fromhex(
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    )
    monkeypatch.setattr(mod, "_cached_master_key", test_key)
    monkeypatch.setattr(mod, "_cached_fernet", None)
    monkeypatch.setattr(mod, "_get_secret_dir", lambda: tmp_path)


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(
        id="github-copilot",
        name="GitHub Copilot",
        auth_type=ProviderAuthType.OAUTH_DEVICE_CODE,
        require_api_key=False,
    )


def _adapter(
    tmp_path: Path,
    handlers: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response]],
    *,
    token_refresh_buffer: int = 300,
) -> GitHubCopilotOAuthAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, str(request.url).split("?", 1)[0])
        if key in handlers:
            return handlers[key](request)
        return httpx.Response(404, json={"error": "missing handler"})

    transport = httpx.MockTransport(handler)
    return GitHubCopilotOAuthAdapter(
        credential_store=OAuthCredentialStore(tmp_path / "providers"),
        http_client_factory=lambda: httpx.AsyncClient(transport=transport),
        token_refresh_buffer=token_refresh_buffer,
    )


async def test_start_device_flow_uses_form_data(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    captured: dict[str, httpx.Request] = {}

    def device_handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "device_code": "dev-abc",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5,
            },
        )

    adapter = _adapter(
        tmp_path,
        {("POST", GITHUB_DEVICE_CODE_URL): device_handler},
    )
    result = await adapter.start(provider, AuthStartRequest())

    assert result.flow_type == "device_code"
    assert result.user_code == "ABCD-EFGH"
    assert result.verification_uri == "https://github.com/login/device"
    assert result.interval == 5
    assert result.flow_id in adapter._device_sessions
    assert (
        captured["request"]
        .headers["content-type"]
        .startswith(
            "application/x-www-form-urlencoded",
        )
    )
    assert b"client_id=" in captured["request"].content
    assert b"scope=read" in captured["request"].content


async def test_poll_pending_does_not_save_credential(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    adapter = _adapter(
        tmp_path,
        {
            ("POST", GITHUB_DEVICE_CODE_URL): lambda req: httpx.Response(
                200,
                json={
                    "device_code": "dev",
                    "user_code": "CODE",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 5,
                },
            ),
            ("POST", GITHUB_TOKEN_URL): lambda req: httpx.Response(
                200,
                json={"error": "authorization_pending"},
            ),
        },
    )

    start = await adapter.start(provider, AuthStartRequest())
    status = await adapter.poll(provider, start.flow_id)

    assert status.status == ProviderAuthStatus.PENDING
    assert adapter.load_credential(provider.id) is None


async def test_poll_slow_down_increases_interval(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    adapter = _adapter(
        tmp_path,
        {
            ("POST", GITHUB_DEVICE_CODE_URL): lambda req: httpx.Response(
                200,
                json={
                    "device_code": "dev",
                    "user_code": "CODE",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 5,
                },
            ),
            ("POST", GITHUB_TOKEN_URL): lambda req: httpx.Response(
                200,
                json={"error": "slow_down"},
            ),
        },
    )

    start = await adapter.start(provider, AuthStartRequest())
    status = await adapter.poll(provider, start.flow_id)

    assert status.status == ProviderAuthStatus.PENDING
    assert adapter._device_sessions[start.flow_id].interval == 10


async def test_poll_authorized_saves_credential_and_clears_session(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    adapter = _adapter(
        tmp_path,
        {
            ("POST", GITHUB_DEVICE_CODE_URL): lambda req: httpx.Response(
                200,
                json={
                    "device_code": "dev",
                    "user_code": "CODE",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 5,
                },
            ),
            ("POST", GITHUB_TOKEN_URL): lambda req: httpx.Response(
                200,
                json={"access_token": "gho_test", "token_type": "token"},
            ),
            ("GET", GITHUB_USER_URL): lambda req: httpx.Response(
                200,
                json={"login": "octocat"},
            ),
        },
    )

    start = await adapter.start(provider, AuthStartRequest())
    status = await adapter.poll(provider, start.flow_id)

    assert status.status == ProviderAuthStatus.AUTHENTICATED
    assert status.account_label == "octocat"
    assert start.flow_id not in adapter._device_sessions
    credential = adapter.load_credential(provider.id)
    assert credential is not None
    assert credential.access_token == "gho_test"
    assert credential.account_label == "octocat"
    assert "gho_test" not in status.model_dump_json()


async def test_expired_session_returns_expired(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    adapter = _adapter(
        tmp_path,
        {
            ("POST", GITHUB_DEVICE_CODE_URL): lambda req: httpx.Response(
                200,
                json={
                    "device_code": "dev",
                    "user_code": "CODE",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 1,
                    "interval": 5,
                },
            ),
        },
    )

    start = await adapter.start(provider, AuthStartRequest())
    adapter._device_sessions[start.flow_id].expires_at = int(time.time()) - 1
    status = await adapter.poll(provider, start.flow_id)

    assert status.status == ProviderAuthStatus.EXPIRED
    assert start.flow_id not in adapter._device_sessions


async def test_restart_restore_status(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    store = OAuthCredentialStore(tmp_path / "providers")
    store.save(
        OAuthCredential(
            provider_id=provider.id,
            access_token="gho_saved",
            account_label="octocat",
            scopes=["read:user"],
            created_at=1,
            updated_at=2,
        ),
        "builtin",
    )
    adapter = GitHubCopilotOAuthAdapter(credential_store=store)
    credential = store.load(provider.id, "builtin")

    status = await adapter.get_status(provider, credential)

    assert status.status == ProviderAuthStatus.AUTHENTICATED
    assert status.account_label == "octocat"


async def test_get_copilot_token_is_lazy_and_cached(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    calls = {"count": 0}
    store = OAuthCredentialStore(tmp_path / "providers")
    store.save(
        OAuthCredential(
            provider_id=provider.id,
            access_token="gho_saved",
            account_label="octocat",
            created_at=1,
            updated_at=2,
        ),
        "builtin",
    )

    def token_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        assert request.headers["authorization"] == "token gho_saved"
        return httpx.Response(
            200,
            json={
                "token": f"runtime-{calls['count']}",
                "expires_at": int(time.time()) + 1800,
                "refresh_in": 1500,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            },
        )

    adapter = _adapter(
        tmp_path,
        {("GET", COPILOT_TOKEN_URL): token_handler},
    )
    adapter.credential_store = store

    first = await adapter.get_copilot_token(provider.id)
    second = await adapter.get_copilot_token(provider.id)

    assert first.token == "runtime-1"
    assert second.token == "runtime-1"
    assert calls["count"] == 1


async def test_get_copilot_token_401_deletes_credential(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    adapter = _adapter(
        tmp_path,
        {
            ("GET", COPILOT_TOKEN_URL): lambda req: httpx.Response(
                401,
                json={"message": "bad credentials"},
            ),
        },
    )
    adapter.save_credential(
        OAuthCredential(
            provider_id=provider.id,
            access_token="gho_revoked",
            created_at=1,
            updated_at=2,
        ),
    )

    with pytest.raises(ProviderAuthRequiredError):
        await adapter.get_copilot_token(provider.id)

    assert adapter.load_credential(provider.id) is None


async def test_get_copilot_token_network_error_keeps_credential(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("temporary", request=request)

    adapter = _adapter(tmp_path, {("GET", COPILOT_TOKEN_URL): handler})
    adapter.save_credential(
        OAuthCredential(
            provider_id=provider.id,
            access_token="gho_saved",
            created_at=1,
            updated_at=2,
        ),
    )

    with pytest.raises(ProviderAuthTemporaryError):
        await adapter.get_copilot_token(provider.id)

    assert adapter.load_credential(provider.id) is not None


async def test_logout_clears_local_state(
    tmp_path: Path,
    provider: OpenAIProvider,
) -> None:
    adapter = _adapter(tmp_path, {})
    adapter.save_credential(
        OAuthCredential(
            provider_id=provider.id,
            access_token="gho_saved",
            created_at=1,
            updated_at=2,
        ),
    )
    adapter._runtime_tokens[provider.id] = CopilotApiToken(token="runtime")
    adapter._device_sessions["flow"] = DeviceSession(
        flow_id="flow",
        device_code="dev",
        user_code="CODE",
        verification_uri="https://github.com/login/device",
        expires_at=int(time.time()) + 900,
        interval=5,
    )

    await adapter.logout(provider, adapter.load_credential(provider.id))

    assert adapter.load_credential(provider.id) is None
    assert provider.id not in adapter._runtime_tokens
    assert not adapter._device_sessions
