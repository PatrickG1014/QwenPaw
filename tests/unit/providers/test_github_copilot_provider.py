# -*- coding: utf-8 -*-
# pylint: disable=protected-access,redefined-outer-name,unused-argument
"""Tests for GitHub Copilot provider."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from qwenpaw.providers.auth.adapters.github_copilot import (
    GitHubCopilotOAuthAdapter,
)
from qwenpaw.providers.auth.credential_store import OAuthCredentialStore
from qwenpaw.providers.auth.models import OAuthCredential, ProviderAuthType
from qwenpaw.providers.auth.registry import auth_registry
from qwenpaw.providers.github_copilot_provider import (
    GITHUB_COPILOT_MODELS,
    GitHubCopilotProvider,
    PROVIDER_GITHUB_COPILOT,
)


@pytest.fixture(autouse=True)
def _isolate_auth(tmp_path: Path, monkeypatch):
    import qwenpaw.security.secret_store as mod

    test_key = bytes.fromhex(
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    )
    monkeypatch.setattr(mod, "_cached_master_key", test_key)
    monkeypatch.setattr(mod, "_cached_fernet", None)
    monkeypatch.setattr(mod, "_get_secret_dir", lambda: tmp_path)
    auth_registry.clear_for_test()
    yield
    auth_registry.clear_for_test()


@pytest.fixture
def adapter(tmp_path: Path) -> GitHubCopilotOAuthAdapter:
    adapter = GitHubCopilotOAuthAdapter(
        credential_store=OAuthCredentialStore(tmp_path / "oauth"),
    )
    auth_registry.register(adapter)
    return adapter


def _provider() -> GitHubCopilotProvider:
    return GitHubCopilotProvider(
        id="github-copilot",
        name="GitHub Copilot",
        base_url="https://api.githubcopilot.com",
        chat_model="OpenAIChatModel",
        require_api_key=False,
        freeze_url=True,
        support_model_discovery=True,
        support_connection_check=True,
        auth_type=ProviderAuthType.OAUTH_DEVICE_CODE,
        models=list(GITHUB_COPILOT_MODELS),
    )


def test_default_provider_metadata() -> None:
    provider = PROVIDER_GITHUB_COPILOT
    assert provider.id == "github-copilot"
    assert provider.auth_type == ProviderAuthType.OAUTH_DEVICE_CODE
    assert provider.require_api_key is False
    assert provider.api_key == ""
    assert provider.freeze_url is True
    assert provider.support_model_discovery is True


async def test_get_info_does_not_expose_token(
    adapter: GitHubCopilotOAuthAdapter,
) -> None:
    adapter.credential_store.save(
        OAuthCredential(
            provider_id="github-copilot",
            access_token="gho_secret",
            account_label="octocat",
            created_at=1,
            updated_at=2,
        ),
    )

    info = await _provider().get_info()

    assert info.auth_type == ProviderAuthType.OAUTH_DEVICE_CODE
    assert info.require_api_key is False
    assert info.auth is not None
    assert info.auth.status == "authenticated"
    assert info.auth.account_label == "octocat"
    assert "gho_secret" not in info.model_dump_json()


async def test_check_connection_unauthenticated_returns_false(
    adapter: GitHubCopilotOAuthAdapter,
) -> None:
    ok, message = await _provider().check_connection()

    assert ok is False
    assert "not authenticated" in message.lower()


async def test_fetch_models_returns_static_catalog_when_unauthenticated(
    adapter: GitHubCopilotOAuthAdapter,
) -> None:
    models = await _provider().fetch_models()
    ids = {model.id for model in models}

    assert "gpt-4o" in ids
    assert "claude-sonnet-4" in ids


async def test_fetch_models_merges_discovered_models(
    adapter: GitHubCopilotOAuthAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.credential_store.save(
        OAuthCredential(
            provider_id="github-copilot",
            access_token="gho_secret",
            created_at=1,
            updated_at=2,
        ),
    )
    provider = _provider()

    class _Models:
        async def list(self, timeout: float = 5):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(id="gpt-4o", name="gpt-4o"),
                    SimpleNamespace(id="new-copilot-model", name="New"),
                ],
            )

    monkeypatch.setattr(
        provider,
        "_client",
        lambda timeout=5: SimpleNamespace(models=_Models()),
    )

    models = await provider.fetch_models()
    ids = [model.id for model in models]

    assert ids.count("gpt-4o") == 1
    assert "new-copilot-model" in ids
    assert "claude-sonnet-4" in ids
    gpt_4o = next(model for model in models if model.id == "gpt-4o")
    assert gpt_4o.supports_image is True


async def test_probe_multimodal_uses_static_catalog(
    adapter: GitHubCopilotOAuthAdapter,
) -> None:
    result = await _provider().probe_model_multimodal("gpt-4o")

    assert result.supports_image is True
    assert result.supports_video is False


def test_get_chat_model_instance_does_not_mutate_provider_api_key(
    adapter: GitHubCopilotOAuthAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    captured: dict = {}

    class _StubChatModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "qwenpaw.providers.openai_chat_model_compat.OpenAIChatModelCompat",
        _StubChatModel,
    )

    model = provider.get_chat_model_instance("gpt-4o")

    assert isinstance(model, _StubChatModel)
    assert provider.api_key == ""
    assert captured["api_key"] == "copilot-oauth"
    assert "http_client" in captured["client_kwargs"]
