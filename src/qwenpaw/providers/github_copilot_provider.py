# -*- coding: utf-8 -*-
"""GitHub Copilot built-in model provider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from openai import APIError, AsyncOpenAI

from .auth import ProviderAuthInfo, ProviderAuthType, auth_registry
from .auth.adapters.github_copilot import (
    GitHubCopilotOAuthAdapter,
    ProviderAuthRequiredError,
    ProviderAuthTemporaryError,
)
from .openai_provider import OpenAIProvider
from .provider import ModelInfo

if TYPE_CHECKING:
    from agentscope.model import ChatModelBase
    from .multimodal_prober import ProbeResult

logger = logging.getLogger(__name__)

GITHUB_COPILOT_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="gpt-4.1",
        name="GPT-4.1",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="o1",
        name="OpenAI o1",
        supports_image=False,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="o3-mini",
        name="OpenAI o3-mini",
        supports_image=False,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="claude-3.7-sonnet",
        name="Claude 3.7 Sonnet",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="claude-sonnet-4",
        name="Claude Sonnet 4",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
]


class GitHubCopilotProvider(OpenAIProvider):
    """GitHub Copilot provider backed by provider auth infrastructure."""

    def _auth_adapter(self) -> GitHubCopilotOAuthAdapter:
        adapter = auth_registry.get(self.id)
        if not isinstance(adapter, GitHubCopilotOAuthAdapter):
            raise ProviderAuthRequiredError(
                "GitHub Copilot OAuth adapter is not registered.",
            )
        return adapter

    def _client(self, timeout: float = 5) -> AsyncOpenAI:
        adapter = self._auth_adapter()
        base_url = (
            adapter.get_cached_copilot_endpoint(self.id) or self.base_url
        )
        return AsyncOpenAI(
            base_url=base_url,
            api_key="copilot-oauth",
            timeout=timeout,
            http_client=adapter.get_or_create_http_client(self.id),
        )

    async def check_connection(
        self,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        adapter = self._auth_adapter()
        if not adapter.has_valid_credential(self.id):
            return False, "GitHub Copilot is not authenticated."
        try:
            await adapter.get_copilot_token(self.id)
        except ProviderAuthRequiredError as exc:
            return False, str(exc)
        except ProviderAuthTemporaryError as exc:
            return False, str(exc)
        except Exception:
            logger.warning(
                "Failed to obtain GitHub Copilot runtime token",
                exc_info=True,
            )
            return False, "Failed to connect to GitHub Copilot."
        return await super().check_connection(timeout=timeout)

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Fetch Copilot models, falling back to the static catalog."""
        adapter = self._auth_adapter()
        if not adapter.has_valid_credential(self.id):
            return list(GITHUB_COPILOT_MODELS)

        seed_by_id = {model.id: model for model in GITHUB_COPILOT_MODELS}
        try:
            payload = await self._client(timeout=timeout).models.list(
                timeout=timeout,
            )
            discovered = self._normalize_models_payload(payload)
        except APIError:
            return list(GITHUB_COPILOT_MODELS)
        except Exception:
            logger.warning(
                "Unexpected error while listing GitHub Copilot models; "
                "returning static catalog.",
                exc_info=True,
            )
            return list(GITHUB_COPILOT_MODELS)

        merged: List[ModelInfo] = []
        seen: set[str] = set()
        for model in discovered:
            seed = seed_by_id.get(model.id)
            if seed is not None:
                model.name = (
                    seed.name if model.name == model.id else model.name
                )
                model.supports_image = seed.supports_image
                model.supports_video = seed.supports_video
                model.supports_multimodal = bool(
                    seed.supports_image or seed.supports_video,
                )
                model.probe_source = seed.probe_source
            seen.add(model.id)
            merged.append(model)
        for seed in GITHUB_COPILOT_MODELS:
            if seed.id not in seen:
                merged.append(seed)
        return merged

    async def probe_model_multimodal(
        self,
        model_id: str,
        timeout: float = 10,  # pylint: disable=unused-argument
        image_only: bool = False,  # pylint: disable=unused-argument
    ) -> "ProbeResult":
        """Return documented multimodal capability without live probing."""
        from .multimodal_prober import ProbeResult

        for model in GITHUB_COPILOT_MODELS:
            if model.id == model_id:
                return ProbeResult(
                    supports_image=bool(model.supports_image),
                    supports_video=bool(model.supports_video),
                    image_message="documented",
                    video_message="documented",
                )
        return ProbeResult(
            image_message="Skipped: model not in Copilot catalog",
            video_message="Skipped: model not in Copilot catalog",
        )

    async def get_info(self, mock_secret: bool = True):
        """Return provider info with credential-store backed auth status."""
        info = await super().get_info(mock_secret=mock_secret)
        try:
            adapter = self._auth_adapter()
            credential = adapter.load_credential(self.id)
            status = await adapter.get_status(self, credential)
            info.auth = ProviderAuthInfo(
                type=ProviderAuthType.OAUTH_DEVICE_CODE,
                status=status.status,
                account_label=status.account_label,
                expires_at=status.expires_at,
                scopes=status.scopes,
                supports_logout=status.status.value == "authenticated",
                message=status.message,
            )
        except Exception:
            logger.debug(
                "Failed to resolve GitHub Copilot auth status",
                exc_info=True,
            )
        return info

    def get_chat_model_instance(self, model_id: str) -> "ChatModelBase":
        from .openai_chat_model_compat import OpenAIChatModelCompat

        adapter = self._auth_adapter()
        base_url = (
            adapter.get_cached_copilot_endpoint(self.id) or self.base_url
        )
        return OpenAIChatModelCompat(
            model_name=model_id,
            stream=True,
            api_key="copilot-oauth",
            stream_tool_parsing=False,
            client_kwargs={
                "base_url": base_url,
                "http_client": adapter.get_or_create_http_client(self.id),
            },
            generate_kwargs=self.get_effective_generate_kwargs(model_id),
        )


PROVIDER_GITHUB_COPILOT = GitHubCopilotProvider(
    id="github-copilot",
    name="GitHub Copilot",
    base_url="https://api.githubcopilot.com",
    chat_model="OpenAIChatModel",
    api_key="",
    api_key_prefix="",
    require_api_key=False,
    freeze_url=True,
    support_model_discovery=True,
    support_connection_check=True,
    auth_type=ProviderAuthType.OAUTH_DEVICE_CODE,
    models=GITHUB_COPILOT_MODELS,
    meta={
        "auth_provider": "github",
        "auth_hint": (
            "GitHub Copilot uses GitHub Device Code authentication. "
            "Sign in with your GitHub account to use Copilot models."
        ),
    },
)
