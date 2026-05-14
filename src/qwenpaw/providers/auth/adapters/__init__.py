# -*- coding: utf-8 -*-
"""Built-in provider auth adapters."""

from __future__ import annotations

from ..registry import ProviderAuthRegistry, auth_registry
from .github_copilot import GitHubCopilotOAuthAdapter


def register_builtin_auth_adapters(
    registry: ProviderAuthRegistry | None = None,
) -> None:
    """Register built-in provider auth adapters."""
    target = registry or auth_registry
    target.register(GitHubCopilotOAuthAdapter())


__all__ = [
    "GitHubCopilotOAuthAdapter",
    "register_builtin_auth_adapters",
]
