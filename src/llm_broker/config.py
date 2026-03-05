"""YAML configuration loader for LLM Broker.

Loads all configuration files at startup, validates with Pydantic,
and provides a singleton-style config object for the application.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from llm_broker.models import (
    KeyMapping,
    ProviderConfig,
    RepoConfig,
    RouterConfig,
)


class BrokerConfig:
    """Holds all loaded and validated configuration."""

    def __init__(
        self,
        providers: dict[str, ProviderConfig],
        repos: dict[str, RepoConfig],
        router: RouterConfig,
        keys: dict[str, KeyMapping],
    ) -> None:
        self.providers = providers
        self.repos = repos
        self.router = router
        self.keys = keys

    def resolve_key(self, api_key: str) -> KeyMapping | None:
        """Resolve an API key to its KeyMapping, or None if not found."""
        return self.keys.get(api_key)

    def get_repo_config(self, repo_name: str) -> RepoConfig | None:
        """Get a repo config by name, or None if not found."""
        return self.repos.get(repo_name)

    @property
    def repo_names(self) -> list[str]:
        """Return sorted list of repo names."""
        return sorted(self.repos.keys())


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_providers(config_dir: Path) -> dict[str, ProviderConfig]:
    """Load providers.yml and return validated provider configs."""
    data = _load_yaml(config_dir / "providers.yml")
    providers: dict[str, ProviderConfig] = {}
    for name, provider_data in data.get("providers", {}).items():
        providers[name] = ProviderConfig(**provider_data)
    return providers


def load_repos(config_dir: Path) -> dict[str, RepoConfig]:
    """Load all repo configs from configs/repos/*.yml."""
    repos_dir = config_dir / "repos"
    repos: dict[str, RepoConfig] = {}
    if repos_dir.is_dir():
        for yml_file in sorted(repos_dir.glob("*.yml")):
            data = _load_yaml(yml_file)
            repo_config = RepoConfig(**data)
            repos[repo_config.repo] = repo_config
    return repos


def load_router(config_dir: Path) -> RouterConfig:
    """Load router.yml and return validated router config."""
    data = _load_yaml(config_dir / "router.yml")
    return RouterConfig(**data.get("router", {}))


def load_keys(config_dir: Path) -> dict[str, KeyMapping]:
    """Load keys.yml and return validated key mappings."""
    data = _load_yaml(config_dir / "keys.yml")
    keys: dict[str, KeyMapping] = {}
    for key, mapping_data in data.get("keys", {}).items():
        keys[key] = KeyMapping(**mapping_data)
    return keys


def load_config(config_dir: str | Path | None = None) -> BrokerConfig:
    """Load all configuration files and return a BrokerConfig instance.

    Args:
        config_dir: Path to the configs directory. If None, uses the
                    CONFIG_DIR environment variable or defaults to ./configs.
    """
    if config_dir is None:
        config_dir = os.environ.get("CONFIG_DIR", "configs")
    config_path = Path(config_dir)

    providers = load_providers(config_path)
    repos = load_repos(config_path)
    router = load_router(config_path)
    keys = load_keys(config_path)

    return BrokerConfig(
        providers=providers,
        repos=repos,
        router=router,
        keys=keys,
    )


# Module-level singleton, initialized on first import of main or explicitly.
_config: BrokerConfig | None = None


def get_config() -> BrokerConfig:
    """Return the singleton config, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the singleton config (useful for testing)."""
    global _config
    _config = None
