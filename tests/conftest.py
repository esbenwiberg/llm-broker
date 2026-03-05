"""Shared test fixtures for LLM Broker tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from llm_broker.config import BrokerConfig, load_config, reset_config
from llm_broker.main import app
from llm_broker.router import reset_router

# Resolve the configs directory relative to the project root.
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture()
def config() -> BrokerConfig:
    """Load and return a fresh BrokerConfig from the project configs."""
    return load_config(CONFIGS_DIR)


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset module-level singletons between tests."""
    reset_config()
    reset_router()


@pytest.fixture()
def client(config: BrokerConfig) -> AsyncClient:
    """Return an async HTTPX test client with config pre-loaded."""
    import llm_broker.config as cfg_module

    cfg_module._config = config
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")
