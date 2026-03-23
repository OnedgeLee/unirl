"""YAML-based component configuration loader.

The YAML schema selects *which* registered implementation to instantiate and
supplies keyword arguments for each component constructor.  All typing logic
lives in Python; YAML only carries string keys and constructor parameters.

Minimal schema::

    env:
      name: simple_env
      kwargs:
        limit: 5.0
        max_steps: 50

    agent:
      name: simple_agent

    obs_adapter:
      name: simple_obs_adapter
      kwargs:
        limit: 5.0

    act_adapter:
      name: simple_act_adapter
      kwargs:
        scale: 1.0
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from unirl.registry.registry import (
    ACT_ADAPTER_REGISTRY,
    AGENT_REGISTRY,
    ENV_REGISTRY,
    OBS_ADAPTER_REGISTRY,
)


def _load_component(
    registry: dict[str, Any],
    config: dict[str, Any],
    label: str,
) -> Any:
    """Instantiate a registered component from a config sub-dict."""
    name: str = config["name"]
    if name not in registry:
        msg = f"Unknown {label} '{name}'. Registered: {sorted(registry.keys())}"
        raise KeyError(msg)
    kwargs: dict[str, Any] = config.get("kwargs", {})
    return registry[name](**kwargs)


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file and return the raw config dict."""
    with Path(path).open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def components_from_config(
    config: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Build the four core components from a parsed config dict.

    Returns ``(env, agent, obs_adapter, act_adapter)`` as plain ``Any``
    objects.  The type parameters collapse to ``Any`` at this boundary because
    YAML cannot carry Python types — pyright enforces types on concrete call
    sites, not here.
    """
    # Ensure any registered example modules are imported so that their
    # @register_* decorators run before we look them up.
    for module_name in config.get("imports", []):
        importlib.import_module(module_name)

    env = _load_component(ENV_REGISTRY, config["env"], "env")
    agent = _load_component(AGENT_REGISTRY, config["agent"], "agent")
    obs_adapter = _load_component(
        OBS_ADAPTER_REGISTRY, config["obs_adapter"], "obs_adapter"
    )
    act_adapter = _load_component(
        ACT_ADAPTER_REGISTRY, config["act_adapter"], "act_adapter"
    )
    return env, agent, obs_adapter, act_adapter


def components_from_yaml(path: str | Path) -> tuple[Any, Any, Any, Any]:
    """Load a YAML file and return ``(env, agent, obs_adapter, act_adapter)``."""
    return components_from_config(load_config(path))
