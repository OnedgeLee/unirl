"""Component registry for the UniRL framework.

The registry maps string keys to concrete factory callables.  Components are
assembled by the config loader or directly by user code.

YAML (or any other config source) should resolve to a key string and
instantiate the factory — the typing boundary is at the manual call site, not
inside YAML.
"""

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Registry tables
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, Callable[..., Any]] = {}
AGENT_REGISTRY: dict[str, Callable[..., Any]] = {}
OBS_ADAPTER_REGISTRY: dict[str, Callable[..., Any]] = {}
ACT_ADAPTER_REGISTRY: dict[str, Callable[..., Any]] = {}
ROLLOUT_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_env(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers an Env factory under *name*."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        ENV_REGISTRY[name] = factory
        return factory

    return decorator


def register_agent(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers an Agent factory under *name*."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        AGENT_REGISTRY[name] = factory
        return factory

    return decorator


def register_obs_adapter(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers an ObsAdapter factory under *name*."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        OBS_ADAPTER_REGISTRY[name] = factory
        return factory

    return decorator


def register_act_adapter(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers an ActAdapter factory under *name*."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        ACT_ADAPTER_REGISTRY[name] = factory
        return factory

    return decorator


# ---------------------------------------------------------------------------
# Rollout decorator
# ---------------------------------------------------------------------------


def register_rollout(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a Rollout factory under *name*."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        ROLLOUT_REGISTRY[name] = factory
        return factory

    return decorator
