"""Component registry and system builder for the UniRL framework.

The registry maps string keys to concrete factory callables.  ``build_system``
uses explicit type variables so that pyright can verify the full type chain at
the call site.

YAML (or any other config source) should resolve to a key string and call
``build_system`` — the typing boundary is at that call, not inside YAML.
"""

from collections.abc import Callable
from typing import Any

from unirl.interfaces.adapter import ActAdapter, ObsAdapter
from unirl.interfaces.agent import Agent
from unirl.interfaces.env import Env
from unirl.system.system import System

# ---------------------------------------------------------------------------
# Registry tables
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, Callable[..., Any]] = {}
AGENT_REGISTRY: dict[str, Callable[..., Any]] = {}
OBS_ADAPTER_REGISTRY: dict[str, Callable[..., Any]] = {}
ACT_ADAPTER_REGISTRY: dict[str, Callable[..., Any]] = {}


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
# System builder — explicit type binding at the build boundary
# ---------------------------------------------------------------------------


def build_system[EnvObsT, EnvActT, AgentObsT, AgentActT](
    *,
    env: Env[EnvObsT, EnvActT],
    agent: Agent[AgentObsT, AgentActT],
    obs_adapter: ObsAdapter[EnvObsT, AgentObsT],
    act_adapter: ActAdapter[AgentActT, EnvActT],
) -> System[EnvObsT, EnvActT, AgentObsT, AgentActT]:
    """Assemble a :class:`System` from pre-constructed, typed components.

    Callers must supply fully-typed objects; type mismatches are caught by
    pyright at the call site before runtime.
    """
    return System(
        env=env,
        agent=agent,
        obs_adapter=obs_adapter,
        act_adapter=act_adapter,
    )
