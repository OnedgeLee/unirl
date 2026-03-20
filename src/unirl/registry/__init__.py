"""UniRL registry package."""

from unirl.registry.registry import (
    ACT_ADAPTER_REGISTRY,
    AGENT_REGISTRY,
    ENV_REGISTRY,
    OBS_ADAPTER_REGISTRY,
    build_system,
    register_act_adapter,
    register_agent,
    register_env,
    register_obs_adapter,
)

__all__ = [
    "ACT_ADAPTER_REGISTRY",
    "AGENT_REGISTRY",
    "ENV_REGISTRY",
    "OBS_ADAPTER_REGISTRY",
    "build_system",
    "register_act_adapter",
    "register_agent",
    "register_env",
    "register_obs_adapter",
]
