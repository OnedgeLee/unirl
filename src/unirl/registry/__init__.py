"""UniRL registry package."""

from unirl.registry.registry import ACT_ADAPTER_REGISTRY as ACT_ADAPTER_REGISTRY
from unirl.registry.registry import AGENT_REGISTRY as AGENT_REGISTRY
from unirl.registry.registry import ENV_REGISTRY as ENV_REGISTRY
from unirl.registry.registry import OBS_ADAPTER_REGISTRY as OBS_ADAPTER_REGISTRY
from unirl.registry.registry import ROLLOUT_REGISTRY as ROLLOUT_REGISTRY
from unirl.registry.registry import register_act_adapter as register_act_adapter
from unirl.registry.registry import register_agent as register_agent
from unirl.registry.registry import register_env as register_env
from unirl.registry.registry import register_obs_adapter as register_obs_adapter
from unirl.registry.registry import register_rollout as register_rollout

__all__ = [
    "ACT_ADAPTER_REGISTRY",
    "AGENT_REGISTRY",
    "ENV_REGISTRY",
    "OBS_ADAPTER_REGISTRY",
    "ROLLOUT_REGISTRY",
    "register_act_adapter",
    "register_agent",
    "register_env",
    "register_obs_adapter",
    "register_rollout",
]
