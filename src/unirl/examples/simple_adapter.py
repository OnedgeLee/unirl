"""Simple observation and action adapters for end-to-end demonstration."""

from unirl.examples.simple_agent import SimpleAgentAct, SimpleAgentObs
from unirl.examples.simple_env import SimpleEnvAct, SimpleEnvObs
from unirl.registry.registry import register_act_adapter, register_obs_adapter


@register_obs_adapter("simple_obs_adapter")
class SimpleObsAdapter:
    """Normalises raw position by the environment limit."""

    def __init__(self, limit: float = 5.0) -> None:
        self._limit = limit

    def to_agent_obs(self, env_obs: SimpleEnvObs) -> SimpleAgentObs:
        return SimpleAgentObs(normalised=env_obs.value / self._limit)


@register_act_adapter("simple_act_adapter")
class SimpleActAdapter:
    """Scales agent direction back to a raw environment delta."""

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale

    def to_env_act(self, agent_act: SimpleAgentAct) -> SimpleEnvAct:
        return SimpleEnvAct(delta=agent_act.direction * self._scale)
