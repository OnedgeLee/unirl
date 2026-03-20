"""Adapter protocols for the UniRL interface system."""

from typing import Protocol


class ObsAdapter[EnvObsT, AgentObsT](Protocol):
    """Maps environment observations to agent observations."""

    def to_agent_obs(self, env_obs: EnvObsT) -> AgentObsT: ...


class ActAdapter[AgentActT, EnvActT](Protocol):
    """Maps agent actions to environment actions."""

    def to_env_act(self, agent_act: AgentActT) -> EnvActT: ...
