"""Agent protocol for the UniRL interface system."""

from typing import Protocol


class Agent[AgentObsT, AgentActT](Protocol):
    """Agent protocol.

    An agent selects actions given observations and learns from experience.
    The ``observe`` method receives a fully-formed transition so that the agent
    can update its internal state (e.g., replay buffer, policy parameters).
    """

    def act(self, obs: AgentObsT) -> AgentActT: ...

    def observe(
        self,
        obs: AgentObsT,
        action: AgentActT,
        reward: float,
        next_obs: AgentObsT,
        terminated: bool,
        truncated: bool,
    ) -> None: ...
