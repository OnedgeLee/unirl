"""Rollout protocol for the UniRL core."""

from typing import Protocol

from unirl.core.adapter import ActAdapter, ObsAdapter
from unirl.core.agent import Agent
from unirl.core.env import Env


class Rollout[EnvObsT, EnvActT, AgentObsT, AgentActT, TrajT](Protocol):
    """Rollout protocol.

    A rollout owns environment interaction for one episode and produces a
    trajectory.  It is responsible for:

    - calling ``env.reset()``
    - calling ``agent.reset()`` at episode start
    - adapting observations and actions through the adapter pair
    - stepping the environment
    - constructing and returning a trajectory of type ``TrajT``

    The trajectory type is intentionally left open so that different
    algorithms can produce their own trajectory representations without
    being constrained by core.  Algorithm-specific types live in
    ``unirl.impl``.
    """

    def run_episode(
        self,
        env: Env[EnvObsT, EnvActT],
        agent: Agent[AgentObsT, AgentActT],
        obs_adapter: ObsAdapter[EnvObsT, AgentObsT],
        act_adapter: ActAdapter[AgentActT, EnvActT],
    ) -> TrajT: ...
