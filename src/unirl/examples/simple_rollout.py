"""Simple concrete rollout for end-to-end demonstration."""

from unirl.core.adapter import ActAdapter, ObsAdapter
from unirl.core.agent import Agent
from unirl.core.env import Env
from unirl.core.types import StepResult, Transition


class SimpleRollout[EnvObsT, EnvActT, AgentObsT, AgentActT]:
    """Stateless rollout that runs one episode and returns a list of Transitions.

    This is the reference implementation of the
    :class:`~unirl.core.rollout.Rollout` protocol for simple, on-policy use
    cases.  It:

    - calls ``agent.reset()`` at the start of each episode
    - adapts observations and actions through the provided adapters
    - steps the environment until ``terminated`` or ``truncated``
    - returns ``list[Transition[AgentObsT, AgentActT]]`` as the trajectory
    """

    def run_episode(
        self,
        env: Env[EnvObsT, EnvActT],
        agent: Agent[AgentObsT, AgentActT],
        obs_adapter: ObsAdapter[EnvObsT, AgentObsT],
        act_adapter: ActAdapter[AgentActT, EnvActT],
    ) -> list[Transition[AgentObsT, AgentActT]]:
        """Run one episode and return the collected transitions."""
        agent.reset()
        raw_obs: EnvObsT = env.reset()
        agent_obs: AgentObsT = obs_adapter.to_agent_obs(raw_obs)

        transitions: list[Transition[AgentObsT, AgentActT]] = []

        while True:
            agent_act: AgentActT = agent.act(agent_obs)
            env_act: EnvActT = act_adapter.to_env_act(agent_act)
            result: StepResult[EnvObsT] = env.step(env_act)
            next_agent_obs: AgentObsT = obs_adapter.to_agent_obs(result.obs)

            transitions.append(
                Transition(
                    obs=agent_obs,
                    action=agent_act,
                    reward=result.reward,
                    next_obs=next_agent_obs,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            )

            if result.terminated or result.truncated:
                break

            agent_obs = next_agent_obs

        return transitions
