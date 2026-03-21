"""System orchestration for the UniRL framework."""

from unirl.interfaces.adapter import ActAdapter, ObsAdapter
from unirl.interfaces.agent import Agent
from unirl.interfaces.env import Env
from unirl.interfaces.types import StepResult, Transition


class System[EnvObsT, EnvActT, AgentObsT, AgentActT]:
    """Orchestrates a single RL loop with explicit, fully-typed data flow.

    Type chain enforced at every step::

        env_obs (EnvObsT)
        → agent_obs (AgentObsT)   via obs_adapter
        → agent_act (AgentActT)   via agent.act
        → env_act   (EnvActT)     via act_adapter
        → StepResult[EnvObsT]     via env.step
        → next_agent_obs          via obs_adapter
        → agent.observe(...)
    """

    def __init__(
        self,
        env: Env[EnvObsT, EnvActT],
        agent: Agent[AgentObsT, AgentActT],
        obs_adapter: ObsAdapter[EnvObsT, AgentObsT],
        act_adapter: ActAdapter[AgentActT, EnvActT],
    ) -> None:
        self._env = env
        self._agent = agent
        self._obs_adapter = obs_adapter
        self._act_adapter = act_adapter

    def run_episode(self) -> list[Transition[AgentObsT, AgentActT]]:
        """Run one episode and return the list of transitions collected."""
        raw_obs: EnvObsT = self._env.reset()
        agent_obs: AgentObsT = self._obs_adapter.to_agent_obs(raw_obs)

        transitions: list[Transition[AgentObsT, AgentActT]] = []

        while True:
            agent_act: AgentActT = self._agent.act(agent_obs)
            env_act: EnvActT = self._act_adapter.to_env_act(agent_act)
            result: StepResult[EnvObsT] = self._env.step(env_act)
            next_agent_obs: AgentObsT = self._obs_adapter.to_agent_obs(result.obs)

            self._agent.observe(
                agent_obs,
                agent_act,
                result.reward,
                next_agent_obs,
                result.terminated,
                result.truncated,
            )

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
