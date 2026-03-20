"""Simple concrete agent for end-to-end demonstration."""

from dataclasses import dataclass

from unirl.registry.registry import register_agent


@dataclass
class SimpleAgentObs:
    """Agent-side observation: normalised position."""

    normalised: float


@dataclass
class SimpleAgentAct:
    """Agent-side action: direction as –1.0 or +1.0."""

    direction: float


@register_agent("simple_agent")
class SimpleAgent:
    """Stateless agent that always steps toward the origin."""

    def act(self, obs: SimpleAgentObs) -> SimpleAgentAct:
        direction = -1.0 if obs.normalised > 0.0 else 1.0
        return SimpleAgentAct(direction=direction)

    def observe(
        self,
        obs: SimpleAgentObs,
        action: SimpleAgentAct,
        reward: float,
        next_obs: SimpleAgentObs,
        terminated: bool,
        truncated: bool,
    ) -> None:
        pass
