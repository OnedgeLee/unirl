"""Core data types for the UniRL interface system."""

from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult[EnvObsT]:
    """Result returned by the environment after a step."""

    obs: EnvObsT
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class Transition[AgentObsT, AgentActT]:
    """A single agent-side transition record."""

    obs: AgentObsT
    action: AgentActT
    reward: float
    next_obs: AgentObsT
    terminated: bool
    truncated: bool
