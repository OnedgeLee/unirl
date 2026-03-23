"""Core data types for the UniRL framework."""

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
    """A single agent-side transition record.

    ``Transition`` is a minimal convenience type suitable for simple
    on-policy examples.  Algorithm-specific trajectory or batch types
    should live in ``unirl.impl`` and are not constrained by this class.
    """

    obs: AgentObsT
    action: AgentActT
    reward: float
    next_obs: AgentObsT
    terminated: bool
    truncated: bool
