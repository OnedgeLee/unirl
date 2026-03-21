"""Env protocol for the UniRL interface system."""

from typing import Protocol

from unirl.interfaces.types import StepResult


class Env[EnvObsT, EnvActT](Protocol):
    """Environment protocol.

    Implementations must produce an observation on reset and a StepResult on
    each step. The environment is not responsible for constructing Transition
    objects — that is the responsibility of the agent or system layer.
    """

    def reset(self) -> EnvObsT: ...

    def step(self, action: EnvActT) -> StepResult[EnvObsT]: ...
