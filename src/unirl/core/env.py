"""Env protocol for the UniRL core."""

from typing import Protocol

from unirl.core.types import StepResult


class Env[EnvObsT, EnvActT](Protocol):
    """Environment protocol.

    Implementations must produce an observation on reset and a
    :class:`StepResult` on each step.  The environment is not responsible
    for constructing :class:`~unirl.core.types.Transition` objects — that
    is the responsibility of the :class:`~unirl.core.rollout.Rollout`.
    """

    def reset(self) -> EnvObsT: ...

    def step(self, action: EnvActT) -> StepResult[EnvObsT]: ...
