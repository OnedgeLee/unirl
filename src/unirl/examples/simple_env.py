"""Simple concrete environment for end-to-end demonstration."""

from dataclasses import dataclass
from typing import Any

from unirl.core.types import StepResult
from unirl.registry.registry import register_env


@dataclass
class SimpleEnvObs:
    """Raw environment observation: a single float value."""

    value: float


@dataclass
class SimpleEnvAct:
    """Raw environment action: a signed step direction."""

    delta: float


@register_env("simple_env")
class SimpleEnv:
    """Stateful 1-D walk environment.

    The agent moves along a line; the episode terminates when the position
    leaves the interval ``[-limit, limit]``.
    """

    def __init__(self, limit: float = 5.0, max_steps: int = 50) -> None:
        self._limit = limit
        self._max_steps = max_steps
        self._position: float = 0.0
        self._step_count: int = 0

    def reset(self) -> SimpleEnvObs:
        self._position = 0.0
        self._step_count = 0
        return SimpleEnvObs(value=self._position)

    def step(self, action: SimpleEnvAct) -> StepResult[SimpleEnvObs]:
        self._position += action.delta
        self._step_count += 1

        terminated = abs(self._position) > self._limit
        truncated = self._step_count >= self._max_steps
        reward = 1.0 if not terminated else -1.0
        info: dict[str, Any] = {"step": self._step_count}

        return StepResult(
            obs=SimpleEnvObs(value=self._position),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
