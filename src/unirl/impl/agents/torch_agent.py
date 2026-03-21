"""TorchAgent — generic glue wrapper that composes an Actor and a Trainer.

``TorchAgent`` implements the core UniRL ``Agent`` protocol by delegating
action-selection to an :class:`~unirl.impl.interfaces.Actor` and
learning to a :class:`~unirl.impl.interfaces.Trainer`.  It accumulates
transitions in an :class:`~unirl.impl.buffers.EpisodeBuffer` and flushes
the buffer to the trainer at the end of each episode.

This design keeps the Actor/Trainer separation clean: the actor only
selects actions, and the trainer only updates parameters.  More advanced
agents (e.g. search-based, model-based) can subclass or compose
``TorchAgent`` and override either component without changing the other.
"""

from __future__ import annotations

from unirl.impl.buffers.episode_buffer import EpisodeBuffer
from unirl.impl.interfaces import Actor, Trainer
from unirl.interfaces.types import Transition


class TorchAgent[ObsT, ActT]:
    """Generic glue wrapper: ``Actor + Trainer → Agent``.

    The class satisfies the ``unirl.interfaces.Agent`` structural protocol,
    so it can be dropped into a :class:`~unirl.system.System` directly.

    Parameters
    ----------
    actor:
        Any object that satisfies :class:`~unirl.impl.interfaces.Actor`.
    trainer:
        Any object that satisfies :class:`~unirl.impl.interfaces.Trainer`.
        Pass ``None`` to disable learning (inference-only mode).
    """

    def __init__(
        self,
        actor: Actor[ObsT, ActT],
        trainer: Trainer[ObsT, ActT] | None = None,
    ) -> None:
        self._actor = actor
        self._trainer = trainer
        self._buffer: EpisodeBuffer[ObsT, ActT] = EpisodeBuffer()

    def act(self, obs: ObsT) -> ActT:
        """Select an action using the wrapped actor."""
        return self._actor.act(obs)

    def observe(
        self,
        obs: ObsT,
        action: ActT,
        reward: float,
        next_obs: ObsT,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record a transition and, at episode end, trigger a training update."""
        self._buffer.add(
            Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
            )
        )
        if (terminated or truncated) and self._trainer is not None:
            self._trainer.train(self._buffer.flush())
