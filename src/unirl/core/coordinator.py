"""Coordinator protocol and generic reference implementation for UniRL."""

from typing import Protocol

from unirl.core.adapter import ActAdapter, ObsAdapter
from unirl.core.agent import Agent
from unirl.core.batch_source import BatchSource
from unirl.core.env import Env
from unirl.core.learner import Learner
from unirl.core.rollout import Rollout


class Coordinator(Protocol):
    """Coordinator protocol.

    A coordinator drives the full training loop, delegating to a
    :class:`~unirl.core.rollout.Rollout`, a
    :class:`~unirl.core.batch_source.BatchSource`, and a
    :class:`~unirl.core.learner.Learner`.

    The control model is::

        Coordinator
          -> Rollout       (env interaction, trajectory production)
          -> BatchSource   (trajectory ingestion, batch sampling)
          -> Learner       (parameter update)
    """

    def run(self) -> None:
        """Execute the training loop."""
        ...


class GenericCoordinator[EnvObsT, EnvActT, AgentObsT, AgentActT, TrajT, BatchT]:
    """Reference coordinator implementation.

    Drives the standard collect-then-update loop::

        for each iteration:
            collect rollouts_per_iter episodes → ingest into batch_source
            perform updates_per_iter gradient updates from batch_source

    Parameter ownership::

        Agent reads shared parameters.
        Learner updates shared parameters.

    The shared parameter object is managed by the concrete ``Learner`` and
    ``Agent`` implementations; ``GenericCoordinator`` does not interact with
    it directly.

    Args:
        rollout: Rollout used to collect one episode at a time.
        env: Environment passed to each ``rollout.run_episode`` call.
        agent: Agent passed to each ``rollout.run_episode`` call.
        obs_adapter: Observation adapter passed to each call.
        act_adapter: Action adapter passed to each call.
        batch_source: Destination for collected trajectories and source of
            training batches.
        learner: Applies parameter updates from sampled batches.
        batch_size: Number of samples requested from ``batch_source`` per
            update step.
        rollouts_per_iter: Number of episodes collected before each update
            phase.
        updates_per_iter: Number of ``learner.update`` calls per iteration.
        max_iters: Maximum number of iterations to run.  ``None`` means run
            forever.
    """

    def __init__(
        self,
        *,
        rollout: Rollout[EnvObsT, EnvActT, AgentObsT, AgentActT, TrajT],
        env: Env[EnvObsT, EnvActT],
        agent: Agent[AgentObsT, AgentActT],
        obs_adapter: ObsAdapter[EnvObsT, AgentObsT],
        act_adapter: ActAdapter[AgentActT, EnvActT],
        batch_source: BatchSource[TrajT, BatchT],
        learner: Learner[BatchT],
        batch_size: int,
        rollouts_per_iter: int,
        updates_per_iter: int,
        max_iters: int | None = None,
    ) -> None:
        self._rollout = rollout
        self._env = env
        self._agent = agent
        self._obs_adapter = obs_adapter
        self._act_adapter = act_adapter
        self._batch_source = batch_source
        self._learner = learner
        self._batch_size = batch_size
        self._rollouts_per_iter = rollouts_per_iter
        self._updates_per_iter = updates_per_iter
        self._max_iters = max_iters

    def run(self) -> None:
        """Execute the collect-then-update loop.

        Runs until ``max_iters`` is reached, or indefinitely if
        ``max_iters`` is ``None``.
        """
        iters = 0
        while self._max_iters is None or iters < self._max_iters:
            for _ in range(self._rollouts_per_iter):
                traj: TrajT = self._rollout.run_episode(
                    self._env,
                    self._agent,
                    self._obs_adapter,
                    self._act_adapter,
                )
                self._batch_source.ingest(traj)

            for _ in range(self._updates_per_iter):
                batch: BatchT = self._batch_source.sample(self._batch_size)
                self._learner.update(batch)

            iters += 1
