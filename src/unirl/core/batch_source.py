"""BatchSource protocol for the UniRL core."""

from typing import Protocol


class BatchSource[TrajT, BatchT](Protocol):
    """BatchSource protocol.

    A batch source ingests trajectories produced by
    :class:`~unirl.core.rollout.Rollout` and supplies training batches to a
    :class:`~unirl.core.learner.Learner`.

    Implementations may cover:

    - on-policy trajectory accumulation
    - replay buffers
    - target-building stores

    Preprocessing may happen on ingest, on sample, or both — this protocol
    does not mandate a ``process()`` step.
    """

    def ingest(self, traj: TrajT) -> None:
        """Store or process a trajectory produced by a rollout."""
        ...

    def sample(self, batch_size: int) -> BatchT:
        """Return a training batch of the requested size."""
        ...
