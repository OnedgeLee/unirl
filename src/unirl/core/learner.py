"""Learner protocol for the UniRL core."""

from typing import Protocol


class Learner[BatchT](Protocol):
    """Learner protocol.

    A learner receives a training batch and performs a parameter update.

    Parameter ownership::

        Agent reads shared parameters.
        Learner updates shared parameters.

    The shared parameter object (e.g. a ``torch.nn.Module`` or a plain
    dict) is not abstracted in core.  Concrete implementations in
    ``unirl.impl`` manage it directly and are responsible for keeping the
    agent and learner in sync.
    """

    def update(self, batch: BatchT) -> None:
        """Compute the loss and apply one parameter update step."""
        ...
