"""Torch-side internal protocols for UniRL implementations.

These protocols define the internal contracts used within ``unirl.impl``.
They are deliberately separate from the top-level ``unirl.interfaces``
protocols so that the core package never imports torch.

Protocol summary
----------------
``Actor[ObsT, ActT]``
    Selects an action given an observation.  Used inside ``TorchAgent`` to
    decouple action-selection from learning.

``Trainer[ObsT, ActT]``
    Runs a learning update given a completed episode's transitions.

``SearchActor[ObsT, ActT]``
    Extension of ``Actor`` for agents that perform explicit search (e.g.
    MCTS / AlphaGo Zero) and expose an optional budget parameter.

``Checkpointable``
    Mix-in protocol for objects that can serialise / deserialise their state
    to/from a filesystem path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from unirl.interfaces.types import Transition


class Actor[ObsT, ActT](Protocol):
    """Selects an action given an agent-side observation."""

    def act(self, obs: ObsT) -> ActT: ...


class Trainer[ObsT, ActT](Protocol):
    """Runs a learning update on a completed episode trajectory."""

    def train(self, transitions: list[Transition[ObsT, ActT]]) -> None: ...


class SearchActor[ObsT, ActT](Protocol):
    """Actor that performs explicit search (e.g. MCTS) within a budget.

    Agents such as AlphaGo Zero, MuZero, or any tree-search variant can
    implement this protocol instead of (or in addition to) ``Actor``.
    The ``budget`` parameter controls the number of search iterations.
    """

    def act_with_search(self, obs: ObsT, budget: int) -> ActT: ...


class Checkpointable(Protocol):
    """Mix-in for objects that can save and restore their state."""

    def save_checkpoint(self, path: Path) -> None: ...

    def load_checkpoint(self, path: Path) -> None: ...
