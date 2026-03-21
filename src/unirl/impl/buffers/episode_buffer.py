"""Episode buffer for storing agent transitions within a single episode."""

from __future__ import annotations

from unirl.interfaces.types import Transition


class EpisodeBuffer[ObsT, ActT]:
    """Accumulates transitions during one episode and flushes them on demand.

    This buffer does **not** depend on torch; it stores plain
    :class:`~unirl.interfaces.types.Transition` dataclass instances.
    """

    def __init__(self) -> None:
        self._transitions: list[Transition[ObsT, ActT]] = []

    def add(self, transition: Transition[ObsT, ActT]) -> None:
        """Append a single transition to the buffer."""
        self._transitions.append(transition)

    def flush(self) -> list[Transition[ObsT, ActT]]:
        """Return all stored transitions and clear the buffer."""
        data = list(self._transitions)
        self._transitions.clear()
        return data

    def __len__(self) -> int:
        return len(self._transitions)
