"""Agent protocol for the UniRL core."""

from typing import Protocol


class Agent[AgentObsT, AgentActT](Protocol):
    """Agent protocol.

    An agent selects actions given observations.  It is a *decision operator*
    used inside a :class:`~unirl.core.rollout.Rollout` — it is not the place
    where learning happens.

    Parameter ownership::

        Agent reads shared parameters.
        Learner updates shared parameters.

    The shared parameter object is not abstracted in core; concrete
    implementations in ``unirl.impl`` manage it directly.
    """

    def act(self, obs: AgentObsT) -> AgentActT: ...

    def reset(self) -> None:
        """Reset any episode-local state (e.g. RNN hidden state, search tree).

        Called by :class:`~unirl.core.rollout.Rollout` at the start of each
        episode, before the first call to :meth:`act`.
        """
        ...
