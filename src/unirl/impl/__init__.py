"""UniRL ``impl`` — torch-based RL agent implementations.

This package is the home for **library-quality, torch-based** implementations
of RL agents, neural modules, learners, and replay buffers.  It is kept
deliberately separate from the top-level ``unirl`` package so that **torch
is an opt-in dependency** — importing ``unirl`` alone never pulls in torch.

Package layout
--------------
::

    unirl.impl
    ├── interfaces.py    # Torch-side internal protocols (Actor, Trainer, …)
    ├── agents/          # Glue wrappers and fully-composed agents
    │   ├── torch_agent.py     # TorchAgent — generic Actor + Trainer wrapper
    │   └── reinforce_agent.py # REINFORCEAgent — discrete policy-gradient agent
    ├── models/          # Reusable neural modules
    │   └── mlp.py             # Multi-layer perceptron
    ├── learners/        # Learning / training logic
    │   └── reinforce.py       # REINFORCE (Monte Carlo PG) trainer
    └── buffers/         # Replay and episode storage
        └── episode_buffer.py  # In-memory per-episode trajectory buffer

Authoring a new agent
---------------------
1.  **Define an Actor** — implement
    :class:`~unirl.impl.interfaces.Actor` (one ``act`` method).
2.  **Define a Trainer** (optional) — implement
    :class:`~unirl.impl.interfaces.Trainer` (one ``train`` method).
3.  **Compose with TorchAgent** — pass your actor and trainer to
    :class:`~unirl.impl.agents.TorchAgent`; the wrapper handles episode
    bookkeeping and calls ``trainer.train`` at episode end automatically.
4.  **Register** — decorate your agent class with
    ``@register_agent("my_agent")`` so it can be loaded from a YAML config.

Example::

    from unirl.impl.agents.torch_agent import TorchAgent
    from unirl.registry import register_agent

    @register_agent("my_agent")
    class MyAgent(TorchAgent[MyObs, MyAct]):
        def __init__(self, ...):
            super().__init__(actor=MyActor(...), trainer=MyTrainer(...))

Extensibility
-------------
The ``Actor`` / ``Trainer`` split means future agent families (search-based,
model-based, AlphaGo Zero / MuZero) can introduce new protocols
(e.g. :class:`~unirl.impl.interfaces.SearchActor`) **without touching the
core interface layer**.  A search agent only needs to satisfy the top-level
``Agent`` protocol (``act`` + ``observe``) to plug into ``System``.

.. note::
    PyTorch is required to use this package::

        pip install "unirl[impl]"
"""

from unirl.impl.agents.torch_agent import TorchAgent
from unirl.impl.buffers.episode_buffer import EpisodeBuffer
from unirl.impl.interfaces import Actor, Checkpointable, SearchActor, Trainer

__all__ = [
    "Actor",
    "Checkpointable",
    "EpisodeBuffer",
    "SearchActor",
    "TorchAgent",
    "Trainer",
]

# Names that require PyTorch — populated when torch is available.
_TORCH_ONLY: frozenset[str] = frozenset({"MLP", "REINFORCEAgent"})

try:
    from unirl.impl.agents.reinforce_agent import REINFORCEAgent
    from unirl.impl.models.mlp import MLP

    __all__ = __all__ + ["MLP", "REINFORCEAgent"]
except (ImportError, RuntimeError):
    pass


def __getattr__(name: str) -> object:
    """Raise a helpful RuntimeError for torch-only names when torch is absent."""
    if name in _TORCH_ONLY:
        raise RuntimeError(
            f"'unirl.impl.{name}' requires PyTorch. "
            "Install it with: pip install 'unirl[impl]'"
        )
    raise AttributeError(f"module 'unirl.impl' has no attribute {name!r}")

