"""UniRL impl agents package."""

from unirl.impl.agents.torch_agent import TorchAgent

try:
    from unirl.impl.agents.reinforce_agent import REINFORCEAgent

    __all__ = ["REINFORCEAgent", "TorchAgent"]
except ImportError:
    __all__ = ["TorchAgent"]
