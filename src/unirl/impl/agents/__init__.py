"""UniRL impl agents package."""

from unirl.impl.agents.torch_agent import TorchAgent as TorchAgent

__all__ = ["TorchAgent"]

try:
    from unirl.impl.agents.reinforce_agent import REINFORCEAgent as REINFORCEAgent
except (ImportError, RuntimeError):
    pass
