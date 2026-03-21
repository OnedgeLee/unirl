"""UniRL impl models package."""

try:
    from unirl.impl.models.mlp import MLP

    __all__ = ["MLP"]
except (ImportError, RuntimeError):
    __all__ = []
