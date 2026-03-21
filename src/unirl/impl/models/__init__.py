"""UniRL impl models package."""

try:
    from unirl.impl.models.mlp import MLP as MLP
except (ImportError, RuntimeError):
    pass
