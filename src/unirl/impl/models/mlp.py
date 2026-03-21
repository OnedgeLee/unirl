"""MLP (multi-layer perceptron) neural network module."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise RuntimeError(
        "unirl.impl requires PyTorch. Install it with: pip install 'unirl[impl]'"
    ) from e


class MLP(nn.Module):
    """Fully-connected feed-forward network with configurable hidden layers.

    Parameters
    ----------
    input_dim:
        Size of the input feature vector.
    hidden_dims:
        Sequence of hidden layer widths.  An empty sequence produces a
        single linear transformation from ``input_dim`` to ``output_dim``.
    output_dim:
        Size of the output vector.
    activation:
        Activation class inserted after every hidden layer (default: ReLU).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
