"""REINFORCE agent — a fully-composed, registerable RL agent.

``REINFORCEAgent`` wires together:

- an MLP policy network (:class:`~unirl.impl.models.MLP`)
- a stochastic discrete actor that samples from a categorical distribution
- a :class:`~unirl.impl.learners.REINFORCETrainer` for policy-gradient updates
- a :class:`~unirl.impl.agents.TorchAgent` for the glue / episode bookkeeping

The agent is registered under the key ``"reinforce_agent"`` and can therefore
be instantiated from a YAML config:

.. code-block:: yaml

    agent:
      name: reinforce_agent
      kwargs:
        obs_dim: 4
        n_actions: 2

The observation type is ``torch.Tensor`` (shape ``(obs_dim,)``) and the
action type is ``int`` (a discrete action index).
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from unirl.impl.agents.torch_agent import TorchAgent
from unirl.impl.learners.reinforce import REINFORCETrainer
from unirl.impl.models.mlp import MLP
from unirl.registry.registry import register_agent


class _CategoricalActor:
    """Stochastic discrete actor backed by a categorical policy network."""

    def __init__(self, policy: MLP) -> None:
        self._policy = policy

    def act(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self._policy(obs)
            dist = Categorical(logits=logits)
            return int(dist.sample().item())


@register_agent("reinforce_agent")
class REINFORCEAgent:
    """REINFORCE (Monte Carlo policy gradient) agent for discrete action spaces.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the (flat) observation vector.
    n_actions:
        Number of discrete actions available.
    hidden_dims:
        Width of each hidden layer in the MLP policy (default: ``[64, 64]``).
    lr:
        Learning rate for the Adam optimiser (default: ``1e-3``).
    gamma:
        Discount factor used when computing episode returns (default: ``0.99``).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        dims = hidden_dims if hidden_dims is not None else [64, 64]
        policy = MLP(obs_dim, dims, n_actions)
        actor = _CategoricalActor(policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        trainer = REINFORCETrainer(policy, optimizer, gamma=gamma)
        self._agent: TorchAgent[torch.Tensor, int] = TorchAgent(
            actor=actor, trainer=trainer
        )

    def act(self, obs: torch.Tensor) -> int:
        """Select a discrete action by sampling from the learned policy."""
        return self._agent.act(obs)

    def observe(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record the transition; triggers a REINFORCE update at episode end."""
        self._agent.observe(obs, action, reward, next_obs, terminated, truncated)
