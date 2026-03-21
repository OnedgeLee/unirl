"""REINFORCE (Monte Carlo policy gradient) learner."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from unirl.interfaces.types import Transition


def _compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted returns for a sequence of rewards."""
    returns: list[float] = []
    cumulative = 0.0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        returns.insert(0, cumulative)
    return returns


class REINFORCETrainer:
    """Policy-gradient trainer using the REINFORCE algorithm.

    After each episode the trainer re-runs the stochastic policy on each
    stored observation, computes the log-probability of the taken action,
    and minimises ``-sum(log_π(aₜ|sₜ) · Gₜ)`` where ``Gₜ`` is the
    discounted return from step ``t``.

    Parameters
    ----------
    policy:
        The neural network that maps observations to action logits.  Must
        accept a ``(obs_dim,)`` tensor and output ``(n_actions,)`` logits.
    optimizer:
        The torch optimiser that will update ``policy`` parameters.
    gamma:
        Discount factor for computing returns (default: 0.99).
    """

    def __init__(
        self,
        policy: nn.Module,
        optimizer: Optimizer,
        gamma: float = 0.99,
    ) -> None:
        self._policy = policy
        self._optimizer = optimizer
        self._gamma = gamma

    def train(self, transitions: list[Transition[torch.Tensor, int]]) -> None:
        """Run one REINFORCE update over the provided episode transitions."""
        if not transitions:
            return

        rewards = [t.reward for t in transitions]
        returns = _compute_returns(rewards, self._gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        # Normalise for training stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        log_probs: list[torch.Tensor] = []
        for t in transitions:
            logits = self._policy(t.obs)
            dist = Categorical(logits=logits)
            log_probs.append(dist.log_prob(torch.tensor(t.action)))

        policy_loss: torch.Tensor = torch.stack(
            [-lp * ret for lp, ret in zip(log_probs, returns_tensor)]
        ).sum()

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
