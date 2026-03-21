"""Tests for the unirl.impl package.

Torch-dependent tests are skipped automatically when torch is not installed.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Phase 1 — Pure-Python components (no torch required)
# ---------------------------------------------------------------------------


class TestEpisodeBuffer:
    """EpisodeBuffer does not depend on torch."""

    def test_add_and_len(self) -> None:
        from unirl.impl.buffers.episode_buffer import EpisodeBuffer
        from unirl.interfaces.types import Transition

        buf: EpisodeBuffer[float, int] = EpisodeBuffer()
        assert len(buf) == 0
        buf.add(
            Transition(
                obs=0.0,
                action=1,
                reward=1.0,
                next_obs=0.1,
                terminated=False,
                truncated=False,
            )
        )
        assert len(buf) == 1

    def test_flush_returns_and_clears(self) -> None:
        from unirl.impl.buffers.episode_buffer import EpisodeBuffer
        from unirl.interfaces.types import Transition

        buf: EpisodeBuffer[float, int] = EpisodeBuffer()
        for i in range(3):
            buf.add(
                Transition(
                    obs=float(i),
                    action=i,
                    reward=float(i),
                    next_obs=float(i + 1),
                    terminated=i == 2,
                    truncated=False,
                )
            )
        data = buf.flush()
        assert len(data) == 3
        assert len(buf) == 0

    def test_flush_empty(self) -> None:
        from unirl.impl.buffers.episode_buffer import EpisodeBuffer

        buf: EpisodeBuffer[float, int] = EpisodeBuffer()
        assert buf.flush() == []


class TestInterfaces:
    """Structural protocol checks — no torch needed."""

    def test_actor_protocol_satisfied_by_callable_class(self) -> None:
        from unirl.impl.interfaces import Actor

        class ConstActor:
            def act(self, obs: float) -> int:
                return 0

        actor: Actor[float, int] = ConstActor()
        assert actor.act(1.0) == 0

    def test_trainer_protocol_satisfied_by_callable_class(self) -> None:
        from unirl.impl.interfaces import Trainer
        from unirl.interfaces.types import Transition

        calls: list[int] = []

        class NoOpTrainer:
            def train(self, transitions: list[Transition[float, int]]) -> None:
                calls.append(len(transitions))

        trainer: Trainer[float, int] = NoOpTrainer()
        t = Transition(
            obs=0.0,
            action=0,
            reward=1.0,
            next_obs=1.0,
            terminated=True,
            truncated=False,
        )
        trainer.train([t])
        assert calls == [1]

    def test_search_actor_protocol(self) -> None:
        from unirl.impl.interfaces import SearchActor

        class TreeSearchActor:
            def act_with_search(self, obs: float, budget: int) -> int:
                return budget % 2

        sa: SearchActor[float, int] = TreeSearchActor()
        assert sa.act_with_search(0.0, 10) == 0


class TestTorchAgent:
    """TorchAgent tests — no torch needed (uses plain-Python actor/trainer)."""

    def test_act_delegates_to_actor(self) -> None:
        from unirl.impl.agents.torch_agent import TorchAgent

        class FixedActor:
            def act(self, obs: float) -> int:
                return 42

        agent: TorchAgent[float, int] = TorchAgent(actor=FixedActor())
        assert agent.act(0.0) == 42

    def test_observe_accumulates_transitions(self) -> None:
        from unirl.impl.agents.torch_agent import TorchAgent

        class FixedActor:
            def act(self, obs: float) -> int:
                return 0

        agent: TorchAgent[float, int] = TorchAgent(actor=FixedActor())
        agent.observe(0.0, 0, 1.0, 1.0, False, False)
        assert len(agent._buffer) == 1

    def test_observe_triggers_train_at_episode_end(self) -> None:
        from unirl.impl.agents.torch_agent import TorchAgent
        from unirl.interfaces.types import Transition

        trained: list[list[Transition[float, int]]] = []

        class FixedActor:
            def act(self, obs: float) -> int:
                return 0

        class RecordingTrainer:
            def train(self, transitions: list[Transition[float, int]]) -> None:
                trained.append(list(transitions))

        agent: TorchAgent[float, int] = TorchAgent(
            actor=FixedActor(), trainer=RecordingTrainer()
        )
        agent.observe(0.0, 0, 1.0, 1.0, False, False)
        agent.observe(1.0, 0, 1.0, 2.0, True, False)  # episode end

        assert len(trained) == 1
        assert len(trained[0]) == 2

    def test_buffer_cleared_after_train(self) -> None:
        from unirl.impl.agents.torch_agent import TorchAgent

        class FixedActor:
            def act(self, obs: float) -> int:
                return 0

        class NoOpTrainer:
            def train(self, transitions: object) -> None:
                pass

        agent: TorchAgent[float, int] = TorchAgent(
            actor=FixedActor(), trainer=NoOpTrainer()
        )
        agent.observe(0.0, 0, 1.0, 1.0, True, False)
        assert len(agent._buffer) == 0

    def test_no_trainer_is_inference_only(self) -> None:
        from unirl.impl.agents.torch_agent import TorchAgent

        class FixedActor:
            def act(self, obs: float) -> int:
                return 7

        agent: TorchAgent[float, int] = TorchAgent(actor=FixedActor())
        agent.observe(0.0, 7, 1.0, 1.0, True, False)  # should not raise


# ---------------------------------------------------------------------------
# Phase 2 — Torch-dependent components (skipped if torch is not installed)
# ---------------------------------------------------------------------------


class TestMLP:
    def test_output_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.models.mlp import MLP

        model = MLP(input_dim=4, hidden_dims=[16, 16], output_dim=2)
        x = torch.zeros(4)
        out = model(x)
        assert out.shape == (2,)

    def test_no_hidden_layers(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.models.mlp import MLP

        model = MLP(input_dim=3, hidden_dims=[], output_dim=5)
        x = torch.ones(3)
        out = model(x)
        assert out.shape == (5,)

    def test_batched_input(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.models.mlp import MLP

        model = MLP(input_dim=4, hidden_dims=[8], output_dim=3)
        x = torch.zeros(10, 4)
        out = model(x)
        assert out.shape == (10, 3)


class TestREINFORCETrainer:
    def test_train_does_not_raise(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.learners.reinforce import REINFORCETrainer
        from unirl.impl.models.mlp import MLP
        from unirl.interfaces.types import Transition

        policy = MLP(4, [16], 2)
        optimizer = torch.optim.Adam(policy.parameters())
        trainer = REINFORCETrainer(policy, optimizer, gamma=0.99)

        transitions = [
            Transition(
                obs=torch.zeros(4),
                action=0,
                reward=1.0,
                next_obs=torch.zeros(4),
                terminated=False,
                truncated=False,
            ),
            Transition(
                obs=torch.ones(4),
                action=1,
                reward=-1.0,
                next_obs=torch.zeros(4),
                terminated=True,
                truncated=False,
            ),
        ]
        trainer.train(transitions)  # should not raise

    def test_train_empty_transitions(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.learners.reinforce import REINFORCETrainer
        from unirl.impl.models.mlp import MLP

        policy = MLP(4, [16], 2)
        optimizer = torch.optim.Adam(policy.parameters())
        trainer = REINFORCETrainer(policy, optimizer)
        trainer.train([])  # should not raise


class TestREINFORCEAgent:
    def test_act_returns_int(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.agents.reinforce_agent import REINFORCEAgent

        agent = REINFORCEAgent(obs_dim=4, n_actions=2)
        obs = torch.zeros(4)
        action = agent.act(obs)
        assert isinstance(action, int)
        assert action in (0, 1)

    def test_observe_and_learn(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.agents.reinforce_agent import REINFORCEAgent

        agent = REINFORCEAgent(obs_dim=4, n_actions=2, hidden_dims=[16], lr=1e-3)
        obs = torch.zeros(4)
        action = agent.act(obs)
        # Provide a full episode transition to trigger a weight update
        agent.observe(obs, action, 1.0, torch.ones(4), terminated=True, truncated=False)

    def test_registered_in_agent_registry(self) -> None:
        pytest.importorskip("torch")
        import unirl.impl.agents.reinforce_agent  # noqa: F401 — side-effect: registers
        from unirl.registry.registry import AGENT_REGISTRY

        assert "reinforce_agent" in AGENT_REGISTRY

    def test_custom_hidden_dims(self) -> None:
        torch = pytest.importorskip("torch")
        from unirl.impl.agents.reinforce_agent import REINFORCEAgent

        agent = REINFORCEAgent(obs_dim=8, n_actions=4, hidden_dims=[32, 32])
        obs = torch.zeros(8)
        action = agent.act(obs)
        assert 0 <= action < 4


class TestImplPackageImport:
    def test_top_level_non_torch_imports(self) -> None:
        """Structural protocols and EpisodeBuffer import without torch."""
        from unirl.impl.buffers.episode_buffer import EpisodeBuffer  # noqa: F401
        from unirl.impl.interfaces import (  # noqa: F401
            Actor,
            Checkpointable,
            SearchActor,
            Trainer,
        )

    def test_top_level_torch_imports(self) -> None:
        pytest.importorskip("torch")
        from unirl.impl import (  # noqa: F401
            Actor,
            Checkpointable,
            EpisodeBuffer,
            SearchActor,
            TorchAgent,
            Trainer,
        )
        from unirl.impl.agents.reinforce_agent import REINFORCEAgent  # noqa: F401
        from unirl.impl.models.mlp import MLP  # noqa: F401

    def test_root_unirl_does_not_import_torch(self) -> None:
        """Importing the root unirl package must not pull in torch."""
        import sys

        # Remove torch from sys.modules to simulate a torch-free environment
        # then re-import unirl to ensure no torch import occurs
        torch_mod = sys.modules.pop("torch", None)
        try:
            import importlib

            import unirl

            importlib.reload(unirl)
            assert "torch" not in sys.modules
        finally:
            if torch_mod is not None:
                sys.modules["torch"] = torch_mod

