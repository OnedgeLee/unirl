"""End-to-end integration tests covering Phases 1–7."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from unirl.examples.simple_adapter import SimpleActAdapter, SimpleObsAdapter
from unirl.examples.simple_agent import SimpleAgent, SimpleAgentAct, SimpleAgentObs
from unirl.examples.simple_env import SimpleEnv, SimpleEnvAct, SimpleEnvObs
from unirl.interfaces.types import StepResult, Transition
from unirl.registry.registry import (
    ACT_ADAPTER_REGISTRY,
    AGENT_REGISTRY,
    ENV_REGISTRY,
    OBS_ADAPTER_REGISTRY,
    build_system,
    register_agent,
    register_env,
)
from unirl.system.system import System

# ---------------------------------------------------------------------------
# Phase 1 — Core types
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_fields(self) -> None:
        result: StepResult[SimpleEnvObs] = StepResult(
            obs=SimpleEnvObs(value=1.0),
            reward=0.5,
            terminated=False,
            truncated=False,
            info={"step": 1},
        )
        assert result.obs.value == 1.0
        assert result.reward == 0.5
        assert result.info == {"step": 1}


class TestTransition:
    def test_fields(self) -> None:
        t: Transition[SimpleAgentObs, SimpleAgentAct] = Transition(
            obs=SimpleAgentObs(normalised=0.2),
            action=SimpleAgentAct(direction=1.0),
            reward=1.0,
            next_obs=SimpleAgentObs(normalised=0.4),
            terminated=False,
            truncated=False,
        )
        assert t.obs.normalised == 0.2
        assert t.action.direction == 1.0
        assert t.next_obs.normalised == 0.4


# ---------------------------------------------------------------------------
# Phase 2 — Interfaces (structural subtyping checks)
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_env_satisfies_protocol(self) -> None:
        from unirl.interfaces.env import Env

        env: Env[SimpleEnvObs, SimpleEnvAct] = SimpleEnv()
        obs = env.reset()
        assert isinstance(obs, SimpleEnvObs)

    def test_agent_satisfies_protocol(self) -> None:
        from unirl.interfaces.agent import Agent

        agent: Agent[SimpleAgentObs, SimpleAgentAct] = SimpleAgent()
        act = agent.act(SimpleAgentObs(normalised=0.5))
        assert isinstance(act, SimpleAgentAct)

    def test_obs_adapter_satisfies_protocol(self) -> None:
        from unirl.interfaces.adapter import ObsAdapter

        adapter: ObsAdapter[SimpleEnvObs, SimpleAgentObs] = SimpleObsAdapter(limit=5.0)
        agent_obs = adapter.to_agent_obs(SimpleEnvObs(value=2.5))
        assert agent_obs.normalised == pytest.approx(0.5)

    def test_act_adapter_satisfies_protocol(self) -> None:
        from unirl.interfaces.adapter import ActAdapter

        adapter: ActAdapter[SimpleAgentAct, SimpleEnvAct] = SimpleActAdapter(scale=1.0)
        env_act = adapter.to_env_act(SimpleAgentAct(direction=-1.0))
        assert env_act.delta == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Phase 3 — System
# ---------------------------------------------------------------------------


def _make_system() -> System[
    SimpleEnvObs, SimpleEnvAct, SimpleAgentObs, SimpleAgentAct
]:
    return System(
        env=SimpleEnv(limit=5.0, max_steps=20),
        agent=SimpleAgent(),
        obs_adapter=SimpleObsAdapter(limit=5.0),
        act_adapter=SimpleActAdapter(scale=1.0),
    )


class TestSystem:
    def test_run_episode_returns_transitions(self) -> None:
        system = _make_system()
        transitions = system.run_episode()
        assert len(transitions) > 0
        assert isinstance(transitions[0], Transition)

    def test_run_episode_terminates(self) -> None:
        system = _make_system()
        transitions = system.run_episode()
        last = transitions[-1]
        assert last.terminated or last.truncated

    def test_run_episode_multiple_times(self) -> None:
        system = _make_system()
        for _ in range(3):
            transitions = system.run_episode()
            assert len(transitions) > 0


# ---------------------------------------------------------------------------
# Phase 4 — Examples
# ---------------------------------------------------------------------------


class TestSimpleEnv:
    def test_reset(self) -> None:
        env = SimpleEnv()
        obs = env.reset()
        assert obs.value == 0.0

    def test_step(self) -> None:
        env = SimpleEnv()
        env.reset()
        result = env.step(SimpleEnvAct(delta=1.0))
        assert result.obs.value == pytest.approx(1.0)
        assert result.reward == 1.0
        assert not result.terminated

    def test_termination_on_out_of_bounds(self) -> None:
        env = SimpleEnv(limit=1.0)
        env.reset()
        result = env.step(SimpleEnvAct(delta=2.0))
        assert result.terminated

    def test_truncation_on_max_steps(self) -> None:
        env = SimpleEnv(limit=100.0, max_steps=1)
        env.reset()
        result = env.step(SimpleEnvAct(delta=0.1))
        assert result.truncated


class TestSimpleAgent:
    def test_act_positive_normalised(self) -> None:
        agent = SimpleAgent()
        act = agent.act(SimpleAgentObs(normalised=0.5))
        assert act.direction == -1.0

    def test_act_negative_normalised(self) -> None:
        agent = SimpleAgent()
        act = agent.act(SimpleAgentObs(normalised=-0.5))
        assert act.direction == 1.0


# ---------------------------------------------------------------------------
# Phase 5 — Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_example_classes_are_registered(self) -> None:
        # Import triggers @register_* decorators
        import unirl.examples.simple_adapter  # noqa: F401
        import unirl.examples.simple_agent  # noqa: F401
        import unirl.examples.simple_env  # noqa: F401

        assert "simple_env" in ENV_REGISTRY
        assert "simple_agent" in AGENT_REGISTRY
        assert "simple_obs_adapter" in OBS_ADAPTER_REGISTRY
        assert "simple_act_adapter" in ACT_ADAPTER_REGISTRY

    def test_register_env_decorator(self) -> None:
        @register_env("_test_env")
        class _TestEnv:
            def reset(self) -> SimpleEnvObs:
                return SimpleEnvObs(value=0.0)

            def step(self, action: SimpleEnvAct) -> StepResult[SimpleEnvObs]:
                return StepResult(
                    obs=SimpleEnvObs(0.0),
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    info={},
                )

        assert "_test_env" in ENV_REGISTRY

    def test_register_agent_decorator(self) -> None:
        @register_agent("_test_agent")
        class _TestAgent:
            def act(self, obs: SimpleAgentObs) -> SimpleAgentAct:
                return SimpleAgentAct(direction=0.0)

            def observe(
                self,
                obs: SimpleAgentObs,
                action: SimpleAgentAct,
                reward: float,
                next_obs: SimpleAgentObs,
                terminated: bool,
                truncated: bool,
            ) -> None:
                pass

        assert "_test_agent" in AGENT_REGISTRY

    def test_build_system_returns_system(self) -> None:
        import unirl.examples.simple_adapter  # noqa: F401
        import unirl.examples.simple_agent  # noqa: F401
        import unirl.examples.simple_env  # noqa: F401

        system = build_system(
            env=SimpleEnv(),
            agent=SimpleAgent(),
            obs_adapter=SimpleObsAdapter(),
            act_adapter=SimpleActAdapter(),
        )
        assert isinstance(system, System)


# ---------------------------------------------------------------------------
# Phase 6 — YAML integration
# ---------------------------------------------------------------------------


class TestYAMLIntegration:
    def test_system_from_yaml(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            imports:
              - unirl.examples.simple_env
              - unirl.examples.simple_agent
              - unirl.examples.simple_adapter

            env:
              name: simple_env
              kwargs:
                limit: 5.0
                max_steps: 10

            agent:
              name: simple_agent

            obs_adapter:
              name: simple_obs_adapter
              kwargs:
                limit: 5.0

            act_adapter:
              name: simple_act_adapter
              kwargs:
                scale: 1.0
        """)
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_text)

        from unirl.config.loader import system_from_yaml

        system = system_from_yaml(config_file)
        transitions = system.run_episode()
        assert len(transitions) > 0

    def test_unknown_env_raises(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            env:
              name: no_such_env
            agent:
              name: simple_agent
            obs_adapter:
              name: simple_obs_adapter
            act_adapter:
              name: simple_act_adapter
        """)
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml_text)

        from unirl.config.loader import system_from_yaml

        with pytest.raises(KeyError, match="no_such_env"):
            system_from_yaml(config_file)
