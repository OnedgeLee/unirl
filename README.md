# UniRL

A strongly-typed, modular reinforcement learning framework built around composable interfaces.

## Overview

UniRL provides a clean, minimal foundation for building reinforcement learning systems in Python. Rather than shipping a monolithic algorithm library, UniRL defines a set of **structural protocols** (using Python's `typing.Protocol`) that describe how environments, agents, rollouts, batch sources, and learners must behave, then wires them together with an explicit, fully-typed data pipeline.

Key properties:

- **No inheritance required** — components satisfy interfaces structurally; any class with the right methods works.
- **Full generic type-safety** — all core protocols carry type parameters so that mismatches are caught by a static type checker (pyright in strict mode).
- **Pluggable by design** — a decorator-based registry and a YAML loader let you swap implementations without touching orchestration code.

---

## Architecture

### Dataflow overview

```
Obs → Act → Trajectory → Batch → Update
```

The control structure is:

```
Coordinator
  → Rollout       (env interaction, trajectory production)
       → Agent
       → Env
       → Adapters
  → BatchSource   (trajectory ingestion, batch sampling)
  → Learner       (parameter update)
```

Four type variables flow through the observation/action pipeline:

| Variable | Produced by | Consumed by |
|---|---|---|
| `EnvObsT` | `Env.reset` / `Env.step` | `ObsAdapter.to_agent_obs` |
| `AgentObsT` | `ObsAdapter.to_agent_obs` | `Agent.act` |
| `AgentActT` | `Agent.act` | `ActAdapter.to_env_act` |
| `EnvActT` | `ActAdapter.to_env_act` | `Env.step` |

### Parameter ownership

```
Agent reads shared parameters.
Learner updates shared parameters.
```

The shared parameter object (e.g. a `torch.nn.Module` or a plain weight dict) is not abstracted in core. Concrete implementations in `unirl.impl` manage it directly and are responsible for keeping the agent and learner in sync.

### Package layout

```
src/unirl/
├── __init__.py          # Top-level re-exports
├── core/                # Framework-level protocol definitions
│   ├── __init__.py
│   ├── types.py         # StepResult, Transition dataclasses
│   ├── env.py           # Env protocol
│   ├── agent.py         # Agent protocol
│   ├── adapter.py       # ObsAdapter, ActAdapter protocols
│   ├── rollout.py       # Rollout protocol
│   ├── batch_source.py  # BatchSource protocol
│   ├── learner.py       # Learner protocol
│   └── coordinator.py   # Coordinator protocol + GenericCoordinator
├── registry/
│   ├── __init__.py
│   └── registry.py      # @register_* decorators
├── config/
│   ├── __init__.py
│   └── loader.py        # components_from_yaml / components_from_config
├── impl/                # Concrete (torch-based) implementations
│   ├── __init__.py
│   ├── agents/
│   ├── rollouts/
│   ├── batch_sources/
│   ├── learners/
│   └── models/
└── examples/            # Reference implementations (torch-free)
    ├── simple_env.py
    ├── simple_agent.py
    ├── simple_adapter.py
    ├── simple_rollout.py
    └── configs/
        └── simple.yaml
```

---

## Core Interfaces

All interfaces are defined as `typing.Protocol` classes. You never need to subclass them — any object that exposes the required methods is automatically compatible.

### `Env[EnvObsT, EnvActT]`

```python
class Env[EnvObsT, EnvActT](Protocol):
    def reset(self) -> EnvObsT: ...
    def step(self, action: EnvActT) -> StepResult[EnvObsT]: ...
```

An environment resets to an initial observation and advances one step at a time given an action, returning a [`StepResult`](#stepresult).

### `Agent[AgentObsT, AgentActT]`

```python
class Agent[AgentObsT, AgentActT](Protocol):
    def act(self, obs: AgentObsT) -> AgentActT: ...
    def reset(self) -> None: ...
```

An agent is a *decision operator* — it selects actions given observations. It is **not** the place where learning happens.

- `act` produces an action from an observation.
- `reset` is called by `Rollout` at the start of each episode to clear any episode-local state (e.g. RNN hidden state, search tree, frame buffer).

### `ObsAdapter[EnvObsT, AgentObsT]`

```python
class ObsAdapter[EnvObsT, AgentObsT](Protocol):
    def to_agent_obs(self, env_obs: EnvObsT) -> AgentObsT: ...
```

Translates raw environment observations into the representation expected by the agent (e.g. normalisation, feature extraction, frame stacking).

### `ActAdapter[AgentActT, EnvActT]`

```python
class ActAdapter[AgentActT, EnvActT](Protocol):
    def to_env_act(self, agent_act: AgentActT) -> EnvActT: ...
```

Translates agent actions back into the format accepted by the environment (e.g. scaling, discretisation, encoding).

### `Rollout[EnvObsT, EnvActT, AgentObsT, AgentActT, TrajT]`

```python
class Rollout[EnvObsT, EnvActT, AgentObsT, AgentActT, TrajT](Protocol):
    def run_episode(
        self,
        env: Env[EnvObsT, EnvActT],
        agent: Agent[AgentObsT, AgentActT],
        obs_adapter: ObsAdapter[EnvObsT, AgentObsT],
        act_adapter: ActAdapter[AgentActT, EnvActT],
    ) -> TrajT: ...
```

`Rollout` owns environment interaction. It resets the env and agent, adapts observations and actions, steps the environment, and returns a trajectory of type `TrajT`. The trajectory type is left open so that different algorithms can use their own representations without being constrained by core.

### `BatchSource[TrajT, BatchT]`

```python
class BatchSource[TrajT, BatchT](Protocol):
    def ingest(self, traj: TrajT) -> None: ...
    def sample(self, batch_size: int) -> BatchT: ...
```

`BatchSource` ingests trajectories produced by `Rollout` and supplies training batches to `Learner`. Implementations cover on-policy accumulators, replay buffers, and target-building stores.

### `Learner[BatchT]`

```python
class Learner[BatchT](Protocol):
    def update(self, batch: BatchT) -> None: ...
```

`Learner` receives a training batch and performs a parameter update. It is responsible for computing the loss and updating the shared parameters read by `Agent`.

### `Coordinator`

```python
class Coordinator(Protocol):
    def run(self) -> None: ...
```

`Coordinator` drives the full training loop. The reference implementation is `GenericCoordinator`.

### `StepResult`

```python
@dataclass
class StepResult[EnvObsT]:
    obs: EnvObsT
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
```

The value returned by `Env.step`. `terminated` signals a natural episode end; `truncated` signals an artificial cut-off (e.g. time limit).

### `Transition`

```python
@dataclass
class Transition[AgentObsT, AgentActT]:
    obs: AgentObsT
    action: AgentActT
    reward: float
    next_obs: AgentObsT
    terminated: bool
    truncated: bool
```

A minimal convenience record for simple on-policy rollouts. Algorithm-specific trajectory types live in `unirl.impl` and are not constrained by this class.

---

## Usage

### 1 — Implement the protocols

```python
from dataclasses import dataclass
from unirl import Env, Agent, ObsAdapter, ActAdapter, StepResult

@dataclass
class MyEnvObs:
    position: float

@dataclass
class MyEnvAct:
    delta: float

class MyEnv:
    """A 1-D walk environment."""

    def reset(self) -> MyEnvObs:
        self._pos = 0.0
        return MyEnvObs(position=self._pos)

    def step(self, action: MyEnvAct) -> StepResult[MyEnvObs]:
        self._pos += action.delta
        terminated = abs(self._pos) > 5.0
        return StepResult(
            obs=MyEnvObs(position=self._pos),
            reward=1.0 if not terminated else -1.0,
            terminated=terminated,
            truncated=False,
            info={},
        )


@dataclass
class MyAgentObs:
    normalised: float

@dataclass
class MyAgentAct:
    direction: float

class MyAgent:
    """Always steps toward the origin."""

    def act(self, obs: MyAgentObs) -> MyAgentAct:
        return MyAgentAct(direction=-1.0 if obs.normalised > 0 else 1.0)

    def reset(self) -> None:
        pass  # stateless — nothing to reset


class MyObsAdapter:
    def to_agent_obs(self, env_obs: MyEnvObs) -> MyAgentObs:
        return MyAgentObs(normalised=env_obs.position / 5.0)

class MyActAdapter:
    def to_env_act(self, agent_act: MyAgentAct) -> MyEnvAct:
        return MyEnvAct(delta=agent_act.direction)
```

### 2 — Run a single episode

```python
from unirl.examples.simple_rollout import SimpleRollout

rollout = SimpleRollout()
transitions = rollout.run_episode(
    MyEnv(),
    MyAgent(),
    MyObsAdapter(),
    MyActAdapter(),
)
print(f"Episode length: {len(transitions)}")
print(f"Total reward:   {sum(t.reward for t in transitions):.1f}")
```

### 3 — Wire into the full training loop

```python
from unirl.core.coordinator import GenericCoordinator

coordinator = GenericCoordinator(
    rollout=rollout,
    env=MyEnv(),
    agent=MyAgent(),
    obs_adapter=MyObsAdapter(),
    act_adapter=MyActAdapter(),
    batch_source=my_batch_source,   # implements BatchSource
    learner=my_learner,             # implements Learner
    batch_size=64,
    rollouts_per_iter=4,
    updates_per_iter=1,
    max_iters=1000,
)
coordinator.run()
```

### 4 — Use the registry and YAML config (optional)

Register implementations with decorators so they can be looked up by name at runtime:

```python
from unirl.registry import register_env, register_agent, register_obs_adapter, register_act_adapter

@register_env("my_env")
class MyEnv:
    ...

@register_agent("my_agent")
class MyAgent:
    ...

@register_obs_adapter("my_obs_adapter")
class MyObsAdapter:
    ...

@register_act_adapter("my_act_adapter")
class MyActAdapter:
    ...
```

Describe the components in a YAML file:

```yaml
imports:
  - mypackage.my_env
  - mypackage.my_agent
  - mypackage.my_adapters

env:
  name: my_env

agent:
  name: my_agent

obs_adapter:
  name: my_obs_adapter

act_adapter:
  name: my_act_adapter
```

Load and assemble:

```python
from unirl.config import components_from_yaml
from unirl.examples.simple_rollout import SimpleRollout

env, agent, obs_adapter, act_adapter = components_from_yaml("path/to/config.yaml")
rollout = SimpleRollout()
transitions = rollout.run_episode(env, agent, obs_adapter, act_adapter)
```

Constructor keyword arguments are forwarded via an optional `kwargs` map under each component entry:

```yaml
env:
  name: my_env
  kwargs:
    limit: 10.0
    max_steps: 200
```

A fully worked example is available in [`src/unirl/examples/configs/simple.yaml`](src/unirl/examples/configs/simple.yaml) together with the concrete implementations in [`src/unirl/examples/`](src/unirl/examples/).

---

## Registry API

| Symbol | Description |
|---|---|
| `register_env(name)` | Decorator — register an `Env` factory under `name` |
| `register_agent(name)` | Decorator — register an `Agent` factory under `name` |
| `register_obs_adapter(name)` | Decorator — register an `ObsAdapter` factory under `name` |
| `register_act_adapter(name)` | Decorator — register an `ActAdapter` factory under `name` |
| `register_rollout(name)` | Decorator — register a `Rollout` factory under `name` |

---

## `impl/` policy

`src/unirl/impl/` is the home for concrete, typically torch-based implementations:

```
src/unirl/impl/
├── __init__.py
├── agents/         # concrete Agent implementations
├── rollouts/       # concrete Rollout implementations
├── batch_sources/  # concrete BatchSource implementations
├── learners/       # concrete Learner implementations
└── models/         # shared neural network components
```

`core/` defines contracts. `impl/` provides implementations. There is no duplicate protocol layer under `impl/`.

---

## Development Workflow

UniRL uses a modern, reproducible CI/CD setup powered by [uv](https://github.com/astral-sh/uv), [ruff](https://github.com/astral-sh/ruff), [pyright](https://github.com/microsoft/pyright), and [pytest](https://pytest.org).

### Python Version Strategy

UniRL targets **Python ≥ 3.12**. Older versions (3.11 and below) are not supported because UniRL relies on PEP 695 generic syntax (`class Foo[T]`) and other 3.12+ language features.

CI validates:
- **3.12** — baseline (minimum supported version)
- **3.13** — forward compatibility

### Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:

```bash
uv sync --extra dev
```

### Running checks locally

```bash
# Lint
uv run ruff check .

# Type check (strict)
uv run pyright

# Tests
uv run pytest
```

### CI Workflows

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | push / pull request | Lint (ruff), Type Check (pyright), Test (pytest) |

All CI jobs run in parallel across Python 3.12 and 3.13 with `fail-fast` disabled so every matrix entry is reported independently.

### Contributing

1. Fork the repository and create a feature branch.
2. Implement your changes and ensure all checks pass (preferably via the pinned toolchain: `uv run ruff check .`, `uv run pyright`, `uv run pytest`).
3. Open a pull request against `main` with a clear description of the change.

New environment or agent implementations should follow the pattern in `src/unirl/examples/`: use `@register_*` decorators, keep observation and action types as plain `dataclass` objects, and include tests in `tests/`.

---

## License

[MIT](LICENSE)

