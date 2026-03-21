# UniRL

A strongly-typed, modular reinforcement learning framework built around composable interfaces.

## Overview

UniRL provides a clean, minimal foundation for building reinforcement learning systems in Python. Rather than shipping a monolithic algorithm library, UniRL defines a set of **structural protocols** (using Python's `typing.Protocol`) that describe how environments, agents, and adapters must behave, then wires them together with an explicit, fully-typed data pipeline.

Key properties:

- **No inheritance required** — components satisfy interfaces structurally; any class with the right methods works.
- **Full generic type-safety** — the `System` class propagates four distinct type parameters so that mismatches are caught by a static type checker (pyright in strict mode).
- **Pluggable by design** — a decorator-based registry and a YAML loader let you swap implementations without touching orchestration code.

---

## Architecture

### Data-flow overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                              System                                 │
│                                                                     │
│  env.reset()                                                        │
│       │ EnvObsT                                                     │
│       ▼                                                             │
│  obs_adapter.to_agent_obs()                                         │
│       │ AgentObsT                                                   │
│       ▼                                                             │
│  agent.act()                                                        │
│       │ AgentActT                                                   │
│       ▼                                                             │
│  act_adapter.to_env_act()                                           │
│       │ EnvActT                                                     │
│       ▼                                                             │
│  env.step()  ──►  StepResult[EnvObsT]                               │
│       │ EnvObsT (next obs)                                          │
│       ▼                                                             │
│  obs_adapter.to_agent_obs()  ──►  AgentObsT (next agent obs)        │
│       │                                                             │
│       ▼                                                             │
│  agent.observe(obs, action, reward, next_obs, terminated, truncated)│
└─────────────────────────────────────────────────────────────────────┘
```

Four type variables flow through the entire pipeline:

| Variable | Produced by | Consumed by |
|---|---|---|
| `EnvObsT` | `env.reset` / `env.step` | `obs_adapter.to_agent_obs` |
| `AgentObsT` | `obs_adapter.to_agent_obs` | `agent.act` / `agent.observe` |
| `AgentActT` | `agent.act` | `act_adapter.to_env_act` / `agent.observe` |
| `EnvActT` | `act_adapter.to_env_act` | `env.step` |

### Package layout

```
src/unirl/
├── __init__.py          # Top-level re-exports (no torch)
├── interfaces/          # Protocol definitions (structural typing)
│   ├── __init__.py
│   ├── types.py         # StepResult, Transition dataclasses
│   ├── env.py           # Env protocol
│   ├── agent.py         # Agent protocol
│   └── adapter.py       # ObsAdapter, ActAdapter protocols
├── system/
│   ├── __init__.py
│   └── system.py        # System — episode-loop orchestrator
├── registry/
│   ├── __init__.py
│   └── registry.py      # @register_* decorators + build_system
├── config/
│   ├── __init__.py
│   └── loader.py        # system_from_yaml / system_from_config
├── examples/            # Minimal reference agents (no torch)
│   ├── simple_env.py
│   ├── simple_agent.py
│   ├── simple_adapter.py
│   └── configs/
│       └── simple.yaml
└── impl/                # Torch-based implementations (opt-in)
    ├── __init__.py      # Convenient re-exports + usage docs
    ├── interfaces.py    # Torch-side protocols: Actor, Trainer, SearchActor, …
    ├── agents/
    │   ├── torch_agent.py     # TorchAgent — generic Actor + Trainer glue
    │   └── reinforce_agent.py # REINFORCEAgent — discrete policy-gradient
    ├── models/
    │   └── mlp.py             # MLP — configurable feed-forward network
    ├── learners/
    │   └── reinforce.py       # REINFORCETrainer — Monte Carlo PG update
    └── buffers/
        └── episode_buffer.py  # EpisodeBuffer — per-episode trajectory store
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
    def observe(
        self,
        obs: AgentObsT,
        action: AgentActT,
        reward: float,
        next_obs: AgentObsT,
        terminated: bool,
        truncated: bool,
    ) -> None: ...
```

An agent selects actions and learns from transitions. The `observe` method receives the complete transition tuple so that the agent can update its internal state (e.g. replay buffer, policy parameters).

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

The value returned by `env.step`. `terminated` signals a natural episode end; `truncated` signals an artificial cut-off (e.g. time limit).

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

A single agent-side transition record built by `System.run_episode` and returned as a list at episode end.

---

## System

`System` wires the four components together and runs the episode loop:

```python
from unirl import System

system = System(
    env=env,
    agent=agent,
    obs_adapter=obs_adapter,
    act_adapter=act_adapter,
)

transitions = system.run_episode()
```

`run_episode()` calls `env.reset()`, then loops until either `terminated` or `truncated` is `True`, collecting a `Transition` at each step and calling `agent.observe()` for online learning. The full list of transitions is returned for offline analysis.

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

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        pass  # stateless — no learning


class MyObsAdapter:
    def to_agent_obs(self, env_obs: MyEnvObs) -> MyAgentObs:
        return MyAgentObs(normalised=env_obs.position / 5.0)

class MyActAdapter:
    def to_env_act(self, agent_act: MyAgentAct) -> MyEnvAct:
        return MyEnvAct(delta=agent_act.direction)
```

### 2 — Assemble and run

```python
from unirl import System

system = System(
    env=MyEnv(),
    agent=MyAgent(),
    obs_adapter=MyObsAdapter(),
    act_adapter=MyActAdapter(),
)

transitions = system.run_episode()
print(f"Episode length: {len(transitions)}")
print(f"Total reward:   {sum(t.reward for t in transitions):.1f}")
```

### 3 — Use the registry and YAML config (optional)

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

Then describe the system in a YAML file:

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

And load it in one call:

```python
from unirl.config import system_from_yaml

system = system_from_yaml("path/to/config.yaml")
transitions = system.run_episode()
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

## Torch-based Implementations (`unirl.impl`)

`unirl.impl` provides **library-quality, torch-based** implementations of RL agents, neural modules, learners, and replay buffers.  Torch is an **opt-in dependency** — importing `unirl` alone never pulls it in.

### Installation

```bash
pip install "unirl[torch]"
# or directly:
pip install torch
```

### Quick start — REINFORCE agent

```python
import torch
from unirl.impl.agents.reinforce_agent import REINFORCEAgent

# Create a discrete-action REINFORCE agent for a 4-dim observation space
agent = REINFORCEAgent(obs_dim=4, n_actions=2)

obs = torch.zeros(4)
action = agent.act(obs)                   # int

agent.observe(obs, action, reward=1.0, next_obs=torch.zeros(4),
              terminated=True, truncated=False)
# ↑ triggers a REINFORCE parameter update at episode end
```

### Authoring a new agent

1. **Define an Actor** — implement [`Actor[ObsT, ActT]`](src/unirl/impl/interfaces.py) (one `act` method).
2. **Define a Trainer** (optional) — implement [`Trainer[ObsT, ActT]`](src/unirl/impl/interfaces.py) (one `train` method).
3. **Compose with `TorchAgent`** — the wrapper handles episode bookkeeping and automatically calls `trainer.train` at episode end.
4. **Register** — decorate with `@register_agent("my_agent")` for YAML/config use.

```python
import torch
from unirl.impl.agents.torch_agent import TorchAgent
from unirl.registry import register_agent


class MyActor:
    def act(self, obs: torch.Tensor) -> int:
        return 0  # replace with your policy


class MyTrainer:
    def train(self, transitions):
        pass  # replace with your learning update


@register_agent("my_agent")
class MyAgent:
    def __init__(self):
        self._agent = TorchAgent(actor=MyActor(), trainer=MyTrainer())

    def act(self, obs):
        return self._agent.act(obs)

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        self._agent.observe(obs, action, reward, next_obs, terminated, truncated)
```

### Torch-side protocols (`unirl.impl.interfaces`)

| Protocol | Description |
|---|---|
| `Actor[ObsT, ActT]` | Selects an action given an observation |
| `Trainer[ObsT, ActT]` | Runs a learning update on episode transitions |
| `SearchActor[ObsT, ActT]` | Actor with explicit search budget (MCTS, AlphaGo Zero, …) |
| `Checkpointable` | Saves/restores state to/from a path |

### Extensibility for non-standard agents

The `Actor` / `Trainer` split is intentional: search-based agents (AlphaGo Zero, MuZero), model-based agents, and any other exotic architecture only need to satisfy the top-level `Agent` protocol (`act` + `observe`) to plug into `System`.  They can implement `SearchActor` alongside or instead of `Actor` without any changes to the core interface layer.

---

## Registry API

| Symbol | Description |
|---|---|
| `register_env(name)` | Decorator — register an `Env` factory under `name` |
| `register_agent(name)` | Decorator — register an `Agent` factory under `name` |
| `register_obs_adapter(name)` | Decorator — register an `ObsAdapter` factory under `name` |
| `register_act_adapter(name)` | Decorator — register an `ActAdapter` factory under `name` |
| `build_system(*, env, agent, obs_adapter, act_adapter)` | Typed helper — assemble a `System` from pre-constructed components |

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
| `release.yml` | tag (`v*`) / manual dispatch | Build, Publish to PyPI |

All CI jobs run in parallel across Python 3.12 and 3.13 with `fail-fast` disabled so every matrix entry is reported independently.

### Contributing

1. Fork the repository and create a feature branch.
2. Implement your changes and ensure all checks pass (preferably via the pinned toolchain: `uv run ruff check .`, `uv run pyright`, `uv run pytest`).
3. Open a pull request against `main` with a clear description of the change.

New environment or agent implementations should follow the pattern in `src/unirl/examples/`: use `@register_*` decorators, keep observation and action types as plain `dataclass` objects, and include tests in `tests/`.

---

## License

[MIT](LICENSE)
