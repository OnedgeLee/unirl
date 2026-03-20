# unirl
A strongly-typed, modular reinforcement learning framework built around composable interfaces.

## Development Workflow

UniRL uses a modern, reproducible CI/CD setup powered by [uv](https://github.com/astral-sh/uv), [ruff](https://github.com/astral-sh/ruff), [pyright](https://github.com/microsoft/pyright), and [pytest](https://pytest.org).

### Python Version Strategy

UniRL targets **Python ≥ 3.12**. Older versions (3.11 and below) are not supported.

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
