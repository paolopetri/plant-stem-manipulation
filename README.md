# Plant stem manipulation

Non-prehensile manipulation of flexible plant stems with reinforcement learning (Isaac Lab 3, FR3 + fork).
Semester project, ETH Zurich MSc RSC, hosted at AUTOLAB (UNIMORE, Modena).

- Architecture and where things go: [docs/architecture.md](docs/architecture.md)
- Roadmap and open items: [docs/TODO.md](docs/TODO.md)
- FR3 + end-effector asset pipeline: [assets/fr3/README.md](assets/fr3/README.md)

## Setup

Requires an Isaac Lab 3 source checkout next to this repo (`../IsaacLab`, see `[tool.uv.sources]` in
`pyproject.toml`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra isaacsim --extra rsl-rl
```

This creates `.venv/` with Isaac Lab (editable, from `../IsaacLab`) and this package (`stem_manip`, editable).

## Commands

```bash
uv run isaaclab list_envs          # lists registered tasks, including this project's
uv run pytest                      # unit tests (no simulator)
```

Training commands are added once the first task is registered (docs/TODO.md, M4/M5).
