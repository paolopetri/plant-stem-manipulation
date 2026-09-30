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

All commands are run from the repo root.

```bash
uv run isaaclab list_envs                               # lists registered tasks, including this project's
uv run pytest                                           # unit tests (no simulator)
uv run --extra isaacsim python scripts/check_fr3.py     # FR3 + fork (fork_v2): pass/fail check in simulation
uv run --extra isaacsim python scripts/check_stem.py    # stem: pass/fail check in simulation
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick   # stem: measure one test (see below)
```

Simulation scripts run without a window unless a viewer is requested with `--viz newton_gl`.
Training commands are added once the first task is registered (docs/TODO.md, M4/M5).

## Stem tests

The stem parameters and solver settings are in `assets/stem/stem.yaml`. Two scripts test them.

### `check_stem.py`: does the stem work? (pass/fail)

```bash
uv run --extra isaacsim python scripts/check_stem.py
```

Spawns two stems per environment from the yaml, both clamped at the base: one upright, whose tip is first
kicked sideways and later pushed with a steady force, and one horizontal, which sags under its own weight. Prints one `[PASS]` / `[FAIL]` line per check:

| Check | Passes if |
|---|---|
| segments, start poses, mass | the simulated stem has the yaml's segment count, shape and mass |
| joint stiffness, joint damping | the solver's per-joint values equal those computed from the yaml |
| base fixed | the clamped segment does not move at all |
| deflects when kicked | the tip moves by more than 1 cm and all values stay finite |
| springs back | the tip is back within 1 mm of upright at the end |
| swing frequency | within 15 % of the first bending frequency of a clamped beam |
| damping ratio | within 0.02 of the value set by `damping_time` |
| deflects when pushed | a steady sideways force of 0.05 N on the last segment deflects it within 10 % of the value computed by hand |
| returns after the push | the pushed point is back within 1 mm after the force is removed |
| cantilever sag | the horizontal stem's tip drop is within 5 % of the value computed by hand |

Run it after every change to the stem or the yaml.

| Option | Default | Meaning |
|---|---|---|
| `--num_envs N` | 2 | number of environments (each has one upright and one horizontal stem) |
| `--steps N` | 400 | simulation steps after the kick (one step = `solver.sim_dt` = 10 ms) |
| `--viz newton_gl` | off | open a viewer window (runs once at full speed; use the script below to watch) |

### `sweep_stem_solver.py`: how accurate is the stem? (measurement, and the script to watch the stem)

Runs **one test with one setting** and prints one `RESULT` line. Without options it uses the values from the
yaml. The options replace single values for that run only; the yaml is not changed.

```bash
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test <sag|kick|push> [options]
```

**The three tests** (`--test`, default `sag`):

| Test | What happens | What the `RESULT` line reports | Expected for the current yaml |
|---|---|---|---|
| `sag` | The stem is clamped horizontally and sags under its own weight. | `sag_mm`: tip sag at the end. `expected_mm`: value computed by hand for the 20-segment model (bending + shear). `ratio`: simulated / expected. `drift_last_100_steps_mm`: how much the tip still moved in the last 100 steps (near 0 = at rest). | `ratio` close to 1 (1.01) |
| `kick` | The stem stands upright and its tip is kicked sideways with 1 m/s. | `first_peak_mm`: largest tip deflection. `frequency_hz`: swing frequency. `damping_ratio`: how fast the swing dies out. `final_offset_mm`: distance of the tip from upright at the end. | about 33 mm, 4.7 Hz, 0.06, near 0 |
| `push` | The stem stands upright and its last segment is pushed sideways with a constant force. | `deflection_mm`: how far the pushed point has moved at the end. `by_hand_without_gravity_mm`: value computed by hand, ignoring the stem's weight. `ratio`: simulated / by hand. `drift_last_100_steps_mm`: near 0 = at rest. | `ratio` about 1.04 (the stem's own weight makes it lean a little further) |

Both also report `cost` (substeps x iterations, the solver work per step) and `ms_per_step` (computing time
per simulation step, without rendering).

**Options to change the setting** (each defaults to the yaml value):

| Option | Yaml default | Meaning |
|---|---|---|
| `--substeps N` | 8 | solver substeps per simulation step |
| `--iterations N` | 10 | solver iterations per substep |
| `--stretch_ratio X` | 0.01 | stretch modulus as a multiple of the bend modulus |
| `--shear_ratio X` | 0.001 | shear modulus as a multiple of the bend modulus |
| `--damping MODE` | `bend_twist` | which deformations are damped: `bend_twist` (as in the project), `all` (also stretch and shear), `none` |

**Options for the run:**

| Option | Default | Meaning |
|---|---|---|
| `--push_force F` | 0.05 | force of the push test [N] |
| `--steps N` | 600 | simulation steps per run (one step = 10 ms, so 600 = 6 s) |
| `--num_envs N` | 1 | number of stems; use a large number to measure `ms_per_step` for training |
| `--viz newton_gl` | off | open a viewer window |
| `--slow_motion X` | 1 | only with a viewer: play X times slower than real time |

**Watching a test.** With `--viz newton_gl` the test plays in real time (or slower with `--slow_motion`), and
after the `RESULT` line it restarts from the beginning again and again until the window is closed. To pause,
press "Pause Simulation" in the viewer's left panel; "Resume Simulation" continues. The camera can be moved
while paused.

**Examples:**

```bash
# watch the kick test, five times slower, 2 s per replay
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --viz newton_gl --slow_motion 5 --steps 200

# the same without damping: the stem keeps swinging
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --viz newton_gl --slow_motion 5 --damping none

# watch the push test: the stem leans over and stays there
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test push --viz newton_gl --slow_motion 5 --steps 300

# watch the horizontal stem sag
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test sag --viz newton_gl

# physical stiffness ratios with the earlier solver setting: bending does not converge, the stem falls over
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --viz newton_gl --slow_motion 5 \
    --stretch_ratio 1 --shear_ratio 1 --damping all --iterations 20

# compare several shear ratios without a viewer (prints one RESULT line each)
for r in 1 0.1 0.01 0.001; do
    uv run --extra isaacsim python scripts/sweep_stem_solver.py --test sag --shear_ratio $r | grep RESULT
done

# computing time per step with 1024 stems
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --steps 120 --num_envs 1024
```

Why stretch and shear are softer than in a real stem, and what that costs: `docs/TODO.md`, open questions,
entry "Stem model: realistic stretch/shear stiffness ..." (with the measurement tables).
