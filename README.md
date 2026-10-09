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
uv run --extra isaacsim python scripts/check_stem.py    # stem (--stem_model, default cable): pass/fail check
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick   # stem: measure one test (see below)
uv run --extra isaacsim python scripts/check_contact.py --test slot       # FR3 + fork push the chain stem (see below)
uv run --extra isaacsim python scripts/check_push_env.py  # push task: controller and action limits (see below)
uv run --extra isaacsim python scripts/sweep_action_poses.py  # push task: following over the workspace (see below)
uv run --extra isaacsim python scripts/check_push_obs.py  # push task: target command and stem observations (see below)
uv run --extra isaacsim python scripts/check_push_terms.py  # push task: rewards and terminations (see below)
uv run --extra isaacsim python scripts/check_push_agents.py  # push task: zero / random agent, shapes and ranges (see below)
uv run --extra isaacsim isaaclab zero_agent --task StemManip-Push-Position-FR3-v0 --num_envs 4 --max_steps 700
uv run --extra isaacsim isaaclab random_agent --task StemManip-Push-Position-FR3-v0 --num_envs 4 --max_steps 700
```

Simulation scripts run without a window unless a viewer is requested with `--viz newton_gl`.
The zero / random agents must be started through the `isaaclab` CLI as above: Isaac Lab's
`scripts/environments/zero_agent.py` run directly does not register this project's tasks.
Training commands are added with the first training run (docs/TODO.md, M5).

## Push task: controller checks

`StemManip-Push-Position-FR3-v0` (stage 1). The policy's action is a tool-tip translation and rotation step,
executed by Franka's Cartesian impedance law with speed caps (10 cm/s, 45 deg/s), acceleration limits (0.08 m/s^2,
20 deg/s^2) and a target clamp (4 mm / 3 deg, at every physics step). Values and reasons:
`docs/overleaf_folder/open_questions/action_limits_problem.tex`. Both scripts test the controller without the stem:
the fork never touches it (the spawn area is switched off and the stem stands out of reach behind the robot; the
clamp test pushes the fork into the ground instead).

```bash
uv run --extra isaacsim python scripts/check_push_env.py                                       # pass/fail
uv run --extra isaacsim python scripts/check_push_env.py --num_envs 1 --viz kit --slow_motion 3  # watch it
uv run --extra isaacsim python scripts/sweep_action_poses.py                # 17 start points, short moves + reversals
uv run --extra isaacsim python scripts/sweep_action_poses.py --full_speed   # one move per axis up to the caps
```

| Check (`check_push_env.py`) | Passes if |
|---|---|
| shapes | observation (n, 57), action (n, 6), all finite |
| start pose, holds still | tool tip within 5 mm / 1 deg of the start pose, moves < 1 mm while holding |
| follows the actions / the rotation | lag <= 2 mm (2 deg), overshoot <= 5 mm (2 deg), tool-tip drift <= 2 mm |
| step limits | from rest the first step is 0.08 mm / 0.02 deg, then exactly the caps 3.2 mm / 1.44 deg |
| follows at full speed | lag <= 2 mm at 10 cm/s, angle lag <= 2 deg and drift <= 2 mm at 45 deg/s |
| target clamp | fork pushed into the ground: target-tool distance <= 4 mm at every physics step, exactly 4 mm while blocked |
| restart after contact | lifting off, the step changes by at most 0.08 mm per policy step |
| applied step within the caps | in every phase |

The expected limits are the decided values, not read from the cfg: with an override flag (`--max_step`, ...) those
checks fail on purpose. A check that comes within 0.25 rad of a joint limit fails as invalid. The sweep uses the same
criteria; start points (default mode) or moves (`--full_speed`) within 0.25 rad of a joint limit are reported but
not counted.

## Push task: target and stem observations

At every reset the stem base is moved to a random place in the spawn area, x 0.50-0.65 m, y +-0.15 m from the robot
base (on the ground, upright, at rest); the target is sampled around the new base.
The target is a position for the stem's tip, sampled once per episode 3-10 cm sideways from its rest position, at the
height of a pushed tip or below it (C / S bends that need the fork's slot, up to 0.8 x the curvature limit). The policy observes 57 values (robot base frame, meters): tool tip 21, stem
base 3, 5 points along the stem (0.08 ... 0.40 m) for the previous and the current policy step 30, target 3.

```bash
uv run --extra isaacsim python scripts/check_push_obs.py                                       # pass/fail
uv run --extra isaacsim python scripts/check_push_obs.py --num_envs 4 --viz kit --slow_motion 3  # watch it
```

In the viewer (4 or more envs show different stem places): the target is a red sphere (green within 1 cm of the tip), the tip and the 5 observed points are
blue spheres; the stem is pushed sideways (+x) by a force on its tip, then released.

| Check (`check_push_obs.py`) | Passes if |
|---|---|
| shapes | observation (n, 57), all finite |
| spawn area | every observed base in x 0.50-0.65 m, y +-0.15 m, z 0, at reset and after a second reset; equal to the stem root (0.01 mm); different between envs; moved by the reset; constant in the episode (1 mm) |
| stem at rest | the 5 points on the upright stem above the observed base, within 1 mm |
| observed tip = tip from segment poses | within 0.01 mm (frame and point helper) |
| previous step | the previous-step block equals the step before |
| pushed stem | every point moves in +x, higher points more, the tip > 1 cm, sideways < 1 mm |
| target in the region | 3-10 cm from the rest tip, on or below the bowl within the curvature budget, fixed in the episode, new after a reset and again in the region around the new base |

## Push task: rewards and terminations

Rewards: stem tip to target (`1 - tanh(d / std)`, coarse 5 cm and fine 1 cm), height error alone (std 3 mm, weight 0
for now), fork-to-stem approach (std 0.1 m), curvature penalty above 0.8 x the limit, contact-force penalty above 2 N
(largest single contact of the robot on the stem, mean over a policy step), action rate, early-termination penalty.
Terminations: time out, curvature above `damage.max_curvature`, contact force above `damage.max_contact_force`
(both in `assets/stem/stem.yaml`), any FR3 joint within 0.25 rad of its limit. Weights and thresholds: `env_cfg.py`.
The check records every term when the managers evaluate it and compares it with values computed independently.

```bash
uv run --extra isaacsim python scripts/check_push_terms.py                                          # pass/fail
uv run --extra isaacsim python scripts/check_push_terms.py --phases push --viz kit --slow_motion 1  # watch a push
```

| Check (`check_push_terms.py`) | Passes if |
|---|---|
| rest / time out | zero action: the time out fires exactly at the episode's last step, nothing else before; penalties, contact force and action rate 0 |
| action rate | equals sum((a_t - a_t-1)^2) of the actions sent |
| push: approach | the scripted drive behind the stem reaches its goal (2 mm) without a reset; approach rises, > 0.9 in contact |
| push: contact | slow 4 cm push in the slot: force > 0.05 N and < 2 N, penalty 0, no contact before |
| push: contact mechanism | with lowered thresholds: penalty = F^2 at 0 N, the limit fires in every env |
| impact | full-speed slot impact (informational: largest mean force; fails only if the fork does not touch the stem) |
| curvature | tip-force ramp: penalty > 0 only above 4 1/m, the limit fires above 5 1/m in every env, nothing earlier |
| joint margin | moving straight down from the start pose fires it in every env |
| no unexpected reset | no reset outside the intended firings (the scripted phases run without a time out) |
| every step: terms = formulas | every reward and termination matches its formula of the independent values, `height_error` = dz |

## Push task: zero / random agent

The task run as Isaac Lab's `zero_agent` / `random_agent` do (zero actions; uniform random actions in [-1, 1],
seeded), 2 episodes each, every policy step checked against the decided values. Fixed step counts, a progress line
every 100 steps, and a failure after `--max_minutes` (default 15; checked between policy steps, so a hang inside a
step is not caught: run it under `timeout 20m ...` when unattended). About 3 min with 4 envs.

```bash
uv run --extra isaacsim python scripts/check_push_agents.py                                          # pass/fail
uv run --extra isaacsim python scripts/check_push_agents.py --agents random --num_envs 16            # more envs
uv run --extra isaacsim python scripts/check_push_agents.py --agents random --num_envs 4 --viz kit --slow_motion 2  # watch it
```

| Check (`check_push_agents.py`) | Passes if |
|---|---|
| shapes | observation (num_envs, 57), action (num_envs, 6), term sizes 3 / 6 / 6 / 6 / 3 / 30 / 3 |
| finite | observations, rewards and done flags finite at every step |
| start pose after reset | tool tip within 5 mm of (0.40, 0, 0.50) m and 1 deg (rotation angle) of the first reset's orientation |
| stem base in the spawn area | x 0.50-0.65 m, y +-0.15 m, z 0 after every reset |
| target_offset / applied_step | each part's norm <= 1 (clamp 4 mm / 3 deg; speed caps, except where the clamp holds the target) |
| stem points | each point no farther from the base than its arc length (+1 mm) |
| constant within an episode | stem base (1e-6 m, see below) and target |
| reward ranges | distance / approach in [0, weight], penalties <= 0, height 0 (only confirms weight 0: Isaac Lab skips the term) |
| reset only with a done flag | the episode length counts up by one or restarts after a done flag |
| tool_tip_rot | both columns unit length and orthogonal (1e-4) |
| zero agent | only the time out fires, exactly at step 469; the tool tip holds (2 mm); penalties 0 |

The random agent's terminations are reported, not checked. Observations in the robot base frame jump by up to
5e-8 m on the first physics step after a reset: the robot root pose PhysX returns then differs from the written one
(about 1e-8 in the quaternion, up to one float32 rounding step in position); hence the 1e-6 m tolerance.
The Isaac Lab agents in the command list need `--max_steps`: without it they run until the process is killed.

## Stem tests

The stem is selected by name (`--stem_model`, default `cable`): `cable` (Newton cable) or `chain` (rigid segments
with spring joints, PhysX). The plant values shared by all stem models are in
`assets/stem/stem.yaml`; each model adds its own values (segment count, solver settings, ...) in
`assets/stem/<model>/<model>.yaml`. Two scripts test them; `sweep_stem_solver.py` is for the `cable` model only.

### `check_stem.py`: does the stem work? (pass/fail)

```bash
uv run --extra isaacsim python scripts/check_stem.py --stem_model cable
uv run --extra isaacsim python scripts/check_stem.py --stem_model chain
uv run --extra isaacsim python scripts/check_stem.py --stem_model chain --viz kit --slow_motion 5        # watch it
uv run --extra isaacsim python scripts/check_stem.py --stem_model cable --viz newton_gl --slow_motion 5  # watch it
```

Spawns two stems per environment from the yaml, both clamped at the base: one upright, whose tip is first
kicked sideways and later pushed with a steady force, and one horizontal, which sags under its own weight. Prints one `[PASS]` / `[FAIL]` line per check:

| Check | Passes if |
|---|---|
| segments, start poses, mass | the simulated stem has the yaml's segment count, shape and mass |
| joint stiffness, joint damping (chain: also joint armature) | the solver's per-joint values equal those computed from the yaml |
| base fixed | the clamped segment does not move at all |
| deflects when kicked | the tip moves by more than 1 cm and all values stay finite |
| springs back | the tip is back within 1 mm of upright at the end |
| swing frequency | within 15 % of the first bending frequency of a clamped beam |
| damping ratio | within 0.02 of the value set by `damping_time` |
| deflects when pushed | a steady sideways force of 0.05 N on the last segment deflects it within 10 % of the value computed by hand |
| returns after the push | the pushed point is back within 1 mm after the force is removed |
| cantilever sag | the horizontal stem's tip drop is within 5 % of the value computed by hand |

For the cable it also prints the axial strain of the upright stem at rest against the hand value (`[INFO]` lines, not a check:
at the current solver cost the stretch direction is not converged, see `docs/TODO.md`).

Run it after every change to the stem or the yaml.

| Option | Default | Meaning |
|---|---|---|
| `--num_envs N` | 2 | number of environments (each has one upright and one horizontal stem) |
| `--env_spacing D` | 1.0 | distance between neighbouring environments [m]; 0 stacks all at the world origin |
| `--stem_model M` | `cable` | stem model to check (`cable`, `chain`) |
| `--kick_time T` | 4.0 | simulated time after the kick [s] |
| `--viz newton_gl` / `--viz kit` | off | open a viewer window: `newton_gl` for both models, `kit` (Isaac Sim's window) for the chain (PhysX) only. Kit's very first start takes several minutes without output (window "not responding"); later starts take seconds |
| `--slow_motion S` | 1.0 | with a viewer: play S times slower than real time |

### `check_contact.py`: does the fork push the stem correctly? (pass/fail)

The FR3 with `fork_v2` pushes the `chain` stem (PhysX, robot and stem in one solver). The robot follows a straight
tool-tip path with differential IK; the stem moves only through contact. The fork points at 45 degrees between
world +x and +y, so that the arm stays behind it, and it reaches the start over the stem's tip and from behind, so
that nothing touches the stem before the push. The contact force is read with Isaac Lab's `ContactSensor` and
compared with the force the stem needs for its measured deflection and with the bending moment in its base joint.

```bash
uv run --extra isaacsim python scripts/check_contact.py --test slot                              # 5 cm, stem in the slot
uv run --extra isaacsim python scripts/check_contact.py --test side                              # 5 cm, outer face of a prong
uv run --extra isaacsim python scripts/check_contact.py --test slot_far --viz kit --slow_motion 3  # 15 cm, watch it
```

| Check | Passes if |
|---|---|
| start pose reached | the tool tip is within 1 mm of the start |
| only the fork touches the stem | no contact with the arm links at all, and no contact before the push |
| contact where expected | first contact within 3 mm of where the fork touches the stem surface |
| follows the fork | stem deflection at the push height within 10 % of the fork travel after contact |
| contact force vs. stiffness / vs. base moment | the sensor force within 10 % of each calculated force |
| finite, below the curvature limit | no NaN, maximum curvature below `damage.max_curvature` |
| springs back | the stem tip is back within 1 mm after the fork has pulled back |

`slot_far` checks only the start pose, the contacts and the spring-back and reports the rest (large deflection, the
linear reference does not apply).

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
