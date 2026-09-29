# FR3 + fork asset

Builds the FR3 robot with our fork end-effector for Isaac Lab.
The Franka URDF is never edited by hand; the fork lives in its own config.

```
fr3_fork/
├── base/fr3.urdf          FR3 without end-effector (generated, not edited)
├── fork/
│   ├── fork.yaml          all fork parameters (frame, mass, inertia, tool tip)
│   ├── fork_visual.stl
│   └── fork_collision.stl
├── build_asset.py         base + fork -> build/fr3_fork.urdf
├── view_urdf.py           sanity check of build/fr3_fork.urdf (printout + browser viewer)
├── convert_to_usd.sh      build/fr3_fork.urdf -> build/fr3_fork.usd
├── fr3_fork_cfg.py        Isaac Lab ArticulationCfg
└── build/                 generated output (git-ignored)
```

## Setup (once)

1. Check the arm type in Desk → Settings → Dashboard (Arm3R = `fr3`, Arm3Rv2 = `fr3v2`, ...).
2. Generate the arm without end-effector in `franka_description`:
   ```bash
   ./scripts/create_urdf.sh fr3 --no-ee
   ```
3. Copy the generated URDF to `base/fr3.urdf`.
4. Put `fork_visual.stl` and `fork_collision.stl` into `fork/`.

## Build

```bash
uv run build_asset.py --franka-description ~/path/to/franka_description
uv run view_urdf.py                                       # sanity check, open http://localhost:8080
ISAACLAB_DIR=~/IsaacLab ./convert_to_usd.sh
```

`view_urdf.py` prints the `tool_tip` pose and the fork mass, CoM and inertia as they ended up in the URDF
(compare with `fork/fork.yaml`). It then serves a browser viewer with joint sliders, visual/collision
toggles and axes for `fr3_link8` and `tool_tip` (red = x, green = y, blue = z). Stop it with Ctrl+C.
It runs in the browser because the yourdfpy/pyglet window only shows white on Wayland.

`--franka-description` is only needed if the base URDF uses `package://` mesh paths.
You can also set `FRANKA_DESCRIPTION` once in your shell.

## Changing the fork

1. Re-export the STLs from Onshape (origin = `fr3_link8`, meters) into `fork/`.
2. Update `fork/fork.yaml` (mass properties, tool tip).
3. Rerun the build and conversion.

After weighing the printed part, set `inertial.measured_mass`. The inertia is scaled automatically.

A different tool: copy `fork/` to e.g. `pusher/`, adapt the yaml, run `uv run build_asset.py --tool pusher/pusher.yaml`.

## Frame conventions

- All fork values are in `fr3_link8`: origin at the center of the flange face, z out of the robot.
- `tool_tip` is the frame to use in tasks (observations, rewards, IK). It sits on the lower-z edge of the
  fork in `fr3_link8` (the top of the fork in world with the flange pointing down); x points out along the
  fork, z = `fr3_link8` z. Revisit if it causes trouble in training.
