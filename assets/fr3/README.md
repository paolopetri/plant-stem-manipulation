# FR3 + end-effector assets

Builds the FR3 robot with one of our end-effectors (e.g. the fork) for Isaac Lab.
The Franka URDF is never edited by hand; each end-effector lives in its own folder with a yaml config.

```
fr3/
├── base/fr3.urdf               FR3 without end-effector (generated, not edited)
├── end_effectors/
│   └── fork/                   one folder per end-effector, folder name = <ee>
│       ├── fork.yaml           all parameters (mount, mass, inertia, tool tip)
│       ├── fork_visual.stl
│       └── fork_collision.stl
│   (end-effectors so far: fork, fork_v2 = longer prongs)
├── build_asset.py              base + <ee> -> build/fr3_<ee>.urdf
├── view_urdf.py                sanity check of build/fr3_<ee>.urdf (printout + browser viewer)
├── convert_to_usd.sh           build/fr3_<ee>.urdf -> build/fr3_<ee>_usd/ (USD stage + payloads)
└── build/                      generated output (git-ignored)
```

## Setup (once)

1. Check the arm type in Desk → Settings → Dashboard (Arm3R = `fr3`, Arm3Rv2 = `fr3v2`, ...).
2. Generate the arm without end-effector in `franka_description`:
   ```bash
   ./scripts/create_urdf.sh fr3 --no-ee
   ```
3. Copy the generated URDF to `base/fr3.urdf`.
4. Tell the build where `franka_description` is (the base URDF uses `package://` mesh paths). Either pass
   `--franka-description <path>` every time or set it once in your shell:
   ```bash
   export FRANKA_DESCRIPTION=/path/to/franka_description
   ```

## Build and check

Replace `fork` with the end-effector you want. All commands run from this folder.

```bash
uv run build_asset.py --ee fork                     # -> build/fr3_fork.urdf
uv run view_urdf.py --ee fork                       # sanity check, open http://localhost:8080
ISAACLAB_DIR=~/IsaacLab ./convert_to_usd.sh fork    # -> build/fr3_fork_usd/fr3_fork/fr3_fork.usda
```

`view_urdf.py` prints the `tool_tip` pose and the end-effector's mass, CoM and inertia as they ended up in
the URDF (compare with the yaml). It then serves a browser viewer with joint sliders, visual/collision
toggles and axes for `fr3_link8` and `tool_tip` (red = x, green = y, blue = z). Stop it with Ctrl+C.
It runs in the browser because the yourdfpy/pyglet window only shows white on Wayland.

In Isaac Lab, use `fr3_cfg("fork")` from `stem_manip.assets.fr3` (`src/stem_manip/assets/fr3.py`). The articulation's bodies are `fr3_link0` … `fr3_link7`
and the end-effector (`fork`). `tool_tip` and `fr3_link8` are **not** bodies: the converter turns massless links
without geometry into plain frames. For the task frame, use the end-effector body plus the offset from the yaml:

```python
from stem_manip.assets.fr3 import fr3_cfg, tool_tip_offset
pos, rot = tool_tip_offset("fork")   # rot as quaternion (x, y, z, w), the Isaac Lab 3.0 order
# e.g. body_name="fork", body_offset=OffsetCfg(pos=pos, rot=rot) in the IK action, or in a FrameTransformer
```

## Adding an end-effector

Create `end_effectors/<ee>/` with three files. `<ee>` becomes the link name in the URDF and USD, so use
lowercase letters, digits and `_`, starting with a letter (e.g. `pusher`, `fork_v2`).

| File | What |
|---|---|
| `<ee>.yaml` | config, format below |
| visual STL | detailed mesh, used for rendering |
| collision STL | simplified mesh with few triangles, used for contact physics |

**Both STLs** must be in **meters** and expressed in the end-effector frame. The easiest way is to put the
CAD origin at `fr3_link8` (center of the flange face, z pointing out of the robot, see
`docs/notes/2026-09-28.md`) and keep the mount at identity. The filenames are free; the yaml points to them.

**The yaml** (copy this, all keys are required; the build lists any missing ones):

```yaml
parent_link: fr3_link8          # FR3 link the end-effector is bolted to

mount:                          # pose of the end-effector frame in parent_link
  xyz: [0, 0, 0]                # m;   identity if the CAD origin is fr3_link8
  rpy: [0, 0, 0]                # rad; roll, pitch, yaw

meshes:                         # paths relative to this folder
  visual: pusher_visual.stl
  collision: pusher_collision.stl

inertial:                       # all in the end-effector frame, SI units
  cad_mass: 0.05                # kg, from CAD
  measured_mass: null           # kg after weighing the print; inertia is then scaled by measured/cad
  com: [0.0, 0.0, 0.02]         # m, center of mass
  inertia:                      # kg·m², about the CoM (not about the origin), axes = end-effector frame
    ixx: 1.0e-5
    iyy: 1.0e-5
    izz: 1.0e-5
    ixy: 0.0
    ixz: 0.0
    iyz: 0.0

tool_tip:                       # task frame (observations, rewards, IK), in the end-effector frame
  xyz: [0.0, 0.0, 0.05]         # m
  rpy: [0, 0, 0]                # rad
```

The build checks that the inertia tensor is physically valid (positive definite).

**Why mass, CoM and inertia are all required:** if the inertia is missing, PhysX computes it from the convex
approximation of the collision mesh with uniform density. That's plausible, but wrong for printed parts
(infill, the gaps a convex hull fills in). Take the values from CAD (Onshape → Mass properties, inertia about
the CoM) and correct the scale with `measured_mass` after weighing.

## Changing an end-effector

1. Re-export the STLs into `end_effectors/<ee>/`.
2. Update `<ee>.yaml` (mass properties, tool tip).
3. Rerun the build, check and conversion.

After weighing the printed part, set `inertial.measured_mass`. The inertia is scaled automatically.

## Frame conventions

- `fr3_link8`: origin at the center of the flange face, z out of the robot.
- `tool_tip` is the frame to use in tasks (observations, rewards, IK). It has the same name for every
  end-effector, so tasks work unchanged when the end-effector is swapped. In Isaac Lab it is reached as
  end-effector body + `tool_tip_offset(ee)` (see above).
- Fork: `tool_tip` sits on the lower-z edge of the fork in `fr3_link8` (the top of the fork in world with the
  flange pointing down); x points out along the fork, z = `fr3_link8` z. Revisit if it causes trouble in
  training.
