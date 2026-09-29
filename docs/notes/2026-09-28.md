# 2026-09-28 — Fork gripper → FR3 model

## Decisions
- Use the **FR3** URDF from `franka_description` (not the Isaac Lab Panda): same kinematics, but different limits and masses.
- Add the gripper in the **URDF** (fixed joint to `fr3_link8`), then convert URDF → USD for Isaac Lab.
- CAD in **Onshape** (browser, Education plan).

## Frame
Everything is expressed in the flange frame **`fr3_link8`**, so the URDF mount joint is `xyz="0 0 0" rpy="0 0 0"`.
- Origin: center of the flange face = `link7` frame **+107 mm in z**
- z: out of the robot, into the gripper
- x/y: same as `link7`; the gripper's rotation about z is set by its bolt holes / dowel pin

## Steps
1. `link7.dae` (fr3, visual) → STL:
   `uv run --with trimesh --with pycollada python -c 'import trimesh; trimesh.load("link7.dae", force="mesh").export("link7.stl")'`
2. Onshape: import `link7.stl` (meters), mate connector `link8` at z = 107 mm.
3. Mate connector `tool_mount` on the gripper mounting face (z into gripper, x toward dowel pin).
4. Transform `tool_mount` → `link8`, check that the holes line up (rotate about z if not).
5. Transform `tool_mount` → origin. Origin is now `fr3_link8`.
6. Read off: mass, CoM, inertia (about CoM), `tool_tip` position.
7. Export visual + simplified collision STL (meters).

## Next
- Weigh the printed gripper
- Generate FR3 URDF without end-effector, add gripper + `tool_tip`, convert to USD