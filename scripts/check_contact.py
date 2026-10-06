"""Contact check: the FR3 with `fork_v2` pushes the `chain` stem in PhysX (robot and stem in one solver).

The robot follows a straight tool-tip path at constant speed with differential IK on the body `fork_v2` +
`tool_tip_offset("fork_v2")`; the stem is moved only by contact with the fork. The fork is horizontal (flange
pointing down) at `PUSH_HEIGHT`, its slot pointing along `FORK_YAW` (45 deg between world +x and +y: the arm then
stays behind the fork; pushing along +y, the stem's top touched the wrist). Tests (`--test`):
- `slot`: the stem slides into the slot (8 mm wide at the bottom) and is pushed 5 cm beyond first contact;
- `side`: the outer face of a prong pushes the stem 5 cm sideways (perpendicular to the fork);
- `slot_far`: as `slot`, but 15 cm (large deflection; reported, not checked against the linear reference).
The robot first moves over the stem's tip to a point behind the start (`CLEAR_HEIGHT`), goes down there and then
moves horizontally to the start, so that nothing touches the stem before the push. After the push the fork holds
still, pulls back to the start and waits.

The contact force on the stem comes from Isaac Lab's `ContactSensor` on the stem segments, filtered to the fork
(a reader of PhysX's contact reports, no physical sensor). It is compared with two independent values: the force
the stem needs for its measured deflection at the push height (`stem_reference.chain_point_compliance`: bending
only, no gravity; the upright stem's own weight makes it about 4 % softer), and the bending moment in the stem's
first joint divided by its lever arm.

Checks (`slot`, `side`; `slot_far` checks only the first two and the last):
- start pose reached: tool tip within 1 mm of the start;
- only the fork touches the stem: no contact with the arm links during the whole test, and no contact at all
  before the push;
- contact where expected: first contact within 3 mm of the position where the fork touches the stem surface;
- follows the fork: stem deflection at the push height within 10 % of the fork travel after first contact
  (a fork passing through the stem would leave the stem behind);
- contact force: sensor within 10 % of the force from the stiffness and of the force from the base moment;
- stays finite, and below the curvature limit (`damage.max_curvature`);
- springs back: the stem tip is back within 1 mm after the fork has pulled back.
Reported: the stem point in the tool frame (slot depth).

Usage (from the repo root; headless unless a visualizer is requested, e.g. `--viz kit --slow_motion 3`):
    uv run --extra isaacsim python scripts/check_contact.py --test slot
    uv run --extra isaacsim python scripts/check_contact.py --test side
    uv run --extra isaacsim python scripts/check_contact.py --test slot_far --viz kit --slow_motion 3
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Contact check: the FR3 with fork_v2 pushes the chain stem.")
parser.add_argument("--test", choices=["slot", "side", "slot_far"], default="slot", help="Which push.")
parser.add_argument("--speed", type=float, default=0.05, help="Tool-tip speed during the push [m/s].")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    skew_symmetric_matrix,
)

from stem_manip.assets.fr3 import fr3_cfg, tool_tip_offset
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.utils import stem_geometry, stem_reference

PUSH_HEIGHT = 0.30  # [m] tool-tip height = height of the push on the stem
FORK_YAW = math.radians(45.0)  # direction of the fork (tool x) in the ground plane, from world +x
CLEAR_HEIGHT = 0.55  # [m] tool-tip height for the move over the stem's tip (stem length 0.4 m)
BACKOFF = 0.10  # [m] the robot goes down this far behind the start
APPROACH_SPEED = 0.10  # [m/s] mean speed of the legs to the start
PRONG_REACH = 0.17  # [m] the fork's collision mesh reaches this far beyond the tool tip along tool x
PRONG_FACE = 0.022  # [m] outer face of a prong from the slot centre line, 10 cm along the fork (collision mesh)
SIDE_CONTACT_AT = 0.10  # [m] side push: stem this far along the fork from the tool tip
APPROACH_GAP = 0.03  # [m] gap between fork and stem at the start
PUSH_DEPTH = {"slot": 0.05, "side": 0.05, "slot_far": 0.15}  # [m] fork travel beyond the stem surface
MOVE_TIME, SETTLE_TIME, HOLD_TIME, REST_TIME = 3.0, 1.0, 1.5, 3.0  # [s] (MOVE_TIME: to the point over the stem)
RENDER_DT = 1.0 / 30.0  # [s] simulated time between rendered frames with a viewer
START_TOL = 1e-3  # [m]
ONSET_TOL = 3e-3  # [m]
FOLLOW_REL_TOL = 0.10
FORCE_REL_TOL = 0.10
REST_TOL = 1e-3  # [m]
CONTACT_THRESHOLD = 1e-3  # [N] first contact


def main() -> None:
    """Run one push and print a pass/fail summary."""
    stem_module = stem_model("chain")
    params = stem_params("chain")
    geometry, material = params["geometry"], params["material"]
    num_segments, radius = geometry["num_segments"], geometry["diameter"] / 2
    segment_length = geometry["length"] / num_segments
    stem_x, stem_y = geometry["base_position"][:2]
    sim_dt = params["solver"]["sim_dt"]
    depth = PUSH_DEPTH[args_cli.test]

    # tool axes in the world (z down), push direction, start and end of the push, and where first contact is
    # expected: distance of the stem axis along the push minus the stem radius (minus the prong face, side push)
    fork_dir = (math.cos(FORK_YAW), math.sin(FORK_YAW), 0.0)  # tool x
    side_dir = (math.sin(FORK_YAW), -math.cos(FORK_YAW), 0.0)  # tool y
    if args_cli.test == "side":
        push_dir = side_dir  # the prong on the tool +y side pushes with its outer face
        contact_offset = radius + PRONG_FACE
        path_origin = (stem_x - SIDE_CONTACT_AT * fork_dir[0], stem_y - SIDE_CONTACT_AT * fork_dir[1])
        start_offset = contact_offset + APPROACH_GAP
    else:
        push_dir = fork_dir
        contact_offset = radius  # the tool tip (slot bottom) touches the stem surface
        path_origin = (stem_x, stem_y)
        start_offset = contact_offset + PRONG_REACH + APPROACH_GAP

    def on_path(distance: float) -> tuple[float, float, float]:
        """Point at `distance` along the push from the point where the path passes the stem axis."""
        return (path_origin[0] + distance * push_dir[0], path_origin[1] + distance * push_dir[1], PUSH_HEIGHT)

    stem_along = stem_x * push_dir[0] + stem_y * push_dir[1]
    expected_onset = stem_along - contact_offset
    start, end = on_path(-start_offset), on_path(depth - contact_offset)
    down = on_path(-start_offset - BACKOFF)
    over = (down[0], down[1], CLEAR_HEIGHT)

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=stem_module.physics_cfg())
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass

        robot_links = [f"fr3_link{i}" for i in range(8)]

        @configclass
        class ContactSceneCfg(InteractiveSceneCfg):
            """Ground, light, FR3 with fork_v2, chain stem, contact sensors on the stem segments."""

            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(
                prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            )
            robot = fr3_cfg("fork_v2").replace(prim_path="{ENV_REGEX_NS}/Robot")
            stem = stem_module.stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")
            fork_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Stem/seg_.*", filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*fork_v2"]
            )
            arm_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Stem/seg_.*",
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/.*{link}" for link in robot_links],
            )

        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.2, -0.9, 0.7), target=(0.45, 0.0, 0.25))
        scene = InteractiveScene(ContactSceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot, stem = scene["robot"], scene["stem"]
        fork_sensor, arm_sensor = scene["fork_contact"], scene["arm_contact"]
        device, origin = sim.device, scene.env_origins[0]
        start_t, end_t, push_dir_t, down_t, over_t = (
            torch.tensor(v, device=device) for v in (start, end, push_dir, down, over)
        )
        tool_axes = torch.tensor([fork_dir, side_dir, (0.0, 0.0, -1.0)], device=device).T  # columns: x, y, z
        tool_quat = quat_from_matrix(tool_axes.unsqueeze(0))

        # the scene starts with all joints at zero (singular, at joint limits): write the default pose
        default_joint_pos = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(position=default_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_joint_pos))
        robot.set_joint_position_target_index(target=default_joint_pos)

        # differential IK on the tool tip: Jacobian of the end-effector body, shifted to the tool tip
        ee = robot.body_names.index("fork_v2")
        jacobian_body = ee - 1 if robot.is_fixed_base else ee
        arm_joints = list(range(7))
        jacobian_columns = [robot.num_base_dofs + j for j in arm_joints]
        offset_pos, offset_quat = (torch.tensor(v, device=device).unsqueeze(0) for v in tool_tip_offset("fork_v2"))
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), 1, device
        )

        def tool_pose() -> tuple[torch.Tensor, torch.Tensor]:
            """Tool-tip position (env frame) and orientation."""
            pos, quat = robot.data.body_pos_w.torch[:, ee] - origin, robot.data.body_quat_w.torch[:, ee]
            return combine_frame_transforms(pos, quat, offset_pos, offset_quat)

        def command(target_pos: torch.Tensor, target_quat: torch.Tensor) -> None:
            pos, quat = tool_pose()
            ik.set_command(torch.cat([target_pos.view(1, 3), target_quat.view(1, 4)], dim=-1))
            jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_body, :, jacobian_columns].clone()
            lever = quat_apply(robot.data.body_quat_w.torch[:, ee], offset_pos)
            jacobian[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(lever), jacobian[:, 3:, :])
            joint_pos = ik.compute(pos, quat, jacobian, robot.data.joint_pos.torch[:, arm_joints])
            robot.set_joint_position_target_index(target=joint_pos, joint_ids=arm_joints)

        render_every = max(1, round(RENDER_DT / sim_dt))
        clock = {"steps": 0, "frame_start": time.perf_counter()}
        max_arm_contact = torch.zeros(len(robot_links), device=device)  # per link, whole test
        approach_contact = 0.0  # any contact with the robot before the push

        def step(target_pos: torch.Tensor, target_quat: torch.Tensor = tool_quat, approach: bool = False) -> None:
            nonlocal max_arm_contact, approach_contact
            command(target_pos, target_quat)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            arm_force = arm_sensor.data.force_matrix_w.torch[0].norm(dim=-1).max(dim=0).values  # per link
            max_arm_contact = torch.maximum(max_arm_contact, arm_force)
            if approach:
                fork_force = float(fork_sensor.data.force_matrix_w.torch[0].sum(dim=(0, 1)).norm())
                approach_contact = max(approach_contact, fork_force, float(arm_force.max()))
            clock["steps"] += 1
            if sim.is_rendering and clock["steps"] % render_every == 0:
                sim.render()
                frame_time = args_cli.slow_motion * render_every * sim_dt
                time.sleep(max(0.0, frame_time - (time.perf_counter() - clock["frame_start"])))
                clock["frame_start"] = time.perf_counter()

        def stem_point(poses: torch.Tensor) -> torch.Tensor:
            """Stem centre-line point at the push height (rest arc length) in the env frame."""
            segment = int(PUSH_HEIGHT // segment_length)
            offset = PUSH_HEIGHT - (segment + 0.5) * segment_length
            return stem_geometry.point_pose(poses, segment, offset)[0, :3] - origin

        rest_poses = stem_module.segment_poses(stem).clone()

        # -- move over the stem's tip to the point behind the start (orientation interpolated), go down, move
        # horizontally to the start, settle
        pos0, quat0 = (v[0].clone() for v in tool_pose())
        quat1 = tool_quat[0] if float(quat0 @ tool_quat[0]) >= 0 else -tool_quat[0]
        move_steps = round(MOVE_TIME / sim_dt)
        for i in range(move_steps):
            s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / move_steps)
            quat = quat0 + s * (quat1 - quat0)
            step(pos0 + s * (over_t - pos0), quat / quat.norm(), approach=True)
        for leg_start, leg_end in ((over_t, down_t), (down_t, start_t)):
            leg_steps = round(float((leg_end - leg_start).norm()) / APPROACH_SPEED / sim_dt)
            for i in range(leg_steps):
                s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / leg_steps)
                step(leg_start + s * (leg_end - leg_start), approach=True)
        for _ in range(round(SETTLE_TIME / sim_dt)):
            step(start_t, approach=True)
        start_error = float((tool_pose()[0][0] - start_t).norm())

        # -- push, hold, pull back, rest
        push_steps = round(float((end_t - start_t).norm()) / args_cli.speed / sim_dt)
        phases = {
            "push": (push_steps, lambda i: start_t + (i + 1) / push_steps * (end_t - start_t)),
            "hold": (round(HOLD_TIME / sim_dt), lambda i: end_t),
            "retract": (push_steps, lambda i: end_t + (i + 1) / push_steps * (start_t - end_t)),
            "rest": (round(REST_TIME / sim_dt), lambda i: start_t),
        }
        segment_lengths = torch.full((num_segments,), segment_length, device=device)
        onset, max_force, max_curvature, finite = None, 0.0, 0.0, True
        state = {}
        for phase, (steps, target) in phases.items():
            for i in range(steps):
                step(target(i))
                fork_force = fork_sensor.data.force_matrix_w.torch[0].sum(dim=(0, 1))  # on the stem from the fork
                poses = stem_module.segment_poses(stem)
                finite = finite and bool(torch.isfinite(poses).all())
                if onset is None and phase == "push" and float(fork_force.norm()) > CONTACT_THRESHOLD:
                    onset = tool_pose()[0][0].clone()
                max_force = max(max_force, float(fork_force.norm()))
                max_curvature = max(max_curvature, float(stem_geometry.joint_curvature(poses, segment_lengths).max()))
            state[phase] = {
                "tool": tool_pose()[0][0].clone(),
                "fork_force": fork_force.clone(),
                "poses": stem_module.segment_poses(stem).clone(),
                "wrenches": stem_module.joint_wrenches(stem)[0].clone(),
            }

        # -- evaluation at the end of the hold
        hold = state["hold"]
        travel = float((hold["tool"] - onset) @ push_dir_t) if onset is not None else math.nan
        onset_along = float(onset @ push_dir_t) if onset is not None else math.nan
        deflection = float((stem_point(hold["poses"]) - stem_point(rest_poses)) @ push_dir_t)
        force_sensor = float(hold["fork_force"] @ push_dir_t)
        compliance = stem_reference.chain_point_compliance(
            PUSH_HEIGHT, geometry["length"], num_segments, geometry["diameter"], material["bend_modulus"]
        )
        force_stiffness = deflection / compliance
        force_moment = float(hold["wrenches"][1, 3:].norm()) / (PUSH_HEIGHT - segment_length)  # joint 1 at height l
        in_tool = quat_apply_inverse(tool_quat, (stem_point(hold["poses"]) - hold["tool"]).unsqueeze(0))[0]
        tip_rest = float((state["rest"]["poses"][0, -1, :3] - rest_poses[0, -1, :3]).norm())
        max_curvature_limit = params["damage"]["max_curvature"]

        def close(value: float, reference: float, tol: float) -> bool:
            return abs(value - reference) <= tol * abs(reference)

        arm = {link: round(float(f), 3) for link, f in zip(robot_links, max_arm_contact) if float(f) > 0.0}
        results = {
            "start pose reached": (start_error < START_TOL, f"tool tip {start_error * 1e3:.2f} mm from the start"),
            "only the fork touches the stem": (
                not arm and approach_contact == 0.0,
                f"arm links (max force) {arm or 'none'}; max contact force before the push {approach_contact:.3f} N",
            ),
            "contact where expected": (
                onset is not None and abs(onset_along - expected_onset) < ONSET_TOL,
                f"first contact with the tool at {onset_along:.4f} m along the push (expected {expected_onset:.4f} m)",
            ),
            "follows the fork": (
                close(deflection, travel, FOLLOW_REL_TOL),
                f"stem deflection at {PUSH_HEIGHT} m {deflection * 1e3:.1f} mm, fork travel after contact "
                f"{travel * 1e3:.1f} mm",
            ),
            "contact force vs. stiffness": (
                close(force_sensor, force_stiffness, FORCE_REL_TOL),
                f"sensor {force_sensor:.3f} N, from the stem's stiffness {force_stiffness:.3f} N "
                f"(bending only, no gravity)",
            ),
            "contact force vs. base moment": (
                close(force_sensor, force_moment, FORCE_REL_TOL),
                f"sensor {force_sensor:.3f} N, from the moment in joint 1 / lever {force_moment:.3f} N",
            ),
            "finite, below the curvature limit": (
                finite and max_curvature < max_curvature_limit,
                f"all finite: {finite}, max curvature {max_curvature:.2f} 1/m (limit {max_curvature_limit} 1/m)",
            ),
            "springs back": (tip_rest < REST_TOL, f"stem tip {tip_rest * 1e3:.2f} mm from rest after pulling back"),
        }
        checked = list(results)
        if args_cli.test == "slot_far":
            checked = ["start pose reached", "only the fork touches the stem", "springs back"]

        print(f"\n=== check_contact, test {args_cli.test} ({depth * 100:.0f} cm push at {PUSH_HEIGHT} m) ===")
        for name, (ok, info) in results.items():
            label = ("PASS" if ok else "FAIL") if name in checked else "INFO"
            print(f"[{label}] {name}: {info}")
        print(f"[INFO] max contact force {max_force:.3f} N; force on the stem at the end of the hold "
              f"{[round(float(v), 3) for v in hold['fork_force']]} N (world)")
        print(f"[INFO] stem point at {PUSH_HEIGHT} m in the tool frame: along the fork {in_tool[0] * 1e3:+.1f} mm, "
              f"sideways {in_tool[1] * 1e3:+.1f} mm, vertical {in_tool[2] * 1e3:+.1f} mm")
        print("=== all passed ===" if all(results[name][0] for name in checked) else "=== FAILED ===")


if __name__ == "__main__":
    main()
