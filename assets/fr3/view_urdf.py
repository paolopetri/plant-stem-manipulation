#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["viser", "yourdfpy"]
# ///
"""Sanity check of the built URDF: print the tool values, then show the robot in the browser.

Prints the tool_tip pose and the end-effector inertial as they ended up in the URDF (compare with
end_effectors/<ee>/<ee>.yaml),
then serves a viewer at http://localhost:8080 with joint sliders, visual/collision toggles and
axes for fr3_link8 and tool_tip (red = x, green = y, blue = z).

Usage:
  uv run view_urdf.py --ee fork_v2
  uv run view_urdf.py --ee fork_v2 --port 8081
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

ROOT = Path(__file__).resolve().parent
FRAMES = ("fr3_link8", "tool_tip")


def print_check(urdf: yourdfpy.URDF, ee: str) -> None:
    np.set_printoptions(formatter={"float": lambda v: f"{v: .6g}"})
    T = urdf.get_transform("tool_tip", "fr3_link8")
    print(f"tool_tip xyz in fr3_link8: {T[:3, 3]}")
    print(f"tool_tip x axis: {T[:3, 0]}  z axis: {T[:3, 2]}")
    inertial = urdf.link_map[ee].inertial
    print(f"{ee} mass: {inertial.mass}  com: {inertial.origin[:3, 3]}")
    print(f"{ee} inertia:\n{inertial.inertia}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ee", required=True, help="end-effector name (e.g. fork); shows build/fr3_<ee>.urdf")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    path = ROOT / "build" / f"fr3_{args.ee}.urdf"
    if not path.is_file():
        raise SystemExit(f"missing {path} -> run: uv run build_asset.py --ee {args.ee}")
    # mesh paths in the URDF are relative to its folder; yourdfpy only warns about missing ones
    missing = [m.get("filename") for m in ET.parse(path).iter("mesh")
               if not (path.parent / m.get("filename")).is_file()]
    if missing:
        raise SystemExit(f"missing meshes {missing} -> run: uv run build_asset.py --ee {args.ee}")
    urdf = yourdfpy.URDF.load(str(path), mesh_dir=str(path.parent),
                              build_collision_scene_graph=True, load_collision_meshes=True)
    print_check(urdf, args.ee)

    server = viser.ViserServer(port=args.port)
    robot = ViserUrdf(server, urdf, load_collision_meshes=True)
    frames = {name: server.scene.add_frame(f"/frames/{name}", axes_length=0.05, axes_radius=0.002)
              for name in FRAMES}

    def update(cfg: np.ndarray) -> None:
        robot.update_cfg(cfg)
        for name, frame in frames.items():
            T = urdf.get_transform(name, urdf.base_link)
            frame.position = T[:3, 3]
            frame.wxyz = viser.transforms.SO3.from_matrix(T[:3, :3]).wxyz

    # start in the middle of the joint limits
    sliders = []
    with server.gui.add_folder("Joints"):
        for name, (lo, hi) in robot.get_actuated_joint_limits().items():
            sliders.append(server.gui.add_slider(name, min=lo, max=hi, step=1e-3, initial_value=(lo + hi) / 2))
    for s in sliders:
        s.on_update(lambda _: update(np.array([s.value for s in sliders])))

    with server.gui.add_folder("Display"):
        show_visual = server.gui.add_checkbox("visual", initial_value=True)
        show_collision = server.gui.add_checkbox("collision", initial_value=False)
    show_visual.on_update(lambda _: setattr(robot, "show_visual", show_visual.value))
    show_collision.on_update(lambda _: setattr(robot, "show_collision", show_collision.value))
    robot.show_collision = False

    update(np.array([s.value for s in sliders]))
    print("Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
