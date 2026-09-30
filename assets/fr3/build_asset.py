#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""Build a self-contained FR3 + end-effector URDF.

Inputs (never edited by this script):
  base/fr3.urdf                     FR3 arm without end-effector, generated with franka_description
  end_effectors/<ee>/<ee>.yaml      end-effector description (meshes, inertia, mount, tool tip)

Output:
  build/fr3_<ee>.urdf     merged URDF with relative mesh paths
  build/meshes/...        every mesh it references

Usage:
  uv run build_asset.py --ee fork_v2 --franka-description ~/franka_description
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EE_DIR = ROOT / "end_effectors"
TOOL_TIP = "tool_tip"  # fixed frame name, so tasks use the same body for every end-effector


def fail(msg: str) -> None:
    sys.exit(f"[build_asset] ERROR: {msg}")


def fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def add_origin(parent: ET.Element, xyz=(0, 0, 0), rpy=(0, 0, 0)) -> None:
    ET.SubElement(parent, "origin", xyz=fmt(xyz), rpy=fmt(rpy))


# --------------------------------------------------------------------------- base URDF meshes


def resolve_base_mesh(filename: str, base_dir: Path, franka_root: Path | None) -> tuple[Path, Path]:
    """Return (source file, destination relative to build/meshes) for a mesh of the base URDF."""
    if filename.startswith("package://"):
        pkg, _, rel = filename[len("package://"):].partition("/")
        if pkg != "franka_description":
            fail(f"unexpected ROS package in mesh path: {filename}")
        if franka_root is None:
            fail("base URDF uses package:// paths -> pass --franka-description <path to franka_description repo>")
        return franka_root / rel, Path("franka") / rel

    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    src = Path(filename)
    if not src.is_absolute():
        src = base_dir / src
    parts = src.parts
    rel = Path(*parts[parts.index("meshes"):]) if "meshes" in parts else Path(src.name)
    return src, Path("franka") / rel


def localize_base_meshes(robot: ET.Element, base_dir: Path, franka_root: Path | None, mesh_out: Path) -> int:
    count = 0
    for mesh in robot.iter("mesh"):
        src, dest_rel = resolve_base_mesh(mesh.get("filename", ""), base_dir, franka_root)
        if not src.is_file():
            fail(f"mesh not found: {src}")
        dest = mesh_out / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        mesh.set("filename", (Path("meshes") / dest_rel).as_posix())
        count += 1
    return count


# --------------------------------------------------------------------------- tool


def check_inertia(i: dict) -> None:
    """Inertia tensor must be positive definite (Sylvester's criterion)."""
    a, b, c = i["ixx"], i["iyy"], i["izz"]
    d, e, f = i["ixy"], i["ixz"], i["iyz"]
    m2 = a * b - d * d
    m3 = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
    if not (a > 0 and m2 > 0 and m3 > 0):
        fail(f"inertia tensor is not positive definite: {i}")


REQUIRED_KEYS = [
    "parent_link", "mount.xyz", "mount.rpy", "meshes.visual", "meshes.collision",
    "inertial.cad_mass", "inertial.com",
    *(f"inertial.inertia.{k}" for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")),
    "tool_tip.xyz", "tool_tip.rpy",
]


def check_config(cfg: dict, path: Path) -> None:
    """Fail with the list of missing keys (see README for the yaml format)."""
    missing = []
    for key in REQUIRED_KEYS:
        node = cfg
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            missing.append(key)
    if missing:
        fail(f"{path}: missing keys {missing}")


def add_tool(robot: ET.Element, name: str, cfg: dict, tool_dir: Path, mesh_out: Path) -> None:
    parent = cfg["parent_link"]
    tip = cfg["tool_tip"]

    links = {l.get("name") for l in robot.findall("link")}
    joints = {j.get("name") for j in robot.findall("joint")}
    if parent not in links:
        fail(f"parent link '{parent}' not in base URDF. Available: {sorted(links)}")
    for new in (name, TOOL_TIP):
        if new in links:
            fail(f"link '{new}' already exists in base URDF (generated with an end-effector?)")
    if f"{name}_mount" in joints:
        fail(f"joint '{name}_mount' already exists in base URDF")

    # copy tool meshes
    (mesh_out / name).mkdir(parents=True, exist_ok=True)
    mesh_paths = {}
    for kind in ("visual", "collision"):
        src = tool_dir / cfg["meshes"][kind]
        if not src.is_file():
            fail(f"{kind} mesh not found: {src}")
        shutil.copy2(src, mesh_out / name / src.name)
        mesh_paths[kind] = f"meshes/{name}/{src.name}"

    # inertia, optionally scaled to the measured mass
    inert = cfg["inertial"]
    inertia = {k: float(v) for k, v in inert["inertia"].items()}
    mass = float(inert["cad_mass"])
    if inert.get("measured_mass") is not None:
        scale = float(inert["measured_mass"]) / mass
        mass = float(inert["measured_mass"])
        inertia = {k: v * scale for k, v in inertia.items()}
        print(f"[build_asset] measured mass {mass} kg -> inertia scaled by {scale:.4f}")
    check_inertia(inertia)

    # tool link
    link = ET.SubElement(robot, "link", name=name)
    for kind in ("visual", "collision"):
        el = ET.SubElement(link, kind)
        add_origin(el)
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "mesh", filename=mesh_paths[kind])
    inertial = ET.SubElement(link, "inertial")
    add_origin(inertial, inert["com"])
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(inertial, "inertia", **{k: f"{inertia[k]:.9g}" for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")})

    # mount joint
    joint = ET.SubElement(robot, "joint", name=f"{name}_mount", type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    add_origin(joint, cfg["mount"]["xyz"], cfg["mount"]["rpy"])

    # tool tip (massless frame)
    ET.SubElement(robot, "link", name=TOOL_TIP)
    tj = ET.SubElement(robot, "joint", name=f"{TOOL_TIP}_joint", type="fixed")
    ET.SubElement(tj, "parent", link=name)
    ET.SubElement(tj, "child", link=TOOL_TIP)
    add_origin(tj, tip["xyz"], tip["rpy"])


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ee", required=True, help="end-effector name = folder in end_effectors/ (e.g. fork)")
    ap.add_argument("--base", type=Path, default=ROOT / "base" / "fr3.urdf", help="FR3 URDF without end-effector")
    ap.add_argument("--franka-description", type=Path, default=os.environ.get("FRANKA_DESCRIPTION"),
                    help="franka_description repo (needed if the base URDF uses package:// paths); "
                         "or set FRANKA_DESCRIPTION")
    ap.add_argument("--out", type=Path, default=ROOT / "build", help="output directory")
    args = ap.parse_args()

    if not args.base.is_file():
        fail(f"base URDF not found: {args.base}")
    tool_cfg = EE_DIR / args.ee / f"{args.ee}.yaml"
    if not tool_cfg.is_file():
        available = sorted(p.name for p in EE_DIR.iterdir() if p.is_dir())
        fail(f"end-effector config not found: {tool_cfg}. Available: {available}")
    franka_root = Path(args.franka_description).expanduser() if args.franka_description else None

    cfg = yaml.safe_load(tool_cfg.read_text()) or {}
    check_config(cfg, tool_cfg)
    tree = ET.parse(args.base)
    robot = tree.getroot()

    # fresh mesh folders so no stale files survive a rebuild; other end-effectors' meshes are kept
    mesh_out = args.out / "meshes"
    for sub in ("franka", args.ee):
        if (mesh_out / sub).exists():
            shutil.rmtree(mesh_out / sub)
    mesh_out.mkdir(parents=True, exist_ok=True)

    n = localize_base_meshes(robot, args.base.parent, franka_root, mesh_out)
    add_tool(robot, args.ee, cfg, tool_cfg.parent, mesh_out)

    robot.set("name", f"fr3_{args.ee}")
    out_urdf = args.out / f"fr3_{args.ee}.urdf"
    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

    print(f"[build_asset] base meshes: {n}")
    print(f"[build_asset] wrote {out_urdf}")


if __name__ == "__main__":
    main()
