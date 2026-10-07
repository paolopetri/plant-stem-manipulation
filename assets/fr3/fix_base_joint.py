"""Make the converted FR3 a fixed-base articulation, in the converted USD.

The URDF converter (`--fix-base`) attaches `fr3_link0` with a fixed joint whose body0 is the asset's root prim, which
is not a rigid body. Isaac Lab and PhysX recognise a fixed base only by a fixed joint with a single body (body1 only,
anchored to the world); with body0 set, the FR3 counts as floating (`is_fixed_base` False). With several envs the
robots then drifted towards env 0 and exploded, and Isaac Lab's `fix_root_link` workaround added a second fixed
joint that Newton's USD import (used by Isaac Lab's tools for rendering) rejects (docs/notes/2026-10-07.md).
This script, in build/fr3_<ee>_usd/fr3_<ee>/payloads/Physics/physics.usda:
1. removes body0 from that joint, so it anchors fr3_link0 to the world;
2. moves the articulation root (PhysicsArticulationRootAPI, NewtonArticulationRootAPI and its attributes) from
   fr3_link0 to the asset's root prim. PhysX makes an articulation fixed-base only if its root API sits on the world
   joint or on an ancestor of it; on the root link itself the world joint is an extra constraint and the
   articulation stays floating. (Our stem model `chain` has the same layout.)
`convert_to_usd.sh` runs it after every conversion.

Usage (from the repo root):  uv run --extra isaacsim python assets/fr3/fix_base_joint.py fork_v2
"""

import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

ROOT = Path(__file__).resolve().parent
BASE_LINK = "fr3_link0"
ROOT_SCHEMAS = ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI")
ROOT_ATTRIBUTES = ("newton:selfCollisionEnabled",)


def main(ee: str) -> None:
    """Anchor `fr3_link0` to the world and put the articulation root on the asset's root prim."""
    path = ROOT / "build" / f"fr3_{ee}_usd" / f"fr3_{ee}" / "payloads" / "Physics" / "physics.usda"
    stage = Usd.Stage.Open(str(path))
    joints = []
    for prim in stage.TraverseAll():  # the payload layer holds "over" prims, which Traverse() skips
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body1) == 1 and body1[0].name == BASE_LINK:
            joints.append(joint)
    if len(joints) != 1:
        sys.exit(f"[fix_base_joint] ERROR: expected one fixed joint on {BASE_LINK} in {path}, found {len(joints)}")
    joint = joints[0]
    changes = []
    body0 = joint.GetBody0Rel().GetTargets()
    if body0:
        joint.GetPrim().RemoveProperty("physics:body0")
        changes.append(f"removed body0 {[str(p) for p in body0]} from {joint.GetPath()}")

    link = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath(f"/fr3_{ee}")
    # edit the apiSchemas list ops of the layer directly: the Newton schemas are not registered in this Python,
    # so Usd.Prim.GetAppliedSchemas / RemoveAppliedSchema do not see them
    layer = stage.GetRootLayer()
    link_spec, root_spec = layer.GetPrimAtPath(link.GetPath()), layer.GetPrimAtPath(root.GetPath())
    link_ops = link_spec.GetInfo("apiSchemas")
    moved = [schema for schema in ROOT_SCHEMAS if schema in link_ops.prependedItems]
    if moved or any(schema in link_ops.deletedItems for schema in ROOT_SCHEMAS):
        link_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=[s for s in link_ops.prependedItems if s not in ROOT_SCHEMAS]),
        )
        root_ops = root_spec.GetInfo("apiSchemas") if root_spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
        root_items = list(root_ops.prependedItems)
        root_spec.SetInfo(
            "apiSchemas", Sdf.TokenListOp.Create(prependedItems=root_items + [s for s in moved if s not in root_items])
        )
        changes.append(f"moved {moved} {link.GetPath()} -> {root.GetPath()}")
    for name in ROOT_ATTRIBUTES:
        attribute = link.GetAttribute(name)
        if attribute and attribute.IsAuthored():
            root.CreateAttribute(name, attribute.GetTypeName()).Set(attribute.Get())
            link.RemoveProperty(name)
            changes.append(f"moved {name}")
    if not changes:
        print(f"[fix_base_joint] {ee}: already fixed-base, nothing to do")
        return
    stage.GetRootLayer().Save()
    print(f"[fix_base_joint] {ee}: " + "; ".join(changes) + f" -> {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: fix_base_joint.py <ee>")
    main(sys.argv[1])
