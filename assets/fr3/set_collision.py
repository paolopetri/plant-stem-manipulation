"""Set the PhysX collision shape of the end-effector's collision mesh in the converted USD.

The URDF converter approximates every collision mesh by its convex hull. For an end-effector whose yaml has a
`collision_approximation` section, this script replaces the approximation of its collision mesh (only that one)
in build/fr3_<ee>_usd/fr3_<ee>/payloads/instances.usda; without the section nothing changes. `convert_to_usd.sh`
runs it after every conversion.

Usage (from the repo root):  uv run python assets/fr3/set_collision.py fork_v2
"""

import sys
from pathlib import Path

import yaml
from pxr import Sdf, Usd

ROOT = Path(__file__).resolve().parent

# yaml key -> (attribute of PhysxConvexDecompositionCollisionAPI, type)
DECOMPOSITION_ATTRIBUTES = {
    "max_convex_hulls": ("maxConvexHulls", Sdf.ValueTypeNames.Int),
    "voxel_resolution": ("voxelResolution", Sdf.ValueTypeNames.Int),
    "error_percentage": ("errorPercentage", Sdf.ValueTypeNames.Float),
    "shrink_wrap": ("shrinkWrap", Sdf.ValueTypeNames.Bool),
    "hull_vertex_limit": ("hullVertexLimit", Sdf.ValueTypeNames.Int),
    "min_thickness": ("minThickness", Sdf.ValueTypeNames.Float),
}


def main(ee: str) -> None:
    """Apply `collision_approximation` from end_effectors/<ee>/<ee>.yaml to the converted USD."""
    cfg = yaml.safe_load((ROOT / "end_effectors" / ee / f"{ee}.yaml").read_text()).get("collision_approximation")
    if not cfg:
        print(f"[set_collision] {ee}: no collision_approximation in the yaml, convex hull kept")
        return
    if cfg["type"] != "convexDecomposition":
        sys.exit(f"[set_collision] ERROR: unsupported type {cfg['type']!r} (supported: convexDecomposition)")
    unknown = set(cfg) - {"type", *DECOMPOSITION_ATTRIBUTES}
    if unknown:
        sys.exit(f"[set_collision] ERROR: unknown keys {sorted(unknown)}")

    path = ROOT / "build" / f"fr3_{ee}_usd" / f"fr3_{ee}" / "payloads" / "instances.usda"
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath(f"/Instances/{ee}_collision/{ee}_collision")
    if not prim or not prim.HasAttribute("physics:approximation"):
        sys.exit(f"[set_collision] ERROR: no collision mesh of {ee} in {path}")
    prim.GetAttribute("physics:approximation").Set("convexDecomposition")
    prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
    for key, (name, value_type) in DECOMPOSITION_ATTRIBUTES.items():
        if key in cfg:
            prim.CreateAttribute(f"physxConvexDecompositionCollision:{name}", value_type).Set(cfg[key])
    stage.GetRootLayer().Save()
    print(f"[set_collision] {ee}: convexDecomposition {({k: v for k, v in cfg.items() if k != 'type'})} -> {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: set_collision.py <ee>")
    main(sys.argv[1])
