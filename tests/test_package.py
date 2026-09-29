"""Package-level checks that run without the simulator (M0 reproducibility).

Covers: the package imports, Isaac Lab finds our tasks through the entry point, and the FR3 asset config
resolves its files and the tool_tip offset from the end-effector yaml.
"""

import math
from importlib.metadata import entry_points

import numpy as np
import yaml

from stem_manip.assets import REPO_ASSETS_DIR
from stem_manip.assets.fr3 import BUILD_DIR, EE_DIR, fr3_cfg, tool_tip_offset


def test_tasks_package_imports():
    """Importing the tasks package (what Isaac Lab does via the entry point) must not fail."""
    import stem_manip.tasks  # noqa: F401


def test_isaaclab_entry_point():
    """Isaac Lab discovers our tasks through the `isaaclab.tasks` entry point in pyproject.toml."""
    eps = {ep.name: ep.value for ep in entry_points(group="isaaclab.tasks")}
    assert eps.get("stem_manip") == "stem_manip.tasks"


def test_asset_paths_point_into_repo():
    """The configs in src/ must find the data in the repo-level assets/ folder."""
    assert REPO_ASSETS_DIR.is_dir()
    assert EE_DIR.is_dir()
    assert BUILD_DIR.parent == REPO_ASSETS_DIR / "fr3"


def test_fr3_cfg_usd_path():
    """fr3_cfg("fork") points to the USD produced by convert_to_usd.sh."""
    usd_path = fr3_cfg("fork").spawn.usd_path
    assert usd_path.endswith("fr3_fork_usd/fr3_fork/fr3_fork.usda")


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: fixed-axis roll, pitch, yaw -> R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quat_xyzw_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_tool_tip_offset_matches_yaml():
    """tool_tip_offset returns the yaml position and a quaternion (x, y, z, w) equal to the yaml rpy."""
    tip = yaml.safe_load((EE_DIR / "fork" / "fork.yaml").read_text())["tool_tip"]
    pos, rot = tool_tip_offset("fork")

    np.testing.assert_allclose(pos, tip["xyz"], atol=1e-9)
    assert math.isclose(float(np.linalg.norm(rot)), 1.0, abs_tol=1e-9)
    # Compare as rotation matrices, computed independently from the rpy and from the quaternion.
    np.testing.assert_allclose(_quat_xyzw_to_matrix(rot), _rpy_to_matrix(*tip["rpy"]), atol=1e-9)
