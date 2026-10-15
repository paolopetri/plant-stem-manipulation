"""The FR3 asset build: a stale build is reported, and build_asset.py finds franka_description; no simulator needed."""

import importlib.util
from pathlib import Path

import pytest

from stem_manip.assets import REPO_ASSETS_DIR, fr3

YAML = """slot_base:
  xyz: [0.01, 0.0, 0.2]
  rpy: [0, 0.5, 0]
"""


def _built_urdf(slot_base_link: str = "slot_base", xyz: str = "0.01 0 0.2", rpy: str = "0 0.5 0") -> str:
    """Built URDF with only the joint of interest: as build_asset.py writes it (9 significant digits)."""
    return f"""<?xml version='1.0' encoding='utf-8'?>
<robot name="fr3_pusher">
  <joint name="{slot_base_link}_joint" type="fixed">
    <parent link="pusher" />
    <child link="{slot_base_link}" />
    <origin xyz="{xyz}" rpy="{rpy}" />
  </joint>
</robot>
"""


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """End-effector `pusher` with YAML in a temporary assets folder; returns the path of its built URDF."""
    (tmp_path / "end_effectors" / "pusher").mkdir(parents=True)
    (tmp_path / "end_effectors" / "pusher" / "pusher.yaml").write_text(YAML)
    (tmp_path / "build").mkdir()
    monkeypatch.setattr(fr3, "EE_DIR", tmp_path / "end_effectors")
    monkeypatch.setattr(fr3, "BUILD_DIR", tmp_path / "build")
    return tmp_path / "build" / "fr3_pusher.urdf"


def test_build_matching_the_yaml_is_not_stale(assets):
    assets.write_text(_built_urdf(xyz="0.01 0 0.200000000", rpy="0 0.5 0"))
    assert fr3.stale_build("pusher") is None


@pytest.mark.parametrize(
    ("xyz", "rpy"),
    [("0.01 0 0.25", "0 0.5 0"), ("0.01 0 0.2", "0 0.5 0.1")],
    ids=["slot base moved", "slot base turned"],
)
def test_build_with_another_slot_base_pose_is_stale(assets, xyz, rpy):
    assets.write_text(_built_urdf(xyz=xyz, rpy=rpy))
    assert "slot_base" in fr3.stale_build("pusher")


def test_build_without_a_slot_base_frame_is_stale(assets):
    """A built URDF without a slot_base frame (e.g. an old build with the frame under another name)."""
    assets.write_text(_built_urdf(slot_base_link="tool_tip"))  # glossary: allow
    assert "slot_base" in fr3.stale_build("pusher")


def _load_build_asset():
    """assets/fr3/build_asset.py is a standalone script, not part of the package: load it from its path."""
    spec = importlib.util.spec_from_file_location("build_asset", REPO_ASSETS_DIR / "fr3" / "build_asset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sibling_franka_description_is_next_to_the_repo():
    """Same convention as ../IsaacLab: the folder next to the repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    assert repo_root.parent / "franka_description" == _load_build_asset().SIBLING_FRANKA_DESCRIPTION


@pytest.fixture
def build_asset(tmp_path, monkeypatch):
    """build_asset.py with FRANKA_DESCRIPTION unset and its sibling franka_description in tmp_path (not created)."""
    module = _load_build_asset()
    monkeypatch.setattr(module, "SIBLING_FRANKA_DESCRIPTION", tmp_path / "franka_description")
    monkeypatch.delenv("FRANKA_DESCRIPTION", raising=False)
    return module


def test_franka_description_variable_wins_over_the_sibling(build_asset, tmp_path, monkeypatch):
    build_asset.SIBLING_FRANKA_DESCRIPTION.mkdir()
    monkeypatch.setenv("FRANKA_DESCRIPTION", str(tmp_path / "from_variable"))
    assert build_asset.default_franka_description() == tmp_path / "from_variable"


def test_sibling_franka_description_is_used_when_the_variable_is_unset(build_asset, tmp_path):
    build_asset.SIBLING_FRANKA_DESCRIPTION.mkdir()
    assert build_asset.default_franka_description() == tmp_path / "franka_description"


def test_no_franka_description_without_variable_and_sibling(build_asset):
    assert build_asset.default_franka_description() is None
