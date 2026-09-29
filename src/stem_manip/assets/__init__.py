"""Isaac Lab configs for the project assets (robot, stem).

The asset source data and build pipelines live in the repo-level `assets/` folder; this package only
holds the Python configs that point to it. Assumes an editable install (`uv sync`), so the repo layout
is available at runtime.
"""

from pathlib import Path

REPO_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
"""Repo-level `assets/` folder (source data, build pipelines, generated USD)."""
