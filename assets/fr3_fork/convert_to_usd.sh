#!/usr/bin/env bash
# Convert build/fr3_fork.urdf -> build/fr3_fork.usd with the Isaac Lab URDF converter.
#
# Usage:  ISAACLAB_DIR=~/IsaacLab ./convert_to_usd.sh
#
# The URDF importer was rewritten in Isaac Lab 3.0. Check the flags with:
#   cd $ISAACLAB_DIR && uv run --extra isaacsim python scripts/tools/convert_urdf.py --help
# Requirements for this asset:
#   - fixed base (arm is bolted to the table)
#   - do NOT merge fixed joints (otherwise the fork and tool_tip bodies disappear)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ISAACLAB_DIR:?set ISAACLAB_DIR to your Isaac Lab checkout}"

URDF="$HERE/build/fr3_fork.urdf"
USD="$HERE/build/fr3_fork.usd"
[[ -f "$URDF" ]] || { echo "missing $URDF -> run: uv run build_asset.py first"; exit 1; }

cd "$ISAACLAB_DIR"
uv run --extra isaacsim python scripts/tools/convert_urdf.py "$URDF" "$USD" --fix-base

echo "wrote $USD"
