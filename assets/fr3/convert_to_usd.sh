#!/usr/bin/env bash
# Convert build/fr3_<ee>.urdf -> build/fr3_<ee>.usd with the Isaac Lab URDF converter.
#
# Usage:  ISAACLAB_DIR=~/IsaacLab ./convert_to_usd.sh <ee>      e.g. ./convert_to_usd.sh fork
#
# The URDF importer was rewritten in Isaac Lab 3.0. Check the flags with:
#   cd $ISAACLAB_DIR && uv run --extra isaacsim python scripts/tools/convert_urdf.py --help
# Requirements for this asset:
#   - fixed base (arm is bolted to the table)
#   - do NOT merge fixed joints (otherwise the end-effector and tool_tip bodies disappear)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EE="${1:?usage: ./convert_to_usd.sh <ee>   (e.g. fork)}"
: "${ISAACLAB_DIR:?set ISAACLAB_DIR to your Isaac Lab checkout}"

URDF="$HERE/build/fr3_$EE.urdf"
USD="$HERE/build/fr3_$EE.usd"
[[ -f "$URDF" ]] || { echo "missing $URDF -> run: uv run build_asset.py --ee $EE first"; exit 1; }

cd "$ISAACLAB_DIR"
uv run --extra isaacsim python scripts/tools/convert_urdf.py "$URDF" "$USD" --fix-base

echo "wrote $USD"
