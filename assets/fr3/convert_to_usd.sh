#!/usr/bin/env bash
# Convert build/fr3_<ee>.urdf -> build/fr3_<ee>_usd/fr3_<ee>/fr3_<ee>.usda with the Isaac Lab URDF converter.
# The converter writes a folder (stage + payloads); the file name comes from the robot name in the URDF.
#
# Usage:  ISAACLAB_DIR=~/IsaacLab ./convert_to_usd.sh <ee>      e.g. ./convert_to_usd.sh fork
#
# The URDF importer was rewritten in Isaac Lab 3.0. Check the flags with:
#   cd $ISAACLAB_DIR && uv run --extra isaacsim python scripts/tools/convert_urdf.py --help
# Requirements for this asset:
#   - fixed base (arm is bolted to the table)
#   - do NOT merge fixed joints (otherwise the end-effector body is merged into fr3_link7)
# Massless links without geometry (fr3_link8, tool_tip) become plain frames, not bodies.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EE="${1:?usage: ./convert_to_usd.sh <ee>   (e.g. fork)}"
: "${ISAACLAB_DIR:?set ISAACLAB_DIR to your Isaac Lab checkout}"

URDF="$HERE/build/fr3_$EE.urdf"
USD_DIR="$HERE/build/fr3_${EE}_usd"
USD="$USD_DIR/fr3_$EE/fr3_$EE.usda"
[[ -f "$URDF" ]] || { echo "missing $URDF -> run: uv run build_asset.py --ee $EE first"; exit 1; }

cd "$ISAACLAB_DIR"
uv run --extra isaacsim python scripts/tools/convert_urdf.py "$URDF" "$USD_DIR" --fix-base

[[ -f "$USD" ]] || { echo "conversion finished but $USD is missing"; exit 1; }
echo "wrote $USD"
