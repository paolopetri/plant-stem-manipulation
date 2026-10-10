"""Upload a run's newest play video to its wandb run (training itself runs without cameras: rendering doubles the
iteration time, 5.3 -> 10.5 s at 4096 envs, 2026-10-10).

Usage: uv run python tools/autotrain/upload_video.py <train output file> <run dir>
The wandb run id is read from the train output ("run id '<id>'"); the video from <run dir>/videos/play/.
"""

import re
import sys
from pathlib import Path

import wandb

PROJECT = "stem-manip"

out_file, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
match = re.search(r"run id '(\w+)'", out_file.read_text(errors="replace"))
videos = sorted((run_dir / "videos" / "play").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
if match is None or not videos:
    sys.exit(f"upload_video: no wandb run id in {out_file} or no video in {run_dir}/videos/play")
run = wandb.init(project=PROJECT, id=match.group(1), resume="must")
run.log({"play_video": wandb.Video(str(videos[-1]), format="mp4", caption=videos[-1].name)})
run.finish()
print(f"upload_video: {videos[-1].name} -> run {match.group(1)}")
