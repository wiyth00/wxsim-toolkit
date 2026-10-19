#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
../.venv/bin/python verify_frames.py
output=apalachee_gust_animation_size2p5x.mp4
temporary=apalachee_gust_animation_size2p5x.encoding.mp4
trap 'rm -f "$temporary"' EXIT
ffmpeg -y -nostdin -xerror -err_detect explode -loglevel error -framerate 12 -i frames/f_%04d.png -vf 'tpad=stop_mode=clone:stop_duration=2.5,format=yuv420p' -c:v libx264 -threads 2 -crf 20 -preset medium -movflags +faststart "$temporary"
ffmpeg -nostdin -v error -xerror -i "$temporary" -f null -
mv "$temporary" "$output"
echo "Verified all frames and decoded $output"
