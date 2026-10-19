#!/usr/bin/env bash
# Verify input frames; encode atomically; decode the complete video before publishing.
set -euo pipefail
python3 verify_frames.py
output=apalachee_gust_animation_v3.mp4
temporary=apalachee_gust_animation_v3.encoding.mp4
trap 'rm -f "$temporary"' EXIT
ffmpeg -y -nostdin -xerror -err_detect explode -loglevel error \
  -framerate 12 -i frames/f_%04d.png \
  -vf "tpad=stop_mode=clone:stop_duration=2.5,format=yuv420p" \
  -c:v libx264 -threads 2 -crf 20 -preset medium -movflags +faststart "$temporary"
ffmpeg -nostdin -v error -xerror -i "$temporary" -f null -
mv "$temporary" "$output"
echo "wrote and decoded $output"
