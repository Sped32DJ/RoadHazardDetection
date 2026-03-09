#!/usr/bin/env bash
# This script runs the YOLO detect command on a video file.
# Usage: ./detectVideo.sh /path/to/video.mp4

set -euo pipefail

# Show brief usage
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: $0 [video-file]"
    echo "If no video-file is provided, you'll be prompted to enter one."
    exit 0
fi

# Use first positional argument if present, otherwise prompt the user
if [ -n "${1:-}" ]; then
    video="$1"
else
    read -r -p "Enter video filename: " video
fi

# Check if the file exists
if [ -f "$video" ]; then
    echo "Video file '$video' found."
else
    echo "Video file '$video' not found. Please make sure the file exists and try again." >&2
    exit 1
fi

# Run the detection command (adjust options as needed)
yolo task=detect mode=predict model=models/best.pt conf=0.25 source="$video" show=True imgsz=640