#!/bin/bash
# Works for video and pictures
# Usage: ./detectVideo.sh /path/to/video.mp4

set -euo pipefail

yolo task=detect mode=predict model=models/best.pt conf=0.25
