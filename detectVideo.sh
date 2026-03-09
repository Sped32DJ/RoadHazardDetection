#!/bin/bash
# This script detects if a video file is present in the current directory
if ls *.mp4 1> /dev/null 2>&1; then
    echo "Video file detected: $(ls *.mp4)"
else
    echo "No video file detected in the current directory."
fi