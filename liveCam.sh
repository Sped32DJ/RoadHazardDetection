#!/bin/bash
yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source=0 show=True imgsz=640
# Decreasing the image size to 640x640 can help improve the frame rate, for more real time performance.
# Change to imgz=320 for even better performance
# Credit: https://www.hackster.io/stefanw1/install-and-test-of-yolov8-on-jetson-nano-4659ed