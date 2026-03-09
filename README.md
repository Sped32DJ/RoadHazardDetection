# RoadHazardDetection

Real-time road hazard detection running on a **Jetson Nano** using **YOLOv8n**.
Detects potholes, cracks, debris, manholes, and speed bumps from a live camera feed.

---

## Table of Contents
1. [Hardware & Software Requirements](#hardware--software-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Running Detection](#running-detection)
6. [Training a Custom Model](#training-a-custom-model)
7. [Troubleshooting](#troubleshooting)

---

## Hardware & Software Requirements

| Component | Requirement |
|-----------|-------------|
| Hardware  | NVIDIA Jetson Nano (2 GB or 4 GB) |
| Camera    | Raspberry Pi Camera v2 (CSI) **or** USB webcam |
| OS        | JetPack 4.6+ (Ubuntu 18.04 / 20.04) |
| Python    | 3.8 or later |
| CUDA      | Provided by JetPack |

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/Sped32DJ/RoadHazardDetection.git
cd RoadHazardDetection

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

> **Jetson Nano tip:** Install `opencv-python` through the system package instead
> of pip to get hardware-accelerated GStreamer support:
> ```bash
> sudo apt-get install python3-opencv
> ```
> Then remove `opencv-python` from `requirements.txt` before running `pip install`.

---

## Project Structure

```
RoadHazardDetection/
├── config.yaml       # Model, camera, display and output settings
├── detect.py         # Real-time detection script (main entry point)
├── train.py          # Fine-tuning script for custom hazard datasets
├── utils.py          # GStreamer helper, drawing utilities, CSV logger
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Configuration

All runtime settings live in **`config.yaml`**. Key sections:

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `weights` | Path to `.pt` weights file (`yolov8n.pt` downloads automatically) |
| `model` | `confidence` | Minimum detection confidence (0–1, default `0.45`) |
| `model` | `device` | `"0"` for GPU, `"cpu"` for CPU-only |
| `camera` | `source` | `"csi"` for CSI cam, `0` for USB webcam, or a file/RTSP URL |
| `camera` | `width/height` | Capture resolution |
| `output` | `save_video` | File path to save annotated video (leave empty to disable) |
| `output` | `log_csv` | File path to log detections as CSV (leave empty to disable) |

---

## Running Detection

```bash
# Default — use settings from config.yaml
python detect.py

# Override camera source (USB webcam index 0)
python detect.py --source 0

# Use a custom trained model
python detect.py --weights runs/train/road_hazard/weights/best.pt

# Headless / SSH session (no display window)
python detect.py --no-display --config config.yaml

# Save annotated output to a file
# (edit config.yaml: output.save_video: "output.mp4")
python detect.py
```

Press **`q`** in the display window (or `Ctrl+C` in the terminal) to stop.

---

## Training a Custom Model

1. **Collect and annotate images** using a tool such as
   [Roboflow](https://roboflow.com) or [CVAT](https://www.cvat.ai).
   Export in **YOLOv8 / Ultralytics** format.

2. **Prepare the dataset directory**:
   ```
   dataset/
   ├── images/train/  ← training images
   ├── images/val/    ← validation images
   ├── labels/train/  ← YOLO .txt annotation files
   ├── labels/val/
   └── dataset.yaml   ← class names + paths
   ```

3. **Run training**:
   ```bash
   # Fine-tune from COCO pretrained YOLOv8n (recommended starting point)
   python train.py --data dataset/dataset.yaml --epochs 50 --batch 8

   # Reduce batch size if you run out of memory on the Jetson Nano
   python train.py --data dataset/dataset.yaml --epochs 100 --batch 4
   ```

4. **Use your trained model** for detection:
   ```bash
   python detect.py --weights runs/train/road_hazard/weights/best.pt
   ```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Could not open video source: 'csi'` | Check that `nvarguscamerasrc` GStreamer plugin is available (`gst-inspect-1.0 nvarguscamerasrc`) |
| Low FPS | Lower `imgsz` in `config.yaml` (e.g. `320`) or use TensorRT export |
| CUDA out of memory | Reduce `batch` size in `train.py` or set `device: "cpu"` |
| Model not downloading | Ensure internet access, or manually place `yolov8n.pt` in the project root |
