"""detect.py – Real-time road hazard detection on a Jetson Nano.

Uses a YOLOv8n model (pretrained on COCO or custom-trained) together with a
CSI / USB camera to detect road hazards (potholes, cracks, debris, …) in
real time and optionally save the annotated output to a video file and/or
log detections to CSV.

Usage
-----
    python detect.py                         # use settings from config.yaml
    python detect.py --config my_cfg.yaml    # override config file
    python detect.py --source 0              # USB webcam
    python detect.py --source /dev/video0    # explicit device node
    python detect.py --weights best.pt       # custom trained weights
    python detect.py --no-display            # headless (no GUI window)
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from utils import (
    draw_detections,
    log_detections,
    open_video_source,
    setup_csv_logger,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time road hazard detection (YOLOv8n + Jetson Nano)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Override the model weights path specified in config.yaml",
    )
    parser.add_argument(
        "--source", default=None,
        help="Override the camera source specified in config.yaml",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override the confidence threshold specified in config.yaml",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the live display window (useful for headless / SSH runs)",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load and return the YAML configuration as a nested dict."""
    config_path = Path(path)
    if not config_path.is_file():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # ── Load configuration ───────────────────────────────────────────────────
    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    cam_cfg = cfg.get("camera", {})
    disp_cfg = cfg.get("display", {})
    out_cfg = cfg.get("output", {})
    class_names: dict[int, str] = {
        int(k): v for k, v in cfg.get("classes", {}).items()
    }
    colors: dict[int, list[int]] = {
        int(k): v for k, v in disp_cfg.get("colors", {}).items()
    }

    # Apply CLI overrides
    weights = args.weights or model_cfg.get("weights", "yolov8n.pt")
    confidence = args.conf if args.conf is not None else float(model_cfg.get("confidence", 0.45))
    iou = float(model_cfg.get("iou", 0.45))
    imgsz = int(model_cfg.get("imgsz", 640))
    device = str(model_cfg.get("device", "0"))

    if args.source is not None:
        cam_cfg["source"] = args.source

    show_display = disp_cfg.get("show", True) and not args.no_display
    window_title = disp_cfg.get("window_title", "Road Hazard Detection")

    save_video_path: str = out_cfg.get("save_video", "") or ""
    log_csv_path: str = out_cfg.get("log_csv", "") or ""

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {weights}  (device={device})")
    model = YOLO(weights)

    # ── Open video source ────────────────────────────────────────────────────
    print(f"[INFO] Opening camera source: {cam_cfg.get('source')}")
    cap = open_video_source(cam_cfg)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[INFO] Stream: {frame_width}x{frame_height} @ {fps_cap:.1f} fps")

    # ── Optional: video writer ───────────────────────────────────────────────
    video_writer = None
    if save_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            save_video_path, fourcc, fps_cap, (frame_width, frame_height)
        )
        print(f"[INFO] Saving annotated video to: {save_video_path}")

    # ── Optional: CSV logger ─────────────────────────────────────────────────
    csv_writer = None
    csv_file = None
    if log_csv_path:
        csv_writer, csv_file = setup_csv_logger(log_csv_path)
        print(f"[INFO] Logging detections to: {log_csv_path}")

    # ── Detection loop ───────────────────────────────────────────────────────
    frame_idx = 0
    print("[INFO] Starting detection loop — press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or camera error — exiting.")
                break

            # Run YOLOv8 inference
            results = model.predict(
                source=frame,
                conf=confidence,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            # Annotate frame
            boxes = results[0].boxes
            annotated = draw_detections(frame, boxes, class_names, colors, confidence)

            # Overlay FPS counter
            fps_text = f"Frame: {frame_idx}"
            cv2.putText(
                annotated,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if video_writer is not None:
                video_writer.write(annotated)

            if csv_writer is not None:
                log_detections(csv_writer, frame_idx, boxes, class_names)

            if show_display:
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 'q' pressed — stopping.")
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if csv_file is not None:
            csv_file.close()
        if show_display:
            cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_idx} frames.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
