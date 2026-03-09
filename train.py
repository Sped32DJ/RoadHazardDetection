"""train.py – Fine-tune YOLOv8n on a custom road hazard dataset.

Expects an Ultralytics-format dataset directory with a ``dataset.yaml``
descriptor file.  See https://docs.ultralytics.com/datasets/ for the
expected layout:

    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml   ← referenced by --data argument

Usage
-----
    # Fine-tune from COCO pretrained YOLOv8n
    python train.py --data dataset/dataset.yaml

    # Resume a previous run
    python train.py --data dataset/dataset.yaml --resume runs/train/exp/weights/last.pt

    # Specify epochs, batch size, and image size
    python train.py --data dataset/dataset.yaml --epochs 100 --batch 8 --imgsz 640
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n for road hazard detection"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the Ultralytics dataset YAML file",
    )
    parser.add_argument(
        "--weights", default="yolov8n.pt",
        help="Starting weights (default: yolov8n.pt for COCO pretrained)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16; lower to 4–8 on Jetson Nano)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--device", default="0",
        help='Training device: "cpu", "0" (GPU), etc. (default: "0")',
    )
    parser.add_argument(
        "--project", default="runs/train",
        help='Output directory for training runs (default: "runs/train")',
    )
    parser.add_argument(
        "--name", default="road_hazard",
        help='Run name sub-directory (default: "road_hazard")',
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience in epochs (default: 20)",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"[ERROR] Dataset YAML not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    if args.resume:
        weights = args.resume
        print(f"[INFO] Resuming training from: {weights}")
    else:
        weights = args.weights
        print(f"[INFO] Starting training from: {weights}")

    model = YOLO(weights)

    print(
        f"[INFO] Training configuration:\n"
        f"  data   = {data_path}\n"
        f"  epochs = {args.epochs}\n"
        f"  batch  = {args.batch}\n"
        f"  imgsz  = {args.imgsz}\n"
        f"  device = {args.device}\n"
        f"  output = {args.project}/{args.name}"
    )

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=bool(args.resume),
        verbose=True,
    )

    print("[INFO] Training complete.")
    print(f"[INFO] Best weights saved to: {args.project}/{args.name}/weights/best.pt")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
