"""utils.py – Helper utilities for RoadHazardDetection.

Provides:
  - gstreamer_pipeline(): build a GStreamer string for the Jetson Nano CSI camera
  - open_video_source():  open the appropriate cv2.VideoCapture for a given source
  - draw_detections():    render bounding boxes and labels onto a frame
  - setup_csv_logger():   create / append a CSV file for logging detections
  - log_detections():     write a batch of detections to the CSV
"""

import csv
import io
import os
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np


def gstreamer_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """Return a GStreamer pipeline string for the Jetson Nano CSI camera.

    Parameters
    ----------
    width, height:
        Capture resolution in pixels.
    fps:
        Target frame-rate.
    flip_method:
        libargus / nvvidconv flip-method index (0 = no flip).

    Returns
    -------
    str
        A GStreamer pipeline string accepted by ``cv2.VideoCapture``.
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink"
    )


def open_video_source(cfg: dict) -> cv2.VideoCapture:
    """Open a ``cv2.VideoCapture`` from the source defined in *cfg*.

    Parameters
    ----------
    cfg:
        The ``camera`` sub-dict from ``config.yaml``.

    Returns
    -------
    cv2.VideoCapture
        An opened capture object.

    Raises
    ------
    RuntimeError
        If the capture device could not be opened.
    """
    source = cfg.get("source", 0)
    width = int(cfg.get("width", 1280))
    height = int(cfg.get("height", 720))
    fps = int(cfg.get("fps", 30))
    flip = int(cfg.get("flip_method", 0))

    if str(source).lower() == "csi":
        pipeline = gstreamer_pipeline(width, height, fps, flip)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


def draw_detections(
    frame: np.ndarray,
    boxes: list[Any],
    class_names: dict[int, str],
    colors: dict[int, list[int]],
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    """Draw bounding boxes and labels for each detection on *frame*.

    Parameters
    ----------
    frame:
        BGR image array to annotate (modified **in-place**).
    boxes:
        Sequence of Ultralytics ``Boxes`` objects from a YOLOv8 result.
    class_names:
        Mapping from integer class id to string label.
    colors:
        Mapping from integer class id to BGR colour tuple/list.
    confidence_threshold:
        Minimum confidence to display (detections below this are skipped).

    Returns
    -------
    np.ndarray
        The annotated frame (same array as *frame*).
    """
    default_color = (128, 128, 128)

    for box in boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, f"class_{cls_id}")
        color = tuple(colors.get(cls_id, default_color))

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def setup_csv_logger(csv_path: str) -> tuple[csv.writer, io.TextIOWrapper]:
    """Open (or create) a CSV file for detection logging.

    Parameters
    ----------
    csv_path:
        Filesystem path for the CSV output file.

    Returns
    -------
    tuple[csv.writer, io.TextIOWrapper]
        A ``(writer, file_handle)`` pair.  The caller is responsible for
        closing *file_handle* when logging is finished.

    Notes
    -----
    The header row (``timestamp,frame,class_id,label,confidence,x1,y1,x2,y2``)
    is written only when the file is newly created.
    """
    file_exists = os.path.isfile(csv_path)
    f = open(csv_path, "a", newline="", buffering=1)  # noqa: SIM115
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(
            ["timestamp", "frame", "class_id", "label", "confidence", "x1", "y1", "x2", "y2"]
        )
    return writer, f


def log_detections(
    writer: csv.writer,
    frame_idx: int,
    boxes: list[Any],
    class_names: dict[int, str],
) -> None:
    """Append detection rows to a CSV log.

    Parameters
    ----------
    writer:
        A :class:`csv.writer` returned by :func:`setup_csv_logger`.
    frame_idx:
        Zero-based index of the current frame.
    boxes:
        Sequence of Ultralytics ``Boxes`` objects from a YOLOv8 result.
    class_names:
        Mapping from integer class id to string label.
    """
    ts = datetime.now(timezone.utc).isoformat()
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        label = class_names.get(cls_id, f"class_{cls_id}")
        writer.writerow([ts, frame_idx, cls_id, label, f"{conf:.4f}", x1, y1, x2, y2])
