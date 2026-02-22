"""
crowd_analyzer.py

Video-based crowd analysis pipeline:
- Detects people per frame using YOLOv8 (COCO "person" class = 0)
- Produces:
  1) annotated.mp4      (bounding boxes + live count overlay)
  2) crowd_data.csv     (time series counts + smoothed counts)
  3) heatmap.png        (spatial density visualization)
  4) summary.txt        (quick metrics + crowded segments)

This module is designed to be run either:
- directly from the CLI, or
- invoked by the Streamlit app via environment variables.

Environment variables (optional):
- VIDEO_PATH_OVERRIDE: path to input video
- CONF_OVERRIDE: detection confidence threshold (float)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# -----------------------------
# Configuration
# -----------------------------

# Allow external callers (Streamlit) to override runtime parameters without editing code.
VIDEO_PATH = os.environ.get("VIDEO_PATH_OVERRIDE", "videos/video.mp4")
CONF = float(os.environ.get("CONF_OVERRIDE", "0.35"))

MODEL_NAME = "yolov8n.pt"  # Lightweight model. Consider yolov8s.pt for improved accuracy.

OUT_DIR = Path("outputs")
OUT_VIDEO = OUT_DIR / "annotated.mp4"
OUT_CSV = OUT_DIR / "crowd_data.csv"
OUT_HEATMAP = OUT_DIR / "heatmap.png"
OUT_SUMMARY = OUT_DIR / "summary.txt"

SMOOTH_WINDOW_FRAMES = 15
CROWDED_THRESHOLD = 15
HEAT_RADIUS = 25


# -----------------------------
# Helpers
# -----------------------------
def seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to mm:ss for readable overlays and summaries."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def add_heat_blob(heat: np.ndarray, x: int, y: int, radius: int) -> None:
    """
    Add a circular heat "blob" centered at (x, y). Using a radius produces a smoother
    density map than incrementing a single pixel.
    """
    h, w = heat.shape

    x1 = max(0, x - radius)
    x2 = min(w - 1, x + radius)
    y1 = max(0, y - radius)
    y2 = min(h - 1, y + radius)

    yy, xx = np.ogrid[y1 : y2 + 1, x1 : x2 + 1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    heat[y1 : y2 + 1, x1 : x2 + 1][mask] += 1.0


def find_crowded_segments(df: pd.DataFrame, threshold: float, gap_limit_sec: float = 1.0):
    """
    Identify time ranges where the smoothed crowd count exceeds the given threshold.
    We split segments when there is a time gap larger than gap_limit_sec.
    """
    crowded = df[df["people_smooth"] >= threshold].copy()
    if crowded.empty:
        return []

    times = crowded["time_sec"].values
    segments = []
    start = times[0]
    prev = times[0]

    for t in times[1:]:
        if t - prev > gap_limit_sec:
            segments.append((start, prev))
            start = t
        prev = t

    segments.append((start, prev))
    return segments


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("crowd_analyzer.py starting")
    print(f"Input video: {VIDEO_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Confidence: {CONF}")

    # Load YOLO model. On first run, ultralytics will download weights automatically.
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {VIDEO_PATH}\n"
            f"Verify the path and ensure the file is a valid video format."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video opened: {width}x{height} @ {fps:.2f} fps")

    # Setup output video writer (annotated MP4).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (width, height))

    # Accumulators for analytics.
    heat = np.zeros((height, width), dtype=np.float32)
    rows = []

    frame_idx = 0
    print("Processing frames...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO "person" class in COCO is class 0. We restrict classes to [0] to avoid
        # detecting other objects.
        result = model.predict(frame, conf=CONF, classes=[0], verbose=False)[0]

        people_count = 0
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), score in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                people_count += 1

                # Draw bounding box and confidence for quick inspection.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # Update heatmap using the box center point.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cx < width and 0 <= cy < height:
                    add_heat_blob(heat, cx, cy, HEAT_RADIUS)

        t_sec = frame_idx / fps

        # Overlay live count and timestamp.
        cv2.putText(
            frame,
            f"People: {people_count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Time: {seconds_to_mmss(t_sec)}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        writer.write(frame)

        rows.append({"time_sec": t_sec, "people_count": people_count})
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx} frames")

    cap.release()
    writer.release()

    print("Frame processing complete. Writing outputs...")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No frames processed. The input video may be empty or unreadable.")

    df["people_smooth"] = df["people_count"].rolling(
        window=SMOOTH_WINDOW_FRAMES,
        min_periods=1,
    ).mean()

    df.to_csv(OUT_CSV, index=False)

    # Heatmap normalization for visualization.
    heat_norm = heat / (heat.max() if heat.max() > 0 else 1.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(heat_norm, cmap="hot")
    plt.title("Crowd Density Heatmap (relative)")
    plt.axis("off")
    plt.savefig(OUT_HEATMAP, dpi=200, bbox_inches="tight")
    plt.close()

    avg = float(df["people_smooth"].mean())
    peak = float(df["people_smooth"].max())
    peak_time = float(df.loc[df["people_smooth"].idxmax(), "time_sec"])

    segments = find_crowded_segments(df, threshold=CROWDED_THRESHOLD)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Crowd Density Analyzer Summary\n")
        f.write("--------------------------------\n")
        f.write(f"Video: {VIDEO_PATH}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Confidence: {CONF}\n\n")
        f.write(f"Average crowd (smoothed): {avg:.2f}\n")
        f.write(f"Peak crowd (smoothed): {peak:.2f} at {seconds_to_mmss(peak_time)}\n")
        f.write(f"Crowded threshold: >= {CROWDED_THRESHOLD}\n\n")

        if segments:
            f.write("Crowded segments (approx):\n")
            for a, b in segments:
                f.write(f"  {seconds_to_mmss(a)} to {seconds_to_mmss(b)}\n")
        else:
            f.write("Crowded segments (approx): none found with current threshold.\n")

        f.write("\nOutputs:\n")
        f.write(f"  Annotated video: {OUT_VIDEO}\n")
        f.write(f"  Counts CSV:      {OUT_CSV}\n")
        f.write(f"  Heatmap image:   {OUT_HEATMAP}\n")

    print("Done. Outputs written to outputs/")


if __name__ == "__main__":
    main()