"""Score a downloaded video for suitability as a fancam source.

Runs three analyses:
  1. PySceneDetect ContentDetector -> cuts per second
  2. YOLOv8n on N uniform frames -> median person count + target bbox stats
  3. Composite 0-100 quality score (visual only; audio gap is in match_video.py)

Writes two sidecars next to the .mp4:
  <stem>.quality.json  -- metrics + composite score
  <stem>.bboxes.npz    -- cached per-frame person bboxes for Phase 2 tracking

Usage:
    python quality_probe.py --video videos/<stem>.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video

# Windows cp950 console can't render Korean/Japanese titles without this.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
VIDEOS_DIR = ROOT / "videos"

# YOLOv8n model cached next to the script on first run.
YOLO_WEIGHTS = ROOT / "yolov8n.pt"
SAMPLE_FRAMES = 20       # frames to run YOLO on
TARGET_H_MIN = 0.40      # wanted: tallest person ≥ 40% of frame height
LATERAL_STD_MAX = 0.30   # wanted: bbox center x std < 30% of frame width


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- scene cuts ----------

def count_cuts(video_path: Path, threshold: float = 27.0) -> int:
    """Number of shot boundaries in the video."""
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    # scenes is [(start, end), ...] one per shot; cuts = shots - 1
    return max(0, len(scenes) - 1)


# ---------- YOLO frame probe ----------

def yolo_sample(video_path: Path, n: int) -> dict:
    """Sample n frames uniformly and run YOLO person detection on each.

    Returns dict of metrics + a list of per-frame bboxes (xyxy normalised).
    """
    from ultralytics import YOLO  # noqa: PLC0415  (slow import; keep local)
    model = YOLO(str(YOLO_WEIGHTS)) if YOLO_WEIGHTS.exists() else YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or width <= 0:
        cap.release()
        raise RuntimeError("video metadata unreadable")

    # Avoid the first/last 5% — title cards and end credits throw off stats.
    lo = int(total * 0.05)
    hi = int(total * 0.95)
    targets = np.linspace(lo, hi, n).astype(int)

    per_frame_bboxes: list[np.ndarray] = []
    person_counts: list[int] = []
    target_h_ratios: list[float] = []
    target_cx: list[float] = []
    ts: list[float] = []

    for i, frame_idx in enumerate(targets, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        ts.append(frame_idx / fps)
        # class 0 = person in COCO; conf floor filters false positives.
        res = model(frame, classes=[0], conf=0.3, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            per_frame_bboxes.append(np.zeros((0, 5), dtype=np.float32))
            person_counts.append(0)
            continue
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32)[:, None]
        bbox = np.concatenate([xyxy, conf], axis=1)
        per_frame_bboxes.append(bbox)
        person_counts.append(len(bbox))

        # "target" = tallest bbox in this frame (largest person tends to be
        # front-of-stage / foreground, which fancams focus on).
        h = (bbox[:, 3] - bbox[:, 1]) / height
        cx = ((bbox[:, 0] + bbox[:, 2]) / 2) / width
        idx = int(np.argmax(h))
        target_h_ratios.append(float(h[idx]))
        target_cx.append(float(cx[idx]))

        if i % 5 == 0 or i == len(targets):
            log(f"  [yolo] {i}/{len(targets)} frames  "
                f"cur person_count={len(bbox)} tallest_h={target_h_ratios[-1]:.2f}")

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "sample_ts": ts,
        "person_counts": person_counts,
        "target_h_ratios": target_h_ratios,
        "target_cx": target_cx,
        "per_frame_bboxes": per_frame_bboxes,
    }


# ---------- composite score ----------

def compute_score(cuts_per_sec: float, median_persons: float,
                  median_h: float, cx_std: float) -> dict:
    """Return per-metric sub-scores (0..1) and a 0..100 composite."""
    # cuts/sec: exp decay. 0 cuts -> 1.0; 0.08 -> 0.45; 0.3 -> 0.05
    s_cuts = math.exp(-10.0 * cuts_per_sec)
    # persons: 1.0 if median is 1, scales down to 0 by 4.
    if median_persons <= 0:
        s_persons = 0.0
    else:
        s_persons = max(0.0, 1.0 - abs(median_persons - 1.0) / 3.0)
    # bbox height: 0 below 0.2, full above TARGET_H_MIN.
    if median_h <= 0.2:
        s_height = 0.0
    elif median_h >= TARGET_H_MIN:
        s_height = 1.0
    else:
        s_height = (median_h - 0.2) / (TARGET_H_MIN - 0.2)
    # lateral stability: wide std drops score. Allow moderate pan.
    s_stability = max(0.0, 1.0 - cx_std / LATERAL_STD_MAX)

    composite = (0.35 * s_cuts + 0.30 * s_persons
                 + 0.20 * s_height + 0.15 * s_stability) * 100
    return {
        "s_cuts": round(s_cuts, 3),
        "s_persons": round(s_persons, 3),
        "s_height": round(s_height, 3),
        "s_stability": round(s_stability, 3),
        "composite": round(composite, 1),
    }


def run(video: Path) -> dict:
    t0 = time.time()
    log(f"[probe] {video.name}")

    log("[cuts] running PySceneDetect...")
    cuts = count_cuts(video)
    # duration from cv2 so we don't need to load the file twice
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    duration = total / fps if fps > 0 else 0
    cuts_per_sec = cuts / duration if duration > 0 else 0
    log(f"  cuts={cuts} duration={duration:.1f}s -> {cuts_per_sec:.3f}/s")

    log(f"[yolo] sampling {SAMPLE_FRAMES} frames...")
    y = yolo_sample(video, SAMPLE_FRAMES)
    median_persons = float(np.median(y["person_counts"])) if y["person_counts"] else 0
    median_h = float(np.median(y["target_h_ratios"])) if y["target_h_ratios"] else 0
    cx_std = float(np.std(y["target_cx"])) if len(y["target_cx"]) > 1 else 0
    log(f"  median_persons={median_persons:.1f} median_target_h={median_h:.2f} cx_std={cx_std:.3f}")

    scores = compute_score(cuts_per_sec, median_persons, median_h, cx_std)
    log(f"[score] {scores}")

    # bbox cache: save per-frame xyxy+conf as object array so variable-length
    # per-frame rows survive np.savez.
    bbox_path = video.with_suffix(".bboxes.npz")
    np.savez_compressed(
        bbox_path,
        sample_ts=np.asarray(y["sample_ts"], dtype=np.float32),
        person_counts=np.asarray(y["person_counts"], dtype=np.int32),
        bboxes=np.asarray(y["per_frame_bboxes"], dtype=object),
        width=y["width"], height=y["height"], fps=y["fps"],
    )

    report = {
        "video": str(video.relative_to(ROOT)).replace("\\", "/"),
        "cuts": cuts,
        "cuts_per_sec": round(cuts_per_sec, 4),
        "duration_sec": round(duration, 1),
        "median_persons": median_persons,
        "median_target_h_ratio": round(median_h, 3),
        "target_cx_std": round(cx_std, 3),
        "scores": scores,
        "bbox_cache": str(bbox_path.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    quality_path = video.with_suffix(".quality.json")
    quality_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[ok] {quality_path.name}  (composite={scores['composite']})")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_absolute():
        video = (ROOT / video).resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    report = run(video)
    print("PROBE " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
