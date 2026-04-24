"""Smoke test for pose_track: run RTMPose on 10 frames of a dance
practice video with hand-rolled bboxes to verify the head keypoints
come back in sane locations.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_track import _rtmpose_model, _KP_NOSE, _KP_L_EYE, _KP_R_EYE


def main():
    if len(sys.argv) < 2:
        print("usage: smoke_pose.py <video.mp4>")
        sys.exit(1)
    video = Path(sys.argv[1])
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {video.name}  {W}x{H}  {total} frames")

    # Sample 10 frames evenly.
    sample_fis = np.linspace(total * 0.1, total * 0.9, 10).astype(int)

    print(f"loading RTMPose-m...")
    t0 = time.time()
    pose = _rtmpose_model()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Fake bbox: center 30-70% x, 15-90% y (roughly person-sized).
    x1, y1 = int(0.30 * W), int(0.15 * H)
    x2, y2 = int(0.70 * W), int(0.90 * H)
    print(f"using fake bbox: ({x1},{y1}) -> ({x2},{y2})")

    times = []
    for fi in sample_fis:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            print(f"  fi={fi}: read failed")
            continue
        t0 = time.time()
        kpts, scores = pose(frame, bboxes=[[x1, y1, x2, y2]])
        elapsed = time.time() - t0
        times.append(elapsed)
        k = kpts[0]
        s = scores[0]
        nose = k[_KP_NOSE]
        l_eye = k[_KP_L_EYE]
        r_eye = k[_KP_R_EYE]
        ns = s[_KP_NOSE]
        les = s[_KP_L_EYE]
        res = s[_KP_R_EYE]
        head_x = (nose[0] + l_eye[0] + r_eye[0]) / 3.0 / W
        head_y = (nose[1] + l_eye[1] + r_eye[1]) / 3.0 / H
        print(f"  fi={fi:5d}  nose=({nose[0]:.0f},{nose[1]:.0f}) conf={ns:.2f}  "
              f"head_xy_norm=({head_x:.3f},{head_y:.3f})  t={elapsed*1000:.0f}ms")
    cap.release()
    print(f"mean inference: {np.mean(times)*1000:.0f}ms/frame "
          f"(rate: {1/np.mean(times):.1f} fps)")


if __name__ == "__main__":
    main()
