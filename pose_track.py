"""RTMPose-m head-keypoint tracker.

Given a tracker's per-frame target bbox, run RTMPose-m on each bbox and
return the per-frame HEAD CENTER (nose + eyes midpoint) in image-
normalised coords. This is what M3b uses to place the head at a fixed
y-fraction of the output frame across sources — so hard-cut "outfit
swap" transitions don't also pop the head vertically.

Why pose, not just the bbox top-edge?

    BoT-SORT bboxes are body bboxes; the detector's top edge is often
    hair/hat/hand-above-head depending on the dancer's pose. A bbox
    whose top edge moves by 5-10% of image height frame-to-frame produces
    a visibly unstable head position in the output crop.

    COCO-17 nose+eyes keypoints are very stable (±0.5% of image height
    in a static camera) and give the geometric head center directly.

Why RTMPose over Wholebody / Body?

    `Body(mode="balanced")` pairs YOLOX-m detection with RTMPose-m pose.
    We already have tracker bboxes — re-running YOLOX-m per frame is
    wasted work. `RTMPose` takes the bbox directly and runs pose only.
    ~30-40ms/frame on CPU (vs ~150ms for full Body).

Cache:
    `<video_stem>__<start_sec:.1f>_<dur_sec:.1f>.pose.json` next to the
    mp4. Large clips (60s @ 30fps = 1800 frames) produce ~25 KB of JSON
    — cheap. Pose model weights are cached by rtmlib under ~/.cache/.

Output shape:
    head_xy_norm  (n_frames, 2) float32 — (x, y) in [0, 1] image coords.
                  (0, 0) on frames where no target bbox exists — the
                  consumer must gate on head_conf before using.
    head_conf     (n_frames,) float32 — mean of (nose, l_eye, r_eye)
                  keypoint confidence; ~0 on missing frames.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


_RTMPOSE_CACHE = None  # lazy singleton
# RTMPose-m body7 checkpoint. body7 = 7 public datasets pretrained →
# robust on performance/dance imagery where limbs are folded/hidden.
# Input 192x256 (WxH), 17 COCO keypoints.
_RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_"
    "20230504.zip"
)

# COCO-17 head keypoint indices.
_KP_NOSE = 0
_KP_L_EYE = 1
_KP_R_EYE = 2


def _rtmpose_model():
    """Lazy-load RTMPose once per process. CPU-only — same reasoning as
    shot_gate: MPS numerical drift isn't worth the speedup when ffmpeg
    decode dominates anyway."""
    global _RTMPOSE_CACHE
    if _RTMPOSE_CACHE is not None:
        return _RTMPOSE_CACHE
    # rtmlib's decode path uses ffmpeg-python for video; bboxes-only mode
    # doesn't, but onnxruntime's provider loader sometimes peeks at PATH.
    # Add brew paths idempotently for uvicorn-launched contexts that don't
    # inherit shell PATH.
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if extra not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
    from rtmlib import RTMPose  # noqa: PLC0415
    model = RTMPose(
        onnx_model=_RTMPOSE_URL,
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cpu",
    )
    _RTMPOSE_CACHE = model
    return model


@dataclass
class HeadTrack:
    """Per-frame head-center trajectory for a tracked segment.

    v2 adds yaw_proxy / yaw_conf, derived from nose-vs-eye geometry in
    the same RTMPose pass. `yaw_proxy` is a signed dimensionless estimate
    of head yaw:
        yaw_proxy = (nose_x - eye_midpoint_x) / eye_span

    Frontal → ≈ 0. Subject turning to their own LEFT (viewer sees
    subject's left cheek — typical of a camera placed STAGE-RIGHT) →
    positive; subject turning to their own RIGHT → negative. Clipped to
    [-1.0, 1.0]. M4a uses the per-source median to bucket sources into
    frontal / left-side / right-side for cut planning.

    v1 caches (head coords only) can still be loaded — yaw fields fall
    back to zero and session_yaw_bucket returns None for that source.
    """
    # Flattened [(x, y), ...] then packed in a plain list for JSON.
    head_xy_norm: np.ndarray       # (n, 2) float32
    head_conf: np.ndarray          # (n,) float32
    yaw_proxy: np.ndarray          # (n,) float32  (M4a)
    yaw_conf: np.ndarray           # (n,) float32  (M4a)
    n_frames: int
    probe_elapsed_sec: float
    version: int = 2


def _sidecar_for(video: Path, start_sec: float, dur_sec: float) -> Path:
    """Pose sidecar is keyed on (video, start_sec, dur_sec) because one
    video can be tracked at multiple offsets (different song alignments).
    Using 1-decimal precision matches the audio-alignment step's output
    and avoids float-equality cache misses."""
    return video.with_suffix(f".{start_sec:.1f}_{dur_sec:.1f}.pose.json")


def load_cached(video: Path, start_sec: float, dur_sec: float
                ) -> HeadTrack | None:
    sc = _sidecar_for(video, start_sec, dur_sec)
    if not sc.exists():
        return None
    try:
        data = json.loads(sc.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = data.get("version")
    if v not in (1, 2):
        return None
    try:
        xy = np.asarray(data["head_xy_norm"], dtype=np.float32).reshape(-1, 2)
        conf = np.asarray(data["head_conf"], dtype=np.float32)
        if xy.shape[0] != conf.shape[0]:
            return None
        n = xy.shape[0]
        if v >= 2:
            yaw = np.asarray(data["yaw_proxy"], dtype=np.float32)
            yaw_c = np.asarray(data["yaw_conf"], dtype=np.float32)
            if yaw.shape[0] != n or yaw_c.shape[0] != n:
                return None
        else:
            # v1 cache: no yaw info. Fill with zeros; session_yaw_bucket
            # will skip this source (yaw_conf==0 everywhere).
            yaw = np.zeros(n, dtype=np.float32)
            yaw_c = np.zeros(n, dtype=np.float32)
        return HeadTrack(
            head_xy_norm=xy,
            head_conf=conf,
            yaw_proxy=yaw,
            yaw_conf=yaw_c,
            n_frames=int(data["n_frames"]),
            probe_elapsed_sec=float(data["probe_elapsed_sec"]),
            version=int(v),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _save_cache(video: Path, start_sec: float, dur_sec: float,
                ht: HeadTrack, log_fn=print) -> None:
    sc = _sidecar_for(video, start_sec, dur_sec)
    try:
        payload = {
            "version": ht.version,
            "n_frames": ht.n_frames,
            "probe_elapsed_sec": ht.probe_elapsed_sec,
            # Pack as flat (n*2,) list so the JSON stays compact and
            # human-inspectable. Round to 4 decimals (≈1 pixel on 1080p).
            "head_xy_norm": [
                round(float(v), 4) for v in ht.head_xy_norm.reshape(-1).tolist()
            ],
            "head_conf": [round(float(v), 3) for v in ht.head_conf.tolist()],
            "yaw_proxy": [round(float(v), 3) for v in ht.yaw_proxy.tolist()],
            "yaw_conf": [round(float(v), 3) for v in ht.yaw_conf.tolist()],
        }
        sc.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        log_fn(f"[pose-track] warn: failed to write sidecar for "
               f"{video.name}: {e}")


def track_head_keypoints(video: Path,
                         tracks: "dict[int, np.ndarray]",
                         target_ids: list[int],
                         start_sec: float,
                         dur_sec: float,
                         meta,
                         force: bool = False,
                         sample_stride: int = 1,
                         min_keypoint_conf: float = 0.30,
                         log_fn=print) -> HeadTrack:
    """Return per-frame head-center for the tracked target inside the
    segment [start_sec, start_sec+dur_sec).

    Frames without a target bbox (or where pose confidence < threshold)
    get head_xy_norm=(0,0) and head_conf=0 — consumer must gate on conf.

    Per-frame loop:
      1. Look up target bbox from `tracks[tid]` (rows = [fi, x1, y1, x2, y2,
         score]). `tid` = first of `target_ids` visible on this frame.
      2. Decode the frame via cv2.VideoCapture.seek(fi+start_fi).
      3. Run RTMPose on (frame, [[x1,y1,x2,y2]]).
      4. head_center = weighted mean of (nose, l_eye, r_eye) using their
         keypoint confidences. Fallback to nose alone if eyes missing.

    `sample_stride`: run pose every Nth frame to save time; linearly
    interpolate between. stride=1 = every frame (~36 frames/sec on CPU).
    """
    cached = None if force else load_cached(video, start_sec, dur_sec)
    if cached is not None and cached.n_frames == meta.n_frames:
        log_fn(f"[pose-track] cached: {video.name} "
               f"{start_sec:.1f}s+{dur_sec:.1f}s → "
               f"{int((cached.head_conf > min_keypoint_conf).sum())}/"
               f"{cached.n_frames} frames with head")
        return cached

    import cv2  # noqa: PLC0415

    n = meta.n_frames
    start_fi = int(round(start_sec * meta.fps))
    head_xy = np.zeros((n, 2), dtype=np.float32)
    head_conf = np.zeros(n, dtype=np.float32)
    yaw_proxy = np.zeros(n, dtype=np.float32)
    yaw_conf = np.zeros(n, dtype=np.float32)

    # Pre-index target bboxes by frame index so we avoid a linear scan
    # per frame. Rows: [fi, x1, y1, x2, y2, score, ...].
    per_frame_bbox: dict[int, tuple[float, float, float, float]] = {}
    for tid in target_ids:
        rows = tracks.get(tid)
        if rows is None:
            continue
        for r in rows:
            fi = int(r[0])
            # First target wins — target_ids is ordered by confidence.
            per_frame_bbox.setdefault(
                fi, (float(r[1]), float(r[2]), float(r[3]), float(r[4])))

    if not per_frame_bbox:
        log_fn(f"[pose-track] {video.name}: no target bbox in segment, "
               f"returning empty trajectory")
        return HeadTrack(
            head_xy_norm=head_xy,
            head_conf=head_conf,
            yaw_proxy=yaw_proxy,
            yaw_conf=yaw_conf,
            n_frames=n,
            probe_elapsed_sec=0.0,
        )

    model = _rtmpose_model()
    t0 = time.time()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log_fn(f"[pose-track] {video.name}: cv2.VideoCapture failed")
        return HeadTrack(
            head_xy_norm=head_xy,
            head_conf=head_conf,
            yaw_proxy=yaw_proxy,
            yaw_conf=yaw_conf,
            n_frames=n,
            probe_elapsed_sec=0.0,
        )

    # Stride-subsampled frame indices: process every Nth frame, then
    # linearly interpolate between anchors in a second pass.
    sample_frames = sorted(set(range(0, n, max(1, sample_stride))))
    if (n - 1) not in sample_frames:
        sample_frames.append(n - 1)

    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fi)
    cur_fi = start_fi
    for fi in sample_frames:
        bbox = per_frame_bbox.get(fi)
        if bbox is None:
            continue
        # Advance reader to the desired frame. Grab() is ~10x faster than
        # read() when we just want to skip ahead; only retrieve() on the
        # target frame. In practice OpenCV's seek is already fast for
        # short jumps so we only seek when we'd grab more than 4 frames.
        target_fi = start_fi + fi
        skip = target_fi - cur_fi
        if skip < 0 or skip > 4:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
        else:
            for _ in range(skip):
                if not cap.grab():
                    break
        ok, frame = cap.read()
        cur_fi = target_fi + 1
        if not ok or frame is None:
            continue

        x1, y1, x2, y2 = bbox
        # Clamp to frame bounds — the tracker can produce slightly
        # out-of-frame bboxes when the person is at the edge.
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0.0, min(float(w_frame - 1), x1))
        x2 = max(0.0, min(float(w_frame - 1), x2))
        y1 = max(0.0, min(float(h_frame - 1), y1))
        y2 = max(0.0, min(float(h_frame - 1), y2))
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue
        try:
            kpts, scores = model(frame, bboxes=[[x1, y1, x2, y2]])
        except Exception as e:
            log_fn(f"[pose-track] warn: pose failed on fi={fi}: {e!r}")
            continue
        # kpts: (1, 17, 2) xy in ORIGINAL image coords (RTMPose's
        # postprocess un-does the affine), scores: (1, 17).
        k = kpts[0]
        s = scores[0]
        # Head center = confidence-weighted mean of nose + eyes.
        head_pts = []
        head_w = []
        for idx in (_KP_NOSE, _KP_L_EYE, _KP_R_EYE):
            c = float(s[idx])
            if c >= min_keypoint_conf:
                head_pts.append(k[idx])
                head_w.append(c)
        if not head_pts:
            continue
        pts = np.asarray(head_pts, dtype=np.float32)
        ws = np.asarray(head_w, dtype=np.float32)
        center = (pts * ws[:, None]).sum(axis=0) / ws.sum()
        head_xy[fi, 0] = center[0] / float(w_frame)
        head_xy[fi, 1] = center[1] / float(h_frame)
        head_conf[fi] = float(ws.mean())

        # Yaw proxy: signed nose offset from eye-midpoint, normalised by
        # eye-span. Needs all three of nose/l_eye/r_eye above threshold.
        # eye_span is in pixels — if too small (face tiny or eyes
        # essentially coincident), the ratio is unstable, so gate on
        # minimum-span of ~1% of frame width.
        nose_c = float(s[_KP_NOSE])
        l_eye_c = float(s[_KP_L_EYE])
        r_eye_c = float(s[_KP_R_EYE])
        if (nose_c >= min_keypoint_conf and
                l_eye_c >= min_keypoint_conf and
                r_eye_c >= min_keypoint_conf):
            nose_x = float(k[_KP_NOSE, 0])
            l_eye_x = float(k[_KP_L_EYE, 0])
            r_eye_x = float(k[_KP_R_EYE, 0])
            eye_mid_x = 0.5 * (l_eye_x + r_eye_x)
            eye_span = abs(l_eye_x - r_eye_x)
            min_span = 0.01 * float(w_frame)
            if eye_span >= min_span:
                # Positive: nose toward subject's-left eye
                #           (viewer's right in image coords).
                # Negative: nose toward subject's-right eye.
                raw = (nose_x - eye_mid_x) / eye_span
                # Clip to [-1, 1] — values outside mean "near-profile" and
                # the geometric model breaks down past ±45° anyway.
                yaw_proxy[fi] = float(max(-1.0, min(1.0, raw)))
                yaw_conf[fi] = min(nose_c, l_eye_c, r_eye_c)
        processed += 1

    cap.release()

    # Linear interpolation between anchors to fill in stride-skipped
    # frames. Only interpolate when the gap is <= 2 * sample_stride
    # frames — longer gaps usually mean the target is off-screen.
    if sample_stride > 1:
        max_gap = sample_stride * 2
        anchor_fis = np.where(head_conf > 0)[0]
        for i in range(len(anchor_fis) - 1):
            a, b = int(anchor_fis[i]), int(anchor_fis[i + 1])
            if b - a <= 1 or b - a > max_gap:
                continue
            pa = head_xy[a]
            pb = head_xy[b]
            ca = head_conf[a]
            cb = head_conf[b]
            for mid in range(a + 1, b):
                t = (mid - a) / float(b - a)
                head_xy[mid] = pa * (1 - t) + pb * t
                head_conf[mid] = ca * (1 - t) + cb * t
        # Interpolate yaw separately: it has its own anchor mask (only
        # frames where all 3 of nose+eyes cleared threshold).
        yaw_anchor_fis = np.where(yaw_conf > 0)[0]
        for i in range(len(yaw_anchor_fis) - 1):
            a, b = int(yaw_anchor_fis[i]), int(yaw_anchor_fis[i + 1])
            if b - a <= 1 or b - a > max_gap:
                continue
            ya, yb = yaw_proxy[a], yaw_proxy[b]
            yca, ycb = yaw_conf[a], yaw_conf[b]
            for mid in range(a + 1, b):
                t = (mid - a) / float(b - a)
                yaw_proxy[mid] = ya * (1 - t) + yb * t
                yaw_conf[mid] = yca * (1 - t) + ycb * t

    elapsed = time.time() - t0
    covered = int((head_conf > min_keypoint_conf).sum())
    yaw_covered = int((yaw_conf > min_keypoint_conf).sum())
    log_fn(f"[pose-track] {video.name} {start_sec:.1f}s+{dur_sec:.1f}s: "
           f"{processed}/{len(sample_frames)} sampled, "
           f"{covered}/{n} frames with head, {yaw_covered}/{n} with yaw "
           f"(stride={sample_stride}, {elapsed:.1f}s)")

    ht = HeadTrack(
        head_xy_norm=head_xy,
        head_conf=head_conf,
        yaw_proxy=yaw_proxy,
        yaw_conf=yaw_conf,
        n_frames=n,
        probe_elapsed_sec=float(elapsed),
    )
    _save_cache(video, start_sec, dur_sec, ht, log_fn=log_fn)
    return ht


def session_head_y_anchor(head_tracks: "list[HeadTrack]",
                          min_conf: float = 0.30) -> float | None:
    """Session-level head Y anchor: the median of per-source median head-y
    (on confident frames). Used by merge_sources.py so every source puts
    its dancer's head at the same vertical fraction in the output.

    Returns None if no source has enough confident frames (< 30 each).
    """
    per_source_medians: list[float] = []
    for ht in head_tracks:
        if ht is None:
            continue
        mask = ht.head_conf > min_conf
        if mask.sum() < 30:
            continue
        per_source_medians.append(float(np.median(ht.head_xy_norm[mask, 1])))
    if not per_source_medians:
        return None
    return float(np.median(per_source_medians))


def per_source_yaw_median(ht: "HeadTrack",
                          min_conf: float = 0.30) -> float | None:
    """Median yaw proxy for a source, on frames where all three of
    nose/l_eye/r_eye were confidently detected.

    Returns None if fewer than 30 confident yaw frames in the track.
    """
    if ht is None:
        return None
    mask = ht.yaw_conf > min_conf
    if mask.sum() < 30:
        return None
    return float(np.median(ht.yaw_proxy[mask]))


def yaw_bucket(yaw: "float | None",
               frontal_thresh: float = 0.10) -> str:
    """Bucket a yaw_proxy median into {"frontal", "left", "right", "unknown"}.

    `frontal_thresh` is a signed margin on the dimensionless yaw_proxy.
    Empirically: raw ≈ 0 for a face staring into the lens; raw ≈ ±0.3-0.5
    for a 3/4 turn; raw → ±1.0 for full profile (eye span collapses).

    - `frontal`  |yaw| < 0.10      (within ~5° of lens)
    - `right`    yaw < -0.10       (subject turning to subject's right)
    - `left`     yaw > +0.10       (subject turning to subject's left)

    Sign convention: positive yaw means the nose has migrated toward the
    subject's-left eye — i.e. the camera is catching more of the subject's
    left cheek. For a centered dancer facing the main stage camera, a
    fancam placed STAGE-RIGHT (viewer-left) will see more of the subject's
    left profile → yaw_proxy > 0 on average.
    """
    if yaw is None:
        return "unknown"
    if yaw > frontal_thresh:
        return "left"
    if yaw < -frontal_thresh:
        return "right"
    return "frontal"


def session_yaw_bucket(head_tracks: "list[HeadTrack]",
                        min_conf: float = 0.30,
                        frontal_thresh: float = 0.10
                        ) -> "list[tuple[str, float | None]]":
    """Per-source (bucket, median_yaw) list, same order as `head_tracks`.

    For sources where we can't estimate yaw (v1 cache, no confident
    frames), returns ("unknown", None).
    """
    out: list[tuple[str, float | None]] = []
    for ht in head_tracks:
        med = per_source_yaw_median(ht, min_conf=min_conf)
        out.append((yaw_bucket(med, frontal_thresh=frontal_thresh), med))
    return out
