"""Per-frame person tracking + face ID + smooth crop trajectory.

Used by match_video.extract_clip when --track-crop is on to produce a 9:16
portrait crop that follows the target member through a (usually 16:9) source.

Pipeline (v2 — per-frame face detect + multi-track + BoT-SORT):
  1. `compute_tracks(video, start, dur)`  — YOLOv8 + BoT-SORT on the segment.
     BoT-SORT adds sparse-optical-flow camera motion compensation on top of
     ByteTrack's IOU matching, which reduces ID switches in handheld fancams
     where the camera pans/shakes.
  2. `identify_target_tracks(...)`         — SAMPLE frames across the segment,
     detect ALL faces on the full frame at det_size=1024 (one SCRFD call per
     sampled frame, not per track), score each detected face against the
     reference pool, then assign each face to the YOLO track whose bbox
     contains it. Returns a list of matching track IDs (sorted by score
     desc). A featured member split into several tracks by a cross-edit is
     correctly recovered as multiple tids — the cropper then switches
     between them per frame.
  3. `crop_trajectory(...)`                — per-output-frame normalized
     centre (cx, cy) with EMA smoothing and last-known freeze. With a list
     of target tids, the highest-confidence tid that has an active bbox
     at that frame is chosen.

Face library layout (shared with DanceMashup via symlink):
    face_library/<Group>/<Member_Name>/embeddings.npy   (N x 512, pooled)
    face_library/<Group>/<Member_Name>/embedding.npy    (512, single ref)
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------- member-name matching ----------

def _name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_member_dir(face_root: Path, group: str, member_lat: str,
                       member_han: str = "") -> Optional[Path]:
    """Find face_library/<group>/<member> directory matching the given member
    name from groups.yaml (which may not exactly match the folder name)."""
    gdir = face_root / group
    if not gdir.exists():
        return None
    want = _name_key(member_lat)
    for d in gdir.iterdir():
        if not d.is_dir():
            continue
        have = _name_key(d.name)
        # Substring in either direction handles "ANYUJIN"↔"An_Yujin",
        # "Naoi Rei"↔"Rei", etc.
        if want and (want == have or want in have or have in want):
            return d
    return None


def load_reference_pool(member_dir: Path) -> Optional[np.ndarray]:
    """Return N×512 L2-normalised reference embeddings, or None."""
    pool = member_dir / "embeddings.npy"
    single = member_dir / "embedding.npy"
    arr = None
    if pool.exists():
        arr = np.load(pool)
    elif single.exists():
        arr = np.load(single)[None, :]
    if arr is None or arr.size == 0:
        return None
    arr = arr.astype(np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    return arr


def load_peer_negatives(face_root: Path, group: str,
                        exclude_member_dir: Path) -> Optional[np.ndarray]:
    """Concat all peer members' pooled embeddings as negatives."""
    gdir = face_root / group
    if not gdir.exists():
        return None
    negs = []
    for d in gdir.iterdir():
        if not d.is_dir() or d == exclude_member_dir:
            continue
        ref = load_reference_pool(d)
        if ref is not None:
            negs.append(ref)
    if not negs:
        return None
    return np.concatenate(negs, axis=0)


def load_peer_gallery(face_root: Path, group: str,
                      exclude_member_dir: Path
                      ) -> tuple[Optional[np.ndarray], list[str], list[int]]:
    """Labelled variant of `load_peer_negatives`.

    Returns (stacked_embeddings, member_names, member_row_counts) such that
    `member_names[i]` owns `member_row_counts[i]` consecutive rows in the
    stacked embedding matrix, in the same order as the iterator. Used by
    `identify_target_tracks` for gallery-style argmax gating — a face is
    only accepted as target if target has the best similarity; if any peer
    member is closer, the face is ignored. Also lets us log "rejected:
    more similar to Yujin (0.57) than Wonyoung (0.52)" for debugging.
    """
    gdir = face_root / group
    if not gdir.exists():
        return None, [], []
    negs = []
    names: list[str] = []
    counts: list[int] = []
    for d in sorted(gdir.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or d == exclude_member_dir:
            continue
        ref = load_reference_pool(d)
        if ref is not None:
            negs.append(ref)
            names.append(d.name)
            counts.append(ref.shape[0])
    if not negs:
        return None, [], []
    return np.concatenate(negs, axis=0), names, counts


# ---------- YOLO + BoT-SORT ----------

_YOLO = None


def _get_yolo(weights: Path) -> "object":
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO  # noqa: PLC0415
        _YOLO = YOLO(str(weights))
    return _YOLO


@dataclass
class SegmentMeta:
    width: int
    height: int
    fps: float
    start_sec: float
    dur_sec: float
    n_frames: int  # frames inside the segment


@dataclass
class TrackedSegment:
    """Full intermediate state of one tracked source.

    Returned by `tracked_segment()`. The multi-source merge pipeline uses
    `tracks` + `target_ids` + `meta` to compute coverage masks for each
    source at song-time granularity, and uses `cxcy` + `h_norm` later
    when actually extracting crops from the chosen source. `plan_crop`
    (the historical 4-tuple API) is a thin shim over this.
    """
    video: Path
    start_sec: float
    dur_sec: float
    meta: SegmentMeta
    tracks: "dict[int, np.ndarray]"
    target_ids: list[int]
    mode: str
    cxcy: np.ndarray
    h_norm: Optional[np.ndarray]
    # Face-ID confidence per accepted track id (same keys as target_ids).
    # Empty when mode != "face" — fallback modes ("largest", "center",
    # "center_low_cov") don't have per-tid confidence. `quality_mask()`
    # defaults to 0.5 when a tid isn't in this map.
    target_scores: dict[int, float] = field(default_factory=dict)
    # Per-tid peer-dominance = target_score - peer_agg, computed in
    # `identify_target_tracks` and forwarded here so `quality_mask` can
    # apply a dominance-aware SOFT gate: a source whose face_score is
    # sub-threshold but whose dominance is strongly positive is still a
    # confident classifier decision (the classifier is sure it's the
    # target — it just doesn't have many tight-face samples). Gating
    # that case out loses genuine coverage. Empty alongside target_scores.
    target_dominance: dict[int, float] = field(default_factory=dict)

    def coverage_mask(self, step_sec: float = 1.0) -> np.ndarray:
        """Boolean array at `step_sec` granularity: True when any target
        track has a bbox somewhere within that time bucket of the segment.

        Index 0 = [start_sec, start_sec+step_sec), index 1 = next, etc.
        Useful for cross-source merging decisions without carrying around
        per-frame structures.
        """
        n_buckets = max(1, int(np.ceil(self.dur_sec / step_sec)))
        out = np.zeros(n_buckets, dtype=bool)
        if not self.target_ids or self.meta.fps <= 0:
            return out
        frames_per_bucket = step_sec * self.meta.fps
        if frames_per_bucket <= 0:
            return out
        for tid in self.target_ids:
            rows = self.tracks.get(tid)
            if rows is None:
                continue
            for r in rows:
                frame_idx = int(r[0])
                bucket = int(frame_idx // frames_per_bucket)
                if 0 <= bucket < n_buckets:
                    out[bucket] = True
        return out

    def quality_mask(self, step_sec: float = 1.0,
                     min_face_conf: float = 0.0,
                     soft_min_face_conf: float = 0.0,
                     soft_dominance_margin: float = 0.08) -> np.ndarray:
        """Continuous float[0,1] array at `step_sec` granularity.

        For each time bucket, compute a quality score combining:
          - presence (target visible in the bucket),
          - bbox size relative to frame height (closer shots score higher),
          - centerness (well-composed shots score higher),
          - face-ID confidence of the tid(s) contributing.

        A bucket with the target visible for the whole step, at medium
        framing, centered, with mediocre face-ID scores ≈ 0.5. A bucket
        with a close-up, well-centered, high-face-ID shot approaches 1.0.
        An empty bucket = 0.0. This is what the merge planner uses to
        pick *which* source to use per bucket — not merely "whoever has
        the person visible", but "whoever has them best-framed".

        `min_face_conf` — merge-mode confidence gate. If the best face-ID
        score among `target_ids` is below this threshold, return an all-
        zero mask so the merge planner will not pick this source. This
        catches the "0.45 = wrong peer member" case: raising sim_thresh
        globally would starve real-but-weakly-scored sources, so we
        instead selectively gate *usage* in merge (where we have other
        sources to fall back to) without changing single-source behavior.

        `soft_min_face_conf` / `soft_dominance_margin` — dominance-aware
        soft path. If the hard `min_face_conf` gate would reject this
        source BUT the best target_score is still >= `soft_min_face_conf`
        AND its peer-dominance is >= `soft_dominance_margin`, the source
        is admitted anyway. Rationale: a sub-threshold score with strong
        peer-dominance means the classifier is sure which member it's
        looking at (big gap over the next-best peer) — it just didn't
        see many tight-face frames. The hard gate's job is to catch
        "0.45 face_score might be a peer"; dominance answers that
        question directly. With `soft_dominance_margin` > the identify-
        time `peer_dominance_margin`, we only unlock sources whose
        dominance is comfortably above the admission threshold. Set
        `soft_min_face_conf=0.0` to disable the soft path.
        """
        n_buckets = max(1, int(np.ceil(self.dur_sec / step_sec)))
        out = np.zeros(n_buckets, dtype=np.float32)
        if not self.target_ids or self.meta.fps <= 0:
            return out
        if min_face_conf > 0.0 and self.target_scores:
            top = max(self.target_scores.values())
            if top < min_face_conf:
                # Soft path: allow sub-threshold face_score through when
                # peer-dominance is strong. The top-scoring tid's
                # dominance is what unlocks admission (it's also the
                # tid that drives crop_trajectory's primary anchor).
                top_tid = max(self.target_scores.items(),
                              key=lambda kv: kv[1])[0]
                top_dom = float(self.target_dominance.get(top_tid, 0.0))
                soft_ok = (soft_min_face_conf > 0.0
                           and top >= soft_min_face_conf
                           and top_dom >= soft_dominance_margin)
                if not soft_ok:
                    return out
        frames_per_bucket = step_sec * self.meta.fps
        if frames_per_bucket <= 0:
            return out
        W = float(self.meta.width) or 1.0
        H = float(self.meta.height) or 1.0

        # Per-frame best score across contributing tids. Using best-per-
        # frame (not sum) avoids double-counting when BoT-SORT fragmented
        # the same person into multiple simultaneous ids.
        frame_best: dict[int, float] = {}
        for tid in self.target_ids:
            rows = self.tracks.get(tid)
            if rows is None:
                continue
            # Map face-ID score (typically 0.42-0.72) onto [0,1].
            face_conf = float(self.target_scores.get(tid, 0.5))
            face_w = max(0.0, min(1.0, (face_conf - 0.40) / 0.30))
            for r in rows:
                fi = int(r[0])
                x1, y1, x2, y2 = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                h_ratio = max(0.0, (y2 - y1) / H)
                # Scale: h_ratio = 0.6 (subject fills most of frame) → 1.0
                h_quality = min(1.0, h_ratio / 0.6)
                cx = (x1 + x2) * 0.5 / W
                # Centerness: 1.0 at cx=0.5, 0.0 at cx∈{0,1}. Now a
                # *multiplier* so an edge-of-frame bbox (cx=0.1) gets
                # heavily penalised — previously it was a +0.15 additive
                # contribution, which barely moved the ranking and let
                # the planner pick sources where the target was visible
                # but hopelessly off-centre (t=36 "Wonyoung cut off left"
                # failure, 2026-04-23).
                centerness = max(0.0, 1.0 - 2.0 * abs(cx - 0.5))
                # 0.5 baseline for "visible", plus quality bonuses for
                # close-up + face-ID confidence (additive), then an
                # overall centerness multiplier that caps output at
                # 0.5× when the subject is at the edge.
                base = (0.50
                        + 0.25 * h_quality
                        + 0.25 * face_w)
                center_mult = 0.5 + 0.5 * centerness
                score = base * center_mult
                if score > frame_best.get(fi, -1.0):
                    frame_best[fi] = score

        # Per-bucket average over the bucket's frame slots. Normalizing by
        # frames_per_bucket (not frames-with-target) means a bucket with
        # half-coverage scores ~0.5× a bucket with full-coverage at equal
        # per-frame quality — partial visibility penalizes correctly.
        sums = np.zeros(n_buckets, dtype=np.float64)
        counts = np.zeros(n_buckets, dtype=np.float64)
        for fi, s in frame_best.items():
            bucket = int(fi // frames_per_bucket)
            if 0 <= bucket < n_buckets:
                sums[bucket] += s
                counts[bucket] += 1.0
        out = (sums / max(1.0, frames_per_bucket)).astype(np.float32)
        # Continuity penalty: a bucket where the target is present for
        # only a handful of frames (e.g. camera passes through momentarily)
        # gets downscaled relative to a bucket with stable presence. This
        # prevents planner from choosing a source whose target visibility
        # is a 3-frame flicker over another with continuous presence —
        # the flickery source's chosen bucket would then render mostly
        # frozen-crop frames where the target isn't actually visible.
        #
        # Penalty curve: ≥50% bucket-coverage → no penalty; linear to 0.3×
        # at 0% coverage. Tunable; 0.5 empirically catches the t=36 case
        # (Wonyoung visible ~10/30 frames in the flickery source).
        MIN_CONT_FRAC = 0.5
        MIN_CONT_SCALE = 0.3
        cov_frac = (counts / max(1.0, frames_per_bucket)).astype(np.float32)
        cont_scale = np.where(
            cov_frac >= MIN_CONT_FRAC,
            np.float32(1.0),
            (MIN_CONT_SCALE
             + (1.0 - MIN_CONT_SCALE) * cov_frac / MIN_CONT_FRAC),
        ).astype(np.float32)
        out = out * cont_scale
        return np.clip(out, 0.0, 1.0)


def compute_tracks(video: Path, start_sec: float, dur_sec: float,
                   weights: Path, conf: float = 0.3, iou: float = 0.5,
                   imgsz: int = 640, device: str = "cpu",
                   tracker: str = "botsort.yaml",
                   ) -> tuple[dict[int, np.ndarray], SegmentMeta]:
    """Run YOLOv8 + BoT-SORT on [start_sec, start_sec+dur_sec].

    BoT-SORT extends ByteTrack with camera-motion compensation via sparse
    optical flow — which is the common failure mode for handheld fancams,
    stage-cam pans, and aggressive edit cuts. ReID is left off (the default
    `with_reid: False`) because the face-ID pass already provides identity
    anchoring downstream.

    Returns:
      tracks: {track_id: float32 array (N,5) = [frame_idx_in_segment, x1,y1,x2,y2]}
      meta:   SegmentMeta
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_in_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = int(round(start_sec * fps))
    n_frames = int(round(dur_sec * fps))
    n_frames = max(1, min(n_frames, total_in_vid - start_frame))

    model = _get_yolo(weights)

    # ultralytics.track doesn't support start/end offsets, so feed a generator
    # of frames we already seeked to.
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    per_track: dict[int, list] = {}
    fi = 0
    try:
        while fi < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # track() on a single frame, with persist=True so tracker IDs
            # carry across calls.
            r = model.track(
                source=frame, persist=True, stream=False, verbose=False,
                classes=[0], conf=conf, iou=iou,
                tracker=tracker, device=device, imgsz=imgsz,
            )[0]
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                xyxy = r.boxes.xyxy.cpu().tolist()
                for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                    per_track.setdefault(int(tid), []).append((fi, x1, y1, x2, y2))
            fi += 1
    finally:
        cap.release()
        # Reset model tracker state so the next clip starts fresh.
        try:
            model.predictor.trackers[0].reset()  # type: ignore[attr-defined]
        except Exception:
            pass

    tracks = {tid: np.asarray(rows, dtype=np.float32)
              for tid, rows in per_track.items()}
    meta = SegmentMeta(width=W, height=H, fps=fps,
                       start_sec=start_sec, dur_sec=dur_sec, n_frames=fi)
    return tracks, meta


# ---------- face ID scoring ----------

_FACE_APP = None


def _get_face_app(det_size: int = 1024):
    """Face app singleton. We only detect on full frames now (per-frame face
    detect), so det_size=1024 is sized for 1920x1080 sources — that gets us
    ~34px effective scale per 60px face bbox, enough for SCRFD's landmarks."""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis  # noqa: PLC0415
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        _FACE_APP = app
    return _FACE_APP


def identify_target_tracks(video: Path, start_sec: float, fps: float,
                           tracks: dict[int, np.ndarray],
                           refs: np.ndarray,
                           negs: Optional[np.ndarray] = None,
                           n_samples: int = 40,
                           sim_thresh: float = 0.42,
                           min_face_px: float = 20.0,
                           min_det_score: float = 0.40,
                           min_valid_samples: int = 2,
                           topk: int = 5,
                           penalty_thresh: float = 0.42,
                           penalty_weight: float = 0.8,
                           gallery_gate: bool = True,
                           gallery_margin: float = 0.0,
                           peer_dominance_margin: float = 0.05,
                           peer_names: Optional[list[str]] = None,
                           peer_counts: Optional[list[int]] = None,
                           ) -> list[tuple[int, float, float]]:
    """Per-frame face detection + track assignment (multi-track output).

    For each of `n_samples` evenly-spaced frames:
      1. Run SCRFD once on the FULL frame (and its horizontal flip — that
         handles "Dance Practice (Mirrored)" uploads).
      2. For every detected face, compute similarity to the target refs,
         with peer-negative penalty.
      3. Assign each face to the YOLO track whose bbox contains the face
         center (smallest-area containing bbox wins when several overlap).
      4. A track accumulates one score sample per frame it hosts an
         identified face on.

    Returns list of (track_id, score, dominance) sorted by score desc,
    filtered to `score >= sim_thresh`. `dominance = score - peer_agg`,
    which the merge planner uses for a dominance-aware soft gate (see
    `quality_mask`): a sub-threshold score with strong peer dominance
    still indicates the classifier knows it's the target, just with
    fewer tight-face samples. Multiple tids can be returned — this is
    the fix for cross-edits where the same member is split into several
    ByteTrack fragments by hard cuts; all those fragments will be
    accepted here, and `crop_trajectory` will smoothly switch between
    them per-frame.

    Memory: one SCRFD call per sampled frame, vs. the old "samples_per_track
    × tracks" path which produced 30*20 = 600 detect calls on stage-mixes.
    This scales with segment length, not track count.
    """
    try:
        app = _get_face_app()
    except Exception as e:
        print(f"[track] face app unavailable: {e}", flush=True)
        return []

    if not tracks:
        return []

    track_lengths = {tid: len(rows) for tid, rows in tracks.items()}

    # Build {frame_idx -> [(tid, (x1,y1,x2,y2))]} for point-in-bbox lookup
    by_frame_tracks: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    n_segment_frames = 0
    for tid, rows in tracks.items():
        for r in rows:
            fi = int(r[0])
            by_frame_tracks.setdefault(fi, []).append(
                (tid, (float(r[1]), float(r[2]), float(r[3]), float(r[4])))
            )
            if fi + 1 > n_segment_frames:
                n_segment_frames = fi + 1
    if n_segment_frames == 0:
        return []

    # Evenly-spaced frame indices; intersect with frames that actually have
    # YOLO detections (no point detecting faces on an empty frame).
    raw_samples = np.linspace(0, n_segment_frames - 1,
                              min(n_samples, n_segment_frames)).astype(int)
    sample_indices = [int(fi) for fi in np.unique(raw_samples)
                      if int(fi) in by_frame_tracks]
    if not sample_indices:
        # Fallback: try all frames that have any track detection
        sample_indices = sorted(by_frame_tracks.keys())[:n_samples]
    if not sample_indices:
        print("[track] no frames with detections to sample", flush=True)
        return []

    print(f"[track] per-frame face-ID: sampling {len(sample_indices)} frames "
          f"across {n_segment_frames}-frame segment (det_size=1024)",
          flush=True)

    per_track_scores: dict[int, list[float]] = {}
    # Parallel to per_track_scores — for every kept face (target >= peer),
    # record the MAX peer similarity and which peer row was the argmax.
    # Used for per-track peer-dominance gating: if a track's aggregate
    # peer score is close to its target score (even though each kept face
    # passed the per-face gate), the track is probably a lookalike peer
    # that's consistently "almost Wonyoung but actually Yujin". 2026-04-23
    # showed tid=136 aggregating to target=0.518 with peer=0.485 → would
    # have been rejected with a 0.05 dominance margin.
    per_track_peer_scores: dict[int, list[float]] = {}
    per_track_peer_idxs: dict[int, list[int]] = {}
    total_face_dets = 0
    # Counter for gallery-gate rejections (faces discarded because a peer
    # reference scored higher than the target). Capped verbose logging
    # so we don't spam the console on a 40-sample run.
    gallery_rejections = {"total": 0}
    cap = cv2.VideoCapture(str(video))
    start_frame = int(round(start_sec * fps))
    W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    try:
        for idx, fi in enumerate(sample_indices):
            active = by_frame_tracks.get(fi, [])
            if not active:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + fi)
            ok, frame = cap.read()
            if not ok:
                continue

            # Run detection on original + h-flipped; keep the best score
            # per track per frame (so flipping doesn't double-count).
            best_per_track_this_frame: dict[int, float] = {}
            best_peer_this_frame: dict[int, tuple[float, int]] = {}
            for frame_variant in (frame, cv2.flip(frame, 1)):
                flipped = frame_variant is not frame
                try:
                    faces = app.get(frame_variant)
                except Exception as e:
                    print(f"[track] SCRFD error on frame {fi}: {e}", flush=True)
                    continue
                if not faces:
                    continue
                total_face_dets += len(faces)
                for f in faces:
                    fx1, fy1, fx2, fy2 = f.bbox
                    face_w = float(fx2 - fx1)
                    det = float(getattr(f, "det_score", 1.0))
                    if face_w < min_face_px or det < min_det_score:
                        continue
                    # Face center back in original (non-flipped) coords
                    fcx = (fx1 + fx2) / 2.0
                    fcy = (fy1 + fy2) / 2.0
                    if flipped:
                        fcx = W_full - fcx

                    # Assign to the track whose bbox contains this center
                    # (smallest-area containing bbox wins — most specific).
                    candidates = []
                    for tid, (x1, y1, x2, y2) in active:
                        if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                            area = (x2 - x1) * (y2 - y1)
                            candidates.append((area, tid))
                    if not candidates:
                        continue
                    candidates.sort()
                    tid_assigned = candidates[0][1]

                    emb = f.normed_embedding
                    pos = float(np.max(refs @ emb))
                    neg = 0.0
                    peer_argmax_idx = -1
                    if negs is not None:
                        neg_sims = negs @ emb
                        neg = float(np.max(neg_sims))
                        peer_argmax_idx = int(np.argmax(neg_sims))
                        # Gallery argmax gate — if any peer reference is
                        # closer than the best target reference (plus an
                        # optional margin), this face is almost certainly
                        # the peer, not the target. Skip entirely rather
                        # than scoring low — a sampled peer face must not
                        # contaminate per_track_scores[tid].
                        if gallery_gate and neg > pos + gallery_margin:
                            # Resolve which peer member scored highest,
                            # for logging when the rejection matters
                            # (close calls). Uses row ranges built from
                            # peer_counts.
                            if gallery_rejections["total"] < 6:
                                peer_label = "peer"
                                if (peer_names and peer_counts
                                        and len(peer_names) == len(peer_counts)):
                                    idx = int(np.argmax(neg_sims))
                                    acc = 0
                                    for name, c in zip(peer_names, peer_counts):
                                        if acc <= idx < acc + c:
                                            peer_label = name
                                            break
                                        acc += c
                                print(f"[track] face@fi={fi} tid={tid_assigned} "
                                      f"rejected by gallery: "
                                      f"{peer_label}={neg:.3f} > "
                                      f"target={pos:.3f}",
                                      flush=True)
                            gallery_rejections["total"] += 1
                            continue
                        if neg > penalty_thresh:
                            pos -= penalty_weight * (neg - penalty_thresh)

                    prev = best_per_track_this_frame.get(tid_assigned)
                    if prev is None or pos > prev:
                        best_per_track_this_frame[tid_assigned] = pos
                        # Remember the peer score associated with the
                        # WINNING face for this tid this frame, so peer
                        # aggregation lines up with target aggregation.
                        best_peer_this_frame[tid_assigned] = (neg, peer_argmax_idx)

            for tid, s in best_per_track_this_frame.items():
                per_track_scores.setdefault(tid, []).append(s)
                if tid in best_peer_this_frame:
                    npeer, pidx = best_peer_this_frame[tid]
                    per_track_peer_scores.setdefault(tid, []).append(npeer)
                    per_track_peer_idxs.setdefault(tid, []).append(pidx)
            # Release per-frame buffers; otherwise the accumulated ONNX
            # workspace across 40 SCRFD calls at det=1024 pressures the
            # 16GB Mac mini into jetsam.
            del frame
            gc.collect()
    finally:
        cap.release()

    print(f"[track] total face detections across sampled frames: "
          f"{total_face_dets}", flush=True)
    if gallery_rejections["total"] > 0:
        print(f"[track] gallery-gate rejected {gallery_rejections['total']} "
              f"face(s) more similar to peer than target", flush=True)

    # Aggregate and filter
    # (tid, score, sample_count, dominance) — dominance carried through
    # so downstream quality_mask can apply dominance-aware soft gating
    # for sources whose aggregate face_score is just below min_face_conf
    # but whose peer-dominance is strong (unambiguous classifier decision).
    ranked: list[tuple[int, float, int, float]] = []
    for tid, sims in per_track_scores.items():
        if len(sims) < min_valid_samples:
            print(f"[track] tid={tid} len={track_lengths.get(tid, 0)} "
                  f"samples={len(sims)} (<{min_valid_samples}) — skip",
                  flush=True)
            continue
        sims_sorted = sorted(sims, reverse=True)
        k = min(topk, len(sims_sorted))
        score = float(np.mean(sims_sorted[:k]))

        # Peer-dominance gate: aggregate peer similarity over the same
        # kept faces, using the same topk ranking. If target score is
        # not clearly above peer score (by `peer_dominance_margin`), the
        # tid is probably a lookalike member and should be rejected.
        # Without this gate, a sequence of near-ties (target 0.52, peer
        # 0.50) all pass the per-face gallery_gate, and the aggregate
        # scores high while actually representing the wrong member.
        peer_sims = per_track_peer_scores.get(tid, [])
        peer_agg = float(np.mean(sorted(peer_sims, reverse=True)[:k])) \
            if len(peer_sims) >= min_valid_samples else 0.0
        dominance = score - peer_agg
        if (peer_sims and peer_dominance_margin > 0.0
                and dominance < peer_dominance_margin):
            # Resolve the most-confused peer for logging.
            peer_label = "peer"
            pidxs = per_track_peer_idxs.get(tid, [])
            if (peer_names and peer_counts and pidxs
                    and len(peer_names) == len(peer_counts)):
                # Mode-like choice: most common argmax across kept faces.
                counts_per_peer: dict[str, int] = {}
                for pidx in pidxs:
                    acc = 0
                    for name, c in zip(peer_names, peer_counts):
                        if acc <= pidx < acc + c:
                            counts_per_peer[name] = counts_per_peer.get(name, 0) + 1
                            break
                        acc += c
                if counts_per_peer:
                    peer_label = max(counts_per_peer.items(),
                                      key=lambda kv: kv[1])[0]
            print(f"[track] tid={tid} len={track_lengths.get(tid, 0)} "
                  f"samples={len(sims)} target={score:.3f} "
                  f"{peer_label}={peer_agg:.3f} dominance={dominance:+.3f} "
                  f"< {peer_dominance_margin:.2f} — REJECT (peer confused)",
                  flush=True)
            continue

        print(f"[track] tid={tid} len={track_lengths.get(tid, 0)} "
              f"samples={len(sims)} face_score={score:.3f} "
              f"(peer={peer_agg:.3f} dom={dominance:+.3f})",
              flush=True)
        ranked.append((tid, score, len(sims), dominance))

    if not ranked:
        print("[track] no track had enough face samples", flush=True)
        return []

    ranked.sort(key=lambda r: -r[1])
    top_tid, top_score, _, _ = ranked[0]
    if top_score < sim_thresh:
        print(f"[track] reject: best_score={top_score:.3f} < thresh={sim_thresh:.3f}",
              flush=True)
        return []

    # Multi-track acceptance: the target person in a cross-edit is usually
    # split into several ByteTrack/BoT-SORT fragments by hard cuts and by
    # face-obscuring moments. Accept the top scorer PLUS any runner-ups
    # within `cluster_range` of the top — those are almost always the same
    # person in another fragment. Cap at `max_tracks` to avoid noisy crops
    # when the scene is ambiguous.
    #
    # Confidence-adaptive cluster width: on group stages the peer-negative
    # penalty doesn't fully separate same-group members (similar outfits /
    # lighting / styling), so the top 3-4 tracks cluster within 0.06-0.08
    # of each other — and they're DIFFERENT members, not the same member
    # fragmented. Rule of thumb from real runs:
    #   top >= 0.60  — classifier is confident, same-person multi-fragment
    #                  case is typical → wide 0.08 window.
    #   top >= 0.55  — borderline; allow at most a couple of near-identical
    #                  scorers (0.03 window).
    #   top <  0.55  — weak match; DO NOT multi-accept. Single track only,
    #                  otherwise we jump between peer members and the clip
    #                  visibly follows the wrong person.
    if top_score >= 0.60:
        cluster_range = 0.08
    elif top_score >= 0.55:
        cluster_range = 0.03
    else:
        cluster_range = 0.0  # single-track mode
    max_tracks: int = 5 if top_score >= 0.55 else 1
    floor = max(sim_thresh - 0.03, top_score - cluster_range)

    # Temporal-overlap guard: two tracks that are simultaneously visible on
    # the same frame MUST be different people (one body can't be in two
    # places). Reject runner-ups that heavily overlap with already-accepted
    # tracks — this is the principled fix for group stages where multiple
    # members all score 0.49-0.57 against the target refs and would
    # otherwise be accepted as "same-person fragments". 15% is tight
    # enough to catch co-visibility but tolerant of brief BoT-SORT ID
    # switches where one fragment's first frame overlaps the previous
    # fragment's last frame.
    MAX_OVERLAP = 0.15

    def _frame_set(tid: int) -> set[int]:
        return set(int(r[0]) for r in tracks.get(tid, []))

    accepted: list[tuple[int, float, float]] = []
    accepted_frames: list[set[int]] = []
    rejected_overlap: list[tuple[int, float, float, int]] = []
    for tid, s, _, dom in ranked:
        if s < floor:
            continue
        if accepted:  # top track always admitted
            frames = _frame_set(tid)
            if frames:
                worst_overlap = 0.0
                worst_against = -1
                for atid, af in zip((a[0] for a in accepted), accepted_frames):
                    if not af:
                        continue
                    inter = len(frames & af)
                    ov = inter / max(1, min(len(frames), len(af)))
                    if ov > worst_overlap:
                        worst_overlap = ov
                        worst_against = atid
                if worst_overlap > MAX_OVERLAP:
                    rejected_overlap.append((tid, s, worst_overlap, worst_against))
                    continue
            accepted.append((tid, s, dom))
            accepted_frames.append(_frame_set(tid))
        else:
            accepted.append((tid, s, dom))
            accepted_frames.append(_frame_set(tid))
        if len(accepted) >= max_tracks:
            break

    for tid, s, ov, against in rejected_overlap:
        print(f"[track] tid={tid} score={s:.3f} rejected from multi-accept: "
              f"overlap={ov:.2f} with tid={against} (different person)",
              flush=True)

    print(f"[track] accepted {len(accepted)} track(s) "
          f"(top={top_score:.3f}, floor={floor:.3f}, cluster={cluster_range:.2f}): "
          + ", ".join(f"tid={t} score={s:.3f} dom={d:+.3f}"
                      for t, s, d in accepted),
          flush=True)
    return accepted


def identify_target_track(video: Path, start_sec: float, fps: float,
                          tracks: dict[int, np.ndarray],
                          refs: np.ndarray,
                          negs: Optional[np.ndarray] = None,
                          **kwargs) -> Optional[int]:
    """Backward-compat shim: returns only the top-scoring tid (or None)."""
    accepted = identify_target_tracks(video, start_sec, fps, tracks,
                                      refs, negs, **kwargs)
    # identify_target_tracks returns (tid, score, dominance) triples.
    return accepted[0][0] if accepted else None


def largest_area_track(tracks: dict[int, np.ndarray]) -> Optional[int]:
    """Fallback: track with largest cumulative bbox area * length."""
    if not tracks:
        return None
    scored = []
    for tid, rows in tracks.items():
        areas = (rows[:, 3] - rows[:, 1]) * (rows[:, 4] - rows[:, 2])
        scored.append((tid, float(areas.sum())))
    scored.sort(key=lambda kv: -kv[1])
    return scored[0][0]


# ---------- smooth crop trajectory ----------

def crop_trajectory(tracks: dict[int, np.ndarray], target_id: int,
                    meta: SegmentMeta, ema_alpha: float = 0.18,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Single-track trajectory. See `crop_trajectory_multi` for multi-track."""
    return crop_trajectory_multi(tracks, [target_id], meta, ema_alpha)


def crop_trajectory_multi(tracks: dict[int, np.ndarray],
                          target_ids: list[int],
                          meta: SegmentMeta,
                          ema_alpha: float = 0.18,
                          gap_hold_sec: float = 1.5,
                          gap_recover_sec: float = 2.5,
                          neutral_cy: float = 0.42,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Multi-track trajectory.

    `target_ids` is ordered by face-score desc. For each output frame, the
    highest-priority (first-in-list) track that has an active bbox at that
    frame wins. If none of the targets have a bbox that frame, freeze at
    last-known position. This correctly handles cross-edits where the
    target member is split into several ByteTrack fragments by hard cuts:
    at the cut boundary we jump from tid_A's last frame to tid_B's first
    frame, with EMA damping the jump.

    Gap recovery is two-phase:

      1. HOLD: for the first `gap_hold_sec` seconds after the target
         goes absent, the anchor stays at the target's last-known
         position. Rationale — on group stages the target often merely
         walks a few positions off the current crop without leaving the
         full frame; pulling to centre prematurely means we crop AWAY
         from the target to show whichever other member happens to be
         in the stage centre (2026-04-23: this was the t=33/35/44/50
         failure mode on fdzojloPlFA, where neutral-centre meant cropping
         LIZ/Rei/Yujin when Wonyoung was still visible off-centre).

      2. LERP: after `gap_hold_sec` the anchor linearly lerps toward the
         neutral stage-centre position (0.5, neutral_cy) over the next
         `gap_recover_sec` seconds. Long absences (tid really left frame)
         end up centre-cropping the stage rather than locking onto the
         edge where the target was last seen.

    Total time to fully converge: `gap_hold_sec + gap_recover_sec`.
    EMA on top of the target position still smooths the transition so
    there's no visible pop at either phase boundary.

    Returns:
      cxcy   — (n_frames, 2) float32 normalised (cx, cy), EMA-smoothed.
      h_norm — (n_frames,) float32 normalised bbox height (raw, consumer
               smooths it).

    cy uses the head-bias formula (y1 + 0.35*h) to place the face in the
    upper third of the portrait crop.
    """
    n = meta.n_frames
    if not target_ids:
        return (np.tile([0.5, neutral_cy], (n, 1)).astype(np.float32),
                np.zeros(n, dtype=np.float32))

    # Pre-compute per-track per-frame (cx, cy, h) in normalised coords
    per_track_frames: dict[int, dict[int, tuple[float, float, float]]] = {}
    for tid in target_ids:
        rows = tracks.get(tid)
        if rows is None:
            continue
        lookup: dict[int, tuple[float, float, float]] = {}
        for r in rows:
            fi = int(r[0])
            x1, y1, x2, y2 = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            cx = (x1 + x2) / 2.0 / meta.width
            cy = (y1 + 0.35 * (y2 - y1)) / meta.height
            h = (y2 - y1) / meta.height
            lookup[fi] = (cx, cy, h)
        per_track_frames[tid] = lookup

    cxcy = np.zeros((n, 2), dtype=np.float32)
    h_norm = np.zeros(n, dtype=np.float32)
    neutral = np.array([0.5, neutral_cy], dtype=np.float32)
    smoothed = neutral.copy()
    last_cxcy = smoothed.copy()
    last_h = 0.0
    have_seen = False
    switches = 0
    last_tid: Optional[int] = None
    frames_by_tid: dict[int, int] = {}
    frames_absent = 0
    # Two-phase gap recovery (see docstring): hold at last-known for
    # `hold_frames`, then lerp to neutral over `recover_frames`. Measured
    # in frames so 30fps and 60fps sources behave identically in seconds.
    hold_frames = max(1, int(round(gap_hold_sec * max(1.0, meta.fps))))
    recover_frames = max(1, int(round(gap_recover_sec * max(1.0, meta.fps))))
    held_frames = 0
    recovered_frames = 0
    for fi in range(n):
        chosen: Optional[tuple[int, tuple[float, float, float]]] = None
        for tid in target_ids:
            pf = per_track_frames.get(tid, {}).get(fi)
            if pf is not None:
                chosen = (tid, pf)
                break
        if chosen is not None:
            tid, (cx, cy, h) = chosen
            if last_tid is not None and tid != last_tid:
                switches += 1
            last_tid = tid
            frames_by_tid[tid] = frames_by_tid.get(tid, 0) + 1
            last_cxcy = np.array([cx, cy], dtype=np.float32)
            last_h = h
            have_seen = True
            frames_absent = 0
            target = last_cxcy
        else:
            frames_absent += 1
            if not have_seen:
                target = smoothed
            elif frames_absent <= hold_frames:
                # HOLD phase — stay locked on the target's last-known
                # position so brief off-centre moments don't flip the
                # crop to neutral centre (which would show whoever
                # happens to be standing at stage middle).
                target = last_cxcy
                held_frames += 1
            else:
                # LERP phase — the target has been gone long enough that
                # freezing at their last bbox risks "stuck at edge" for
                # the rest of the clip. Transition toward neutral.
                pull = min(1.0,
                           (frames_absent - hold_frames) / float(recover_frames))
                target = last_cxcy * (1.0 - pull) + neutral * pull
                if pull >= 1.0:
                    recovered_frames += 1
        smoothed = ema_alpha * target + (1 - ema_alpha) * smoothed
        cxcy[fi] = smoothed
        h_norm[fi] = last_h

    if len(target_ids) > 1:
        dist = ", ".join(f"tid={t}:{frames_by_tid.get(t, 0)}" for t in target_ids)
        print(f"[track] multi-track coverage: {dist}; {switches} switches",
              flush=True)
    if held_frames > 0 or recovered_frames > 0:
        print(f"[track] gap-recover: {held_frames} frames held at last-known "
              f"(<{gap_hold_sec:.2f}s absence), "
              f"{recovered_frames} frames fully pulled to centre "
              f"(>{gap_hold_sec + gap_recover_sec:.2f}s absence)",
              flush=True)
    return cxcy, h_norm


# ---------- high-level helper ----------

def tracked_segment(video: Path, start_sec: float, dur_sec: float,
                    group: str, member_lat: str, member_han: str,
                    face_root: Path, yolo_weights: Path,
                    disable_low_cov_fallback: bool = False,
                    ) -> TrackedSegment:
    """Run tracking + ID + trajectory on one source segment and return the
    full intermediate state.

    This is the shared backend for both the historical single-source
    pipeline (`plan_crop`, which discards tracks/target_ids) and the
    multi-source merge pipeline (which needs them for coverage masks).

    `disable_low_cov_fallback` — when True, skip the "coverage < 35%
    means freeze at centre" guard. The guard was added for single-source
    extraction where a low-coverage trajectory freezes at the target's
    last-known position (usually near a frame edge), producing mostly
    background. In the merge pipeline the guard is counter-productive:
    the planner already selects per-bucket source on quality, so having
    each source keep its face-tracked trajectory is exactly what lets
    merge "follow the target when it's there, on whichever source has
    it". Defaults to False (preserves single-source behaviour).
    """
    tracks, meta = compute_tracks(video, start_sec, dur_sec, yolo_weights)
    print(f"[track] {len(tracks)} tracks in {meta.n_frames} frames "
          f"(BoT-SORT)", flush=True)

    target_ids: list[int] = []
    target_scores: dict[int, float] = {}
    target_dominance: dict[int, float] = {}
    mode = "center"
    mdir = resolve_member_dir(face_root, group, member_lat, member_han)
    if mdir:
        refs = load_reference_pool(mdir)
        if refs is not None:
            # Labelled gallery: per-peer-member embeddings + names for
            # argmax gating ("is this face more similar to Yujin than to
            # Wonyoung? skip it") and for richer rejection logs.
            negs, peer_names, peer_counts = load_peer_gallery(
                face_root, group, mdir)
            print(f"[track] face lib: {mdir.name} refs={len(refs)} "
                  f"peers={len(peer_names)} "
                  f"({', '.join(peer_names) if peer_names else '-'}) "
                  f"neg_rows={0 if negs is None else len(negs)}", flush=True)
            accepted = identify_target_tracks(video, start_sec, meta.fps,
                                              tracks, refs, negs,
                                              peer_names=peer_names,
                                              peer_counts=peer_counts)
            if accepted:
                target_ids = [tid for tid, _, _ in accepted]
                target_scores = {tid: float(sc) for tid, sc, _ in accepted}
                target_dominance = {tid: float(d) for tid, _, d in accepted}
                mode = "face"

    if not target_ids:
        fallback = largest_area_track(tracks)
        if fallback is not None:
            target_ids = [fallback]
            mode = "largest"

    if not target_ids:
        n = meta.n_frames
        return TrackedSegment(
            video=video, start_sec=start_sec, dur_sec=dur_sec, meta=meta,
            tracks=tracks, target_ids=[], mode="center",
            cxcy=np.tile([0.5, 0.5], (n, 1)).astype(np.float32),
            h_norm=None,
            target_scores={},
            target_dominance={},
        )

    cxcy, h_norm = crop_trajectory_multi(tracks, target_ids, meta)
    # Only expose the height trajectory to the cropper when we have a
    # confident face-ID. 'largest' fallback points at *some* performer but
    # not necessarily the requested member, so we don't want to zoom in.
    if mode != "face":
        h_norm = None

    # Low-coverage guard: when the target tracks only cover a small fraction
    # of the segment (typical of group-stage cross-edits where the camera
    # frequently cuts to other members), crop_trajectory_multi freezes at
    # the target's last-known position during the gaps. If that position is
    # near a frame edge, the resulting 9:16 crop shows mostly background
    # (LED walls, stage props) with the subject only partially visible on
    # one side — the "empty background" failure mode visible on group-stage
    # outputs. Safer to give up tracking and do a static centered full-
    # height crop: we don't see "the right person" continuously, but we at
    # least always see the middle of the stage where they usually are.
    if mode == "face" and target_ids:
        covered_frames: set[int] = set()
        for tid in target_ids:
            rows = tracks.get(tid)
            if rows is None:
                continue
            for r in rows:
                covered_frames.add(int(r[0]))
        coverage = len(covered_frames) / max(1, meta.n_frames)
        if coverage < 0.35:
            if disable_low_cov_fallback:
                print(f"[track] low target coverage ({coverage:.0%} < 35%) — "
                      f"fallback DISABLED (merge mode); keeping face-tracked "
                      f"trajectory so the merge planner can pick this source "
                      f"only on buckets where target is actually visible",
                      flush=True)
            else:
                print(f"[track] low target coverage ({coverage:.0%} < 35%) — "
                      f"falling back to static center crop to avoid frozen-"
                      f"at-edge frames",
                      flush=True)
                n = meta.n_frames
                cxcy = np.tile([0.5, 0.5], (n, 1)).astype(np.float32)
                h_norm = None
                mode = "center_low_cov"

    return TrackedSegment(
        video=video, start_sec=start_sec, dur_sec=dur_sec, meta=meta,
        tracks=tracks, target_ids=target_ids, mode=mode,
        cxcy=cxcy, h_norm=h_norm,
        target_scores=target_scores,
        target_dominance=target_dominance,
    )


def plan_crop(video: Path, start_sec: float, dur_sec: float,
              group: str, member_lat: str, member_han: str,
              face_root: Path, yolo_weights: Path,
              ) -> tuple[np.ndarray, Optional[np.ndarray], SegmentMeta, str]:
    """Backwards-compatible 4-tuple shim. New callers should use
    `tracked_segment()` instead.

    Returns (cxcy, h_norm, meta, mode):
      cxcy   — (n, 2) normalised centre trajectory
      h_norm — (n,) normalised bbox-height trajectory, OR None when we are
               not confident about the person's identity. Returning None
               tells `extract_clip_tracked` to fall back to a full-height
               9:16 crop.
      mode   — 'face' (face-ID succeeded), 'largest' (fallback to biggest
               track on a landscape shot, no ID), 'center' / 'center_low_cov'
               when tracking is skipped/degraded.
    """
    ts = tracked_segment(video, start_sec, dur_sec, group, member_lat,
                         member_han, face_root, yolo_weights)
    return ts.cxcy, ts.h_norm, ts.meta, ts.mode
