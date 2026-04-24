"""Multi-source fancam merge.

Takes a list of successfully-matched source videos (each with a known
audio alignment offset and a tracked-segment context for the target
member), and produces a single merged 9:16 clip that hops between sources
second-by-second to maximise target-on-screen time.

Design decisions (from discussion 2026-04-23):
  A = 2  ⇒ cross-event fill is allowed: if the primary (broadcaster, date)
          cluster has a gap where the target isn't visible, a secondary
          cluster can fill that second even though the stage cuts.
  B = self-cluster ⇒ sources whose broadcaster/date can't be parsed form
          singleton clusters, so they never pool with other parse-failures.

Pipeline:
  1. `cluster_by_event` from sources.py → primary cluster = highest total
     target coverage among all clusters.
  2. Per song-second bucket, pick a source that has the target visible:
        a. primary cluster, sources ranked by individual coverage
        b. any other cluster (cross-event fill)
        c. fall back to primary rank-0 (accept a gap — better than nothing)
  3. Collapse consecutive same-source picks into chunks; smooth 1-second
     isolated flips (keeps cut rate sane).
  4. Extract each chunk from its source via `extract_clip_tracked` using
     that source's own per-frame trajectory, with 0.5s post-roll for xfade.
  5. xfade-chain the chunks, mux canonical mp3 as audio, output single mp4.

`plan_merge` is pure and testable. `merge_clip` does the ffmpeg work.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from person_track import TrackedSegment, SegmentMeta
from sources import SourceMeta


@dataclass
class MergeSource:
    meta: SourceMeta          # broadcaster/date/path/video_id
    offset_sec: float         # song t=0 in this source's video timeline
    tracked: TrackedSegment   # trajectory + per-frame tracks for the clip window
    matched_title: str        # e.g. "After LIKE" (for logs)
    # M3: RTMPose-m head-center trajectory (optional). Populated by
    # merge_clip when called with use_pose=True. Absent / None when
    # pose mode is off; consumer must handle both paths.
    head: "Optional[object]" = None  # pose_track.HeadTrack


@dataclass
class MergeChunk:
    """One contiguous segment of the output pulled from a single source."""
    song_start: float    # inclusive, seconds in output timeline
    song_end: float      # exclusive
    src_idx: int         # index into sources list

    @property
    def dur(self) -> float:
        return max(0.0, self.song_end - self.song_start)


# ---------- beat detection ----------

_BEAT_THIS_CACHE = None  # lazy singleton; type: File2Beats | None


def _beat_this_model():
    """Load Beat This! once per process. The checkpoint download happens
    on first call; subsequent merges reuse the same instance."""
    global _BEAT_THIS_CACHE
    if _BEAT_THIS_CACHE is not None:
        return _BEAT_THIS_CACHE
    # `final0` is the released checkpoint. CPU is fine — Beat This! is
    # ~3s for a 3-min track on an M-series Mac, dwarfed by video
    # rendering. Keep dbn=False: the DBN post-processor adds a madmom
    # dependency (heavy, Cython, brittle on modern Python) for marginal
    # structural quality. Raw model output is already much better than
    # librosa on K-pop.
    from beat_this.inference import File2Beats  # noqa: PLC0415
    _BEAT_THIS_CACHE = File2Beats(checkpoint_path="final0",
                                   device="cpu", dbn=False)
    return _BEAT_THIS_CACHE


def detect_beats(audio_path: Path, sr: int = 22050,
                 log_fn=print) -> np.ndarray:
    """Return beat times (seconds) in the audio file.

    Primary backend: Beat This! (ISMIR 2024). Much better than librosa on
    modern pop — handles K-pop's 4-on-the-floor + half-time breakdowns +
    syncopated drops without the classic librosa "octave-halving" failure
    (calling 128 BPM as 64). Falls back to librosa when Beat This! isn't
    installed, so dev boxes without the ML deps still work.

    Used by `plan_merge` to snap chunk boundaries to the music's pulse.
    The K-pop editing convention is to cut 1-2 frames *before* each beat,
    so the new shot is already on screen when the downbeat lands; the
    caller supplies that lead time via `_snap_to_beat`.

    Returns empty array on any failure. Callers must treat "no beats" as
    "don't snap".
    """
    # --- Beat This! (preferred) ------------------------------------------
    try:
        model = _beat_this_model()
        t0 = time.time()
        beats_arr, downbeats_arr = model(str(audio_path))
        elapsed = time.time() - t0
        beats = np.asarray(beats_arr, dtype=np.float32)
        # Derive BPM from median inter-beat gap — good enough for the log
        # line, no need to wire downbeats through to the planner yet (the
        # snap logic treats all beats equally; M4 can revisit if we want
        # to bias cuts to downbeats).
        bpm_txt = ""
        if beats.size >= 2:
            median_gap = float(np.median(np.diff(beats)))
            if median_gap > 0:
                bpm_txt = f" at ~{60.0/median_gap:.1f} BPM"
        log_fn(f"[beat] Beat This! detected {len(beats)} beats"
               f"{bpm_txt} ({len(downbeats_arr)} downbeats, "
               f"{elapsed:.1f}s)")
        return beats
    except ImportError:
        log_fn("[beat] Beat This! not installed — falling back to librosa")
    except Exception as e:
        log_fn(f"[beat] Beat This! failed ({e!r}) — falling back to librosa")

    # --- librosa fallback -----------------------------------------------
    try:
        import librosa  # noqa: PLC0415
    except ImportError:
        log_fn("[beat] librosa also missing — skipping beat snap")
        return np.zeros(0, dtype=np.float32)
    try:
        y, sr_out = librosa.load(str(audio_path), sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_out)
        beats = librosa.frames_to_time(beat_frames, sr=sr_out)
        bpm = float(tempo.item() if hasattr(tempo, "item") else tempo)
        log_fn(f"[beat] librosa detected {len(beats)} beats at ~{bpm:.1f} "
               f"BPM over {len(y)/sr_out:.1f}s")
        return beats.astype(np.float32)
    except Exception as e:
        log_fn(f"[beat] librosa detection failed: {e}")
        return np.zeros(0, dtype=np.float32)


def _snap_to_beat(t: float, beats: np.ndarray,
                  lead_sec: float = 0.05,
                  max_shift_sec: float = 0.35) -> float:
    """Snap timestamp `t` to the nearest beat, placing the cut `lead_sec`
    *before* the beat so the new shot is on screen when the beat hits.
    If the nearest beat is farther than `max_shift_sec` away, don't snap
    (the original plan_merge decision had no nearby pulse to lock to,
    e.g. intro/outro without strong beats).
    """
    if len(beats) == 0:
        return t
    target = t + lead_sec
    idx = int(np.argmin(np.abs(beats - target)))
    snapped = float(beats[idx]) - lead_sec
    if abs(snapped - t) > max_shift_sec:
        return t
    return max(0.0, snapped)


# ---------- color matching ----------

def compute_color_stats(video: Path, offset_sec: float, dur_sec: float,
                        n_samples: int = 5
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n_samples` frames across [offset_sec, offset_sec+dur_sec]
    and return (mean_bgr, std_bgr), each shape (3,) float32.

    Used for per-chunk colour matching: different broadcasts grade to
    different LED colour temperatures, so chunks cut from two shows side
    by side flip colour palette visibly. We compute per-source stats on
    the EXACT source window that will be used by the chunk (not the full
    source video) so the correction target is the chunk's own lighting
    state rather than a whole-song average.
    """
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_f = int(round(offset_sec * fps))
        end_f = int(round((offset_sec + dur_sec) * fps))
        if total > 0:
            end_f = min(end_f, total - 1)
        if end_f <= start_f:
            return (np.array([128, 128, 128], dtype=np.float32),
                    np.array([60, 60, 60], dtype=np.float32))
        sample_fs = np.linspace(start_f, end_f,
                                 min(n_samples, max(1, end_f - start_f + 1))
                                 ).astype(int)
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for fi in sample_fs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            f = frame.astype(np.float32)
            means.append(f.reshape(-1, 3).mean(axis=0))
            stds.append(f.reshape(-1, 3).std(axis=0))
        if not means:
            return (np.array([128, 128, 128], dtype=np.float32),
                    np.array([60, 60, 60], dtype=np.float32))
        return (np.mean(means, axis=0).astype(np.float32),
                np.mean(stds, axis=0).astype(np.float32))
    finally:
        cap.release()


def color_match_params(src_mean: np.ndarray, src_std: np.ndarray,
                        ref_mean: np.ndarray, ref_std: np.ndarray,
                        strength: float = 0.7,
                        max_gain: float = 1.25, min_gain: float = 0.8
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Reinhard-style per-channel (BGR) mean/std matching.

    Returns (gain, offset) shaped (3,) such that
        out = clip(frame * gain + offset, 0, 255)
    shifts `src` toward `ref`. `strength` lerps between no-op (0.0,
    returns (ones, zeros)) and full Reinhard (1.0). Default 0.7 keeps
    30 percent of the source's own character so heavily-graded shows
    don't get flattened.

    `min_gain` / `max_gain` clamp how aggressively contrast is altered;
    otherwise a near-constant source channel would produce a huge ratio.
    """
    src_std = np.where(src_std < 1.0, 1.0, src_std).astype(np.float32)
    ref_std = ref_std.astype(np.float32)
    gain_full = np.clip(ref_std / src_std, min_gain, max_gain)
    offset_full = ref_mean - src_mean * gain_full
    # Lerp toward identity by `1 - strength`
    gain = (1.0 - strength) * np.ones(3, dtype=np.float32) + strength * gain_full
    offset = strength * offset_full
    return gain.astype(np.float32), offset.astype(np.float32)


# ---------- planning ----------

def plan_merge(sources: Sequence[MergeSource], clip_dur: float,
               step_sec: float = 1.0,
               min_chunk_sec: float = 1.5,
               min_face_conf: float = 0.50,
               soft_min_face_conf: float = 0.40,
               soft_dominance_margin: float = 0.08,
               beats: np.ndarray | None = None,
               beat_lead_sec: float = 0.05,
               yaw_buckets: "list[tuple[str, float | None]] | None" = None,
               log_fn=print) -> list[MergeChunk]:
    """Decide, per song-second bucket, which source to use. Returns the
    resulting list of merge chunks covering [0, clip_dur] contiguously.

    Deterministic given the same inputs — all tie-breakers use insertion
    order from the sources list + cluster insertion order.

    `min_face_conf` — sources whose best face-ID score is below this
    value have their quality_mask zeroed out, so the planner never picks
    them on quality. Group-stage cross-edits where the 0.42 acceptance
    threshold admitted a peer member (e.g. 0.45 "Yujin looks a bit like
    Wonyoung") are the typical victim. These sources can still serve
    as gap-fill fallback when no other source has the target visible.

    `soft_min_face_conf` / `soft_dominance_margin` — dominance-aware
    soft gate. A source whose `face_score < min_face_conf` is admitted
    anyway if `face_score >= soft_min_face_conf` AND
    `dominance >= soft_dominance_margin`. This rescues the case where
    the top tid has strong peer-dominance but the absolute face score
    is limited by low tight-face sample count (e.g. stage-wide shots
    where the member is visible but rarely in a head-on close-up).
    2026-04-23: without this, the BOjSR3L "After LIKE" fancam with
    face_score=0.449 dominance=+0.105 was centre-cropped, showing
    LIZ/Rei in the middle of the frame instead of Wonyoung. Set
    `soft_min_face_conf=0.0` to fall back to the hard gate only.
    """
    if not sources:
        return []
    n_buckets = max(1, int(np.ceil(clip_dur / step_sec)))

    def _pad(a: np.ndarray, dtype) -> np.ndarray:
        if len(a) < n_buckets:
            a = np.pad(a, (0, n_buckets - len(a)))
        elif len(a) > n_buckets:
            a = a[:n_buckets]
        return a.astype(dtype)

    # Two masks per source:
    #   coverage_masks — boolean, "target visible at all" — used for the
    #                    fallback "accept gap" logic and cross-event fill.
    #   quality_masks  — float[0,1] continuous quality score — used to
    #                    PICK among sources that have the target, so a
    #                    close-up well-framed shot beats a distant corner
    #                    shot even when both have 100% presence.
    coverage_masks: list[np.ndarray] = []
    quality_masks: list[np.ndarray] = []
    gated_sources: list[int] = []
    for i, s in enumerate(sources):
        raw_cm = _pad(s.tracked.coverage_mask(step_sec=step_sec), bool)
        qm = _pad(
            s.tracked.quality_mask(step_sec=step_sec,
                                   min_face_conf=min_face_conf,
                                   soft_min_face_conf=soft_min_face_conf,
                                   soft_dominance_margin=soft_dominance_margin),
            np.float32,
        )
        top_score = (max(s.tracked.target_scores.values())
                     if s.tracked.target_scores else 0.0)
        # Dominance of whichever tid was the top scorer — matches the
        # logic inside quality_mask's soft gate.
        top_tid = (max(s.tracked.target_scores.items(),
                       key=lambda kv: kv[1])[0]
                   if s.tracked.target_scores else None)
        top_dom = (float(s.tracked.target_dominance.get(top_tid, 0.0))
                   if top_tid is not None else 0.0)
        # If the source passed only via the soft gate, log it so we can
        # see which sources are being rescued.
        soft_admitted = (top_score < min_face_conf
                         and top_score >= soft_min_face_conf
                         and top_dom >= soft_dominance_margin
                         and float(qm.max()) > 0.0)
        if soft_admitted:
            log_fn(f"[merge] soft-admit: {s.meta.cluster_label} "
                   f"({s.meta.video_id}) face_score={top_score:.3f} "
                   f"(< min_face_conf={min_face_conf:.2f}) "
                   f"dominance={top_dom:+.3f} "
                   f"(>= {soft_dominance_margin:+.2f}) — admitted via soft gate")
        is_gated = (float(qm.max()) == 0.0 and raw_cm.any()
                    and top_score < min_face_conf)
        if is_gated:
            gated_sources.append(i)
            log_fn(f"[merge] gate: {s.meta.cluster_label} ({s.meta.video_id}) "
                   f"face_score={top_score:.3f} < min_face_conf="
                   f"{min_face_conf:.2f} dominance={top_dom:+.3f} "
                   f"(< soft_margin={soft_dominance_margin:+.2f}) — "
                   f"demoted to last-resort, trajectory forced to centre")
            # The face-ID is too weak to trust — we don't know if tid=N
            # is actually the target or a peer member that happened to
            # score 0.45. Override this source's trajectory to static
            # centre so when the planner DOES fall back to this source
            # (because nothing else covers a given bucket), the output
            # is an honest centre-crop of the stage rather than actively
            # following the possibly-wrong person. Shared state: we
            # mutate s.tracked in place, which is fine because
            # MergeSource instances are constructed for each merge call.
            n = s.tracked.meta.n_frames
            s.tracked.cxcy = np.tile([0.5, 0.5], (n, 1)).astype(np.float32)
            s.tracked.h_norm = None
            # KEEP coverage — we want the gated source to be PREFERRED
            # as gap-filler over an ungated source with q=0 (whose
            # interpolated trajectory between distant target frames may
            # follow a peer the editor cut to). Quality stays 0 so an
            # ungated source with real target visible still wins.
            cm = raw_cm
        else:
            # Coverage and quality must agree on presence. If either says
            # "no target here", force quality to 0 so selection can't
            # pick this bucket from this source.
            cm = raw_cm
            qm = qm * cm.astype(np.float32)
        coverage_masks.append(cm)
        quality_masks.append(qm)

    # Group source indices by cluster key, preserving order.
    cluster_members: dict[str, list[int]] = {}
    for i, s in enumerate(sources):
        cluster_members.setdefault(s.meta.cluster_key, []).append(i)

    # Score each cluster by total average quality (not just coverage): a
    # cluster with 80% coverage at great framing should outrank a cluster
    # with 95% coverage at terrible framing. Fall back to coverage sum when
    # quality is uniformly zero (e.g. non-face mode sources).
    cluster_score = {
        k: sum(float(quality_masks[i].mean()) for i in idxs)
        for k, idxs in cluster_members.items()
    }
    if all(v == 0.0 for v in cluster_score.values()):
        cluster_score = {
            k: sum(float(coverage_masks[i].mean()) for i in idxs)
            for k, idxs in cluster_members.items()
        }
    cluster_order = sorted(cluster_score.keys(),
                           key=lambda k: -cluster_score[k])
    primary_key = cluster_order[0]

    # Within each cluster, rank sources by individual mean quality desc.
    # Used for the "accept gap" fallback — when no source has the target
    # in a bucket, fall back to the cluster's best overall source.
    for k in cluster_members:
        cluster_members[k].sort(key=lambda i: -float(quality_masks[i].mean()))

    log_fn(f"[merge] clusters: " +
           ", ".join(f"{sources[v[0]].meta.cluster_label}(q="
                     f"{cluster_score[k]:.2f},cov="
                     f"{sum(float(coverage_masks[i].mean()) for i in v):.2f},"
                     f"n={len(v)})"
                     for k, v in cluster_members.items()) +
           f" | primary={sources[cluster_members[primary_key][0]].meta.cluster_label}")

    # Primary-cluster bonus: prefer staying within the primary cluster's
    # sources when quality is comparable — a same-event multi-cam swap
    # reads cleaner than a cross-event jump. Only jump when the other
    # cluster's quality beats primary by more than this margin.
    PRIMARY_BONUS = 0.05

    # M4b: yaw-bucket match bonus. When we have per-source yaw estimates
    # from pose_track, prefer sources whose yaw bucket matches the
    # *previous* pick's bucket — so a cut from a frontal-angle source
    # tends to land on another frontal-angle source rather than jumping
    # to a 3/4 side-profile angle (which looks like a rotation error
    # rather than an outfit swap). Smaller than PRIMARY_BONUS so it's a
    # tiebreaker only; never suppresses a real quality difference.
    YAW_MATCH_BONUS = 0.03

    yaw_bucket_by_src: "list[str]" = []
    if yaw_buckets is not None:
        if len(yaw_buckets) != len(sources):
            log_fn(f"[merge] warn: yaw_buckets has {len(yaw_buckets)} entries "
                   f"but sources has {len(sources)}; ignoring yaw bonus")
        else:
            yaw_bucket_by_src = [b for b, _ in yaw_buckets]
            # Log once so operators can see what plan_merge is about to
            # reward. Group by bucket to make the camera-angle cluster
            # obvious.
            by_bucket: dict[str, list[str]] = {}
            for i, s in enumerate(sources):
                b = yaw_bucket_by_src[i]
                by_bucket.setdefault(b, []).append(s.meta.video_id)
            log_fn("[merge] yaw plan: " +
                   ", ".join(f"{b}={vids}" for b, vids in by_bucket.items()))

    # Per-bucket source selection by argmax of (quality + primary bonus
    # + yaw-match bonus). Gated sources have qm=0 always, so ungated
    # sources with any target visible will always outrank them. When no
    # ungated covers a bucket but a gated does, the gated wins by
    # default (qm=0 > -1 initial). When NOTHING covers, fall back to a
    # gated source if any (safe centre-crop) before the ungated rank-0
    # (wandering trajectory).
    picks: list[int] = []
    prev_bucket: str | None = None
    for t in range(n_buckets):
        best_i = -1
        best_q = -1.0
        for i, s in enumerate(sources):
            if not coverage_masks[i][t]:
                continue
            q = float(quality_masks[i][t])
            if s.meta.cluster_key == primary_key:
                q += PRIMARY_BONUS
            # M4b: match bonus only when we have a prior pick with a
            # known bucket AND this source's bucket is known AND matches.
            # `unknown` (v1 cache or too few confident frames) never
            # earns the bonus — treat as neutral.
            if (yaw_bucket_by_src and prev_bucket is not None
                    and yaw_bucket_by_src[i] != "unknown"
                    and yaw_bucket_by_src[i] == prev_bucket):
                q += YAW_MATCH_BONUS
            if q > best_q:
                best_q = q
                best_i = i
        if best_i < 0:
            # No source has the target in this bucket. Prefer a gated
            # source (forced centre-crop) over an ungated rank-0 whose
            # interpolated trajectory may follow a peer member.
            if gated_sources:
                best_i = gated_sources[0]
            else:
                best_i = cluster_members[primary_key][0]
        picks.append(best_i)
        # Carry forward the newly picked source's bucket — but DON'T
        # overwrite with "unknown" (we want to remember the last known
        # bucket across gaps).
        if yaw_bucket_by_src:
            b = yaw_bucket_by_src[best_i]
            if b != "unknown":
                prev_bucket = b

    # Sticky-fallback pass: when a run of consecutive buckets has raw q=0
    # for every source (nobody is actually tracking the target there), the
    # argmax above would pick purely by PRIMARY_BONUS — which causes jarring
    # cuts from a source that was actively tracking to a different-cluster
    # source whose centre-crop may show a different member. Instead, carry
    # forward the source that was just in use (or pull back the next one,
    # if the run is at the head). This preserves stage continuity in the
    # "nobody knows where the target is" regions.
    raw_positive = np.zeros(n_buckets, dtype=bool)
    for t in range(n_buckets):
        for i in range(len(sources)):
            if coverage_masks[i][t] and float(quality_masks[i][t]) > 0.0:
                raw_positive[t] = True
                break
    t = 0
    sticky_overrides = 0
    while t < n_buckets:
        if raw_positive[t]:
            t += 1
            continue
        run_start = t
        while t < n_buckets and not raw_positive[t]:
            t += 1
        run_end = t  # exclusive
        if run_start > 0:
            anchor = picks[run_start - 1]
        elif run_end < n_buckets:
            anchor = picks[run_end]
        else:
            continue  # entire clip is q=0, leave the original fallback
        for k in range(run_start, run_end):
            if picks[k] != anchor:
                picks[k] = anchor
                sticky_overrides += 1
    if sticky_overrides:
        log_fn(f"[merge] sticky-fallback: carried forward source "
               f"on {sticky_overrides} q=0 bucket(s) for stage continuity")

    # Smooth 1-bucket isolated flips (A-B-A → A-A-A). Avoids a 1-second cut
    # to another angle and back, which reads as a glitch.
    smoothed = picks[:]
    for t in range(1, n_buckets - 1):
        if (smoothed[t] != smoothed[t - 1]
                and smoothed[t - 1] == smoothed[t + 1]):
            smoothed[t] = smoothed[t - 1]

    # Collapse consecutive same-source buckets into chunks.
    chunks: list[MergeChunk] = []
    i = 0
    while i < n_buckets:
        j = i
        while j < n_buckets and smoothed[j] == smoothed[i]:
            j += 1
        chunks.append(MergeChunk(
            song_start=min(i * step_sec, clip_dur),
            song_end=min(j * step_sec, clip_dur),
            src_idx=smoothed[i],
        ))
        i = j

    # Merge any chunk shorter than `min_chunk_sec` into the longer neighbour.
    # Short cuts cause the xfade to eat the entire chunk (xfade_dur=0.5s,
    # chunk=1s → only 0.5s visible). Better to stay on the prior/next source.
    if len(chunks) > 1 and min_chunk_sec > step_sec:
        merged: list[MergeChunk] = []
        for c in chunks:
            if merged and c.dur < min_chunk_sec:
                # extend previous to absorb this short one
                prev = merged[-1]
                merged[-1] = MergeChunk(prev.song_start, c.song_end,
                                         prev.src_idx)
            else:
                merged.append(c)
        # One more pass forward→backward for head case
        while len(merged) > 1 and merged[0].dur < min_chunk_sec:
            a, b = merged[0], merged[1]
            merged = [MergeChunk(a.song_start, b.song_end, b.src_idx)] + merged[2:]
        chunks = merged

    # Beat-snap pass: nudge each internal chunk boundary (not 0.0, not
    # clip_dur) to the nearest beat, placed `beat_lead_sec` before the
    # beat so the new shot is already on screen when the downbeat hits.
    # K-pop editing convention. If a chunk would shrink below min_chunk_sec
    # after snap, skip that snap (keep the pre-snap boundary).
    if beats is not None and len(beats) > 0 and len(chunks) > 1:
        n_snapped = 0
        for k in range(len(chunks) - 1):
            a = chunks[k]
            b = chunks[k + 1]
            orig = a.song_end
            snapped = _snap_to_beat(orig, beats, lead_sec=beat_lead_sec)
            # Guard: don't collapse either neighbour below min_chunk_sec.
            new_a_dur = snapped - a.song_start
            new_b_dur = b.song_end - snapped
            if (new_a_dur >= min_chunk_sec
                    and new_b_dur >= min_chunk_sec
                    and 0.0 < snapped < clip_dur):
                chunks[k] = MergeChunk(a.song_start, snapped, a.src_idx)
                chunks[k + 1] = MergeChunk(snapped, b.song_end, b.src_idx)
                if abs(snapped - orig) > 0.01:
                    n_snapped += 1
        log_fn(f"[merge] beat-snap: adjusted {n_snapped}/"
               f"{len(chunks) - 1} boundaries (lead={beat_lead_sec:.2f}s)")

    for c in chunks:
        s = sources[c.src_idx]
        # Mean quality over the chunk's buckets, for visibility into why
        # the planner chose this source (a low number here on a chosen
        # chunk means no source had good framing in that window).
        b0 = int(c.song_start / step_sec)
        b1 = max(b0 + 1, int(np.ceil(c.song_end / step_sec)))
        q_chunk = float(quality_masks[c.src_idx][b0:b1].mean())
        log_fn(f"[merge] song[{c.song_start:.1f},{c.song_end:.1f}]s ← "
               f"{s.meta.cluster_label} ({s.meta.video_id}) q={q_chunk:.2f}")
    return chunks


# ---------- ffmpeg rendering ----------

def merge_clip(chunks: Sequence[MergeChunk],
               sources: Sequence[MergeSource],
               canonical_audio_mp3: Path,
               clip_dur: float,
               dst: Path,
               xfade_dur: float = 0.5,
               use_pose: bool = False,
               log_fn=print) -> None:
    """Write each chunk to a temp mp4 (with post-roll for xfade), then
    xfade-concat them and mux canonical_audio_mp3 as the audio track.

    When `use_pose=True`, run RTMPose-m on the target bbox per frame for
    each source that appears in the final chunk list, and derive a
    session-level head_y anchor. This lets extract_clip_tracked place the
    dancer's head at the same vertical fraction across every source — so
    hard-cut transitions don't pop the head vertically (a common source
    of "not quite outfit-swap" feel when sources are shot at different
    camera heights).

    Note: we import extract_clip_tracked here (not at module top) to avoid
    circular imports — merge_sources is called *by* match_video's pipeline
    which already owns extract_clip_tracked.
    """
    from match_video import extract_clip_tracked  # lazy import

    if not chunks:
        raise ValueError("no chunks to merge")
    if len(chunks) == 1:
        # Single-source case — just run the normal extractor with canonical
        # audio mux at the end.
        c = chunks[0]
        s = sources[c.src_idx]
        _extract_single(s, c.song_start, clip_dur, dst, log_fn)
        _mux_canonical_audio(dst, canonical_audio_mp3, clip_dur, log_fn)
        return

    # ---- Session-level zoom anchor (M1b) --------------------------------
    # When we cut between two sources that tracked the same member, the
    # bbox-derived zoom levels differ source-to-source (camera distance,
    # lens, detector jitter). Each source's EMA breathing then produces
    # slightly different on-screen sizes at the cut boundary, which reads
    # as a zoom pop rather than a costume change. We eliminate that by
    # picking ONE h_anchor for the whole merge session: median of each
    # source's per-frame bbox heights, then median across sources. All
    # chunks then zoom to the same constant fill — the outfit-swap look.
    #
    # Selected sources = only those referenced by at least one chunk
    # (gated/unused sources shouldn't influence the anchor).
    src_used = sorted({c.src_idx for c in chunks})
    src_medians: list[float] = []
    for i in src_used:
        hn = sources[i].tracked.h_norm
        if hn is None:
            continue
        valid = hn[hn > 0]
        if valid.size >= 30:  # at least ~1s worth of bbox observations
            src_medians.append(float(np.median(valid)))
    if src_medians:
        session_h_anchor: float | None = float(np.median(src_medians))
        log_fn(f"[merge] session h_anchor={session_h_anchor:.3f} "
               f"(median across {len(src_medians)} sources' per-frame "
               f"bbox heights)")
    else:
        # No confident bbox data — extract_clip_tracked will fall back to
        # its own per-chunk behaviour (fixed crop or EMA breathing if any
        # chunk happens to carry h_traj).
        session_h_anchor = None
        log_fn("[merge] session h_anchor: none available "
               "(no source carried enough bbox data)")

    # ---- Session-level head-y anchor (M3a) -------------------------------
    # Parallel to h_anchor above, but for vertical head position. The
    # body bbox's top edge is unstable (hair/hands/hats); RTMPose's
    # nose+eyes give a stable head center. If two sources put the dancer
    # at body-center y=0.5 but head at y=0.25 vs y=0.35 (different camera
    # heights), the hard cut shows the head jumping. Lock them to a
    # single fraction.
    #
    # Only fires when `use_pose=True` (default False) because it adds
    # ~30s per source of CPU pose inference. Results are cached by
    # pose_track per (video, start_sec, dur_sec) so re-runs are free.
    session_head_y: float | None = None
    if use_pose:
        try:
            import pose_track  # noqa: PLC0415
        except Exception as e:
            log_fn(f"[merge] pose: import failed ({e!r}) — "
                   f"disabling head-y anchor")
        else:
            head_tracks = []
            for i in src_used:
                s = sources[i]
                try:
                    ht = pose_track.track_head_keypoints(
                        s.meta.path,
                        tracks=s.tracked.tracks,
                        target_ids=s.tracked.target_ids,
                        start_sec=s.tracked.meta.start_sec,
                        dur_sec=s.tracked.meta.dur_sec,
                        meta=s.tracked.meta,
                        log_fn=log_fn,
                    )
                except Exception as e:
                    log_fn(f"[merge] pose: track failed for "
                           f"{s.meta.video_id}: {e!r}")
                    ht = None
                s.head = ht
                if ht is not None:
                    head_tracks.append(ht)
            session_head_y = pose_track.session_head_y_anchor(head_tracks)
            if session_head_y is not None:
                log_fn(f"[merge] session head_y_anchor={session_head_y:.3f} "
                       f"(median across {len(head_tracks)} sources' head "
                       f"keypoint y-positions)")
            else:
                log_fn("[merge] session head_y_anchor: none available "
                       "(no source had enough high-confidence head frames)")

            # M4a: log per-source yaw bucket. Used by plan_merge in M4b
            # to prefer same-bucket cuts over cross-bucket cuts.
            yaw_buckets = pose_track.session_yaw_bucket(head_tracks)
            # src_used indexes into the original `sources` list in the
            # same order as head_tracks was filled, so (src_used[i], ...)
            # pairs the bucket with its source.
            for bi, (bucket, med) in enumerate(yaw_buckets):
                if bi < len(src_used):
                    src_i = src_used[bi]
                    sid = sources[src_i].meta.video_id
                else:
                    sid = f"#{bi}"
                med_s = f"{med:+.3f}" if med is not None else " none "
                log_fn(f"[merge] yaw bucket: {sid:12s} "
                       f"median={med_s}  → {bucket}")

    # ---- Per-chunk colour match ------------------------------------------
    # Sample each chunk's source window, compute BGR mean/std, pick the
    # LONGEST chunk as reference (most screen time anchors the look), and
    # derive (gain, offset) to map every other chunk toward that ref.
    # Single-chunk outputs don't reach this path (early return above).
    chunk_stats: list[tuple[np.ndarray, np.ndarray]] = []
    for c in chunks:
        s = sources[c.src_idx]
        m, sd = compute_color_stats(
            s.meta.path, s.offset_sec + c.song_start, c.dur, n_samples=5)
        chunk_stats.append((m, sd))
    ref_idx = max(range(len(chunks)), key=lambda k: chunks[k].dur)
    ref_mean, ref_std = chunk_stats[ref_idx]
    color_params: list[tuple[np.ndarray, np.ndarray] | None] = []
    for k, (m, sd) in enumerate(chunk_stats):
        if k == ref_idx:
            color_params.append(None)
            continue
        gain, offset = color_match_params(m, sd, ref_mean, ref_std,
                                           strength=0.7)
        # Log the shift in human-readable BGR mean terms.
        predicted = m * gain + offset
        delta = predicted - ref_mean
        log_fn(f"[merge] color-match chunk {k} "
               f"({sources[chunks[k].src_idx].meta.video_id}): "
               f"src_mean=[{m[0]:.0f},{m[1]:.0f},{m[2]:.0f}] "
               f"→ target=[{ref_mean[0]:.0f},{ref_mean[1]:.0f},{ref_mean[2]:.0f}] "
               f"residual=[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f}]")
        color_params.append((gain, offset))
    log_fn(f"[merge] color-match reference: chunk {ref_idx} "
           f"({sources[chunks[ref_idx].src_idx].meta.video_id}, "
           f"dur={chunks[ref_idx].dur:.1f}s)")

    tmpdir = Path(tempfile.mkdtemp(prefix="merge_", dir=tempfile.gettempdir()))
    try:
        temp_paths: list[Path] = []
        extract_durs: list[float] = []
        for i, c in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            pad = 0.0 if is_last else xfade_dur
            s = sources[c.src_idx]
            extract_dur = c.dur + pad
            # Don't seek beyond source.
            src_probe_end = s.offset_sec + c.song_end + pad
            # If the seek+dur would exceed source, clip tail (rare in
            # practice: clip_dur is typically much shorter than the source).
            # We leave the exact bounds check to ffmpeg (it'll just stop).

            # Build a per-chunk SegmentMeta + trajectory slice from the
            # parent TrackedSegment. The parent's n_frames covered the
            # whole clip_dur; we take the slice [song_start, song_start+ed)
            # in frame-index terms.
            fps = s.tracked.meta.fps
            parent_n = s.tracked.meta.n_frames
            f0 = int(round(c.song_start * fps))
            f1 = int(round((c.song_start + extract_dur) * fps))
            f0 = max(0, min(parent_n, f0))
            f1 = max(f0, min(parent_n, f1))
            if f1 <= f0:
                log_fn(f"[merge:warn] chunk {i} produced empty slice "
                       f"[{f0},{f1}] from parent n_frames={parent_n}")
                continue
            traj = s.tracked.cxcy[f0:f1].copy()
            h_t = (s.tracked.h_norm[f0:f1].copy()
                   if s.tracked.h_norm is not None else None)
            # M3b: slice this source's head trajectory to match the chunk.
            # When use_pose is off, s.head is None and these stay None.
            head_y_t = None
            head_conf_t = None
            if s.head is not None and s.head.head_xy_norm.shape[0] >= f1:
                head_y_t = s.head.head_xy_norm[f0:f1, 1].copy()
                head_conf_t = s.head.head_conf[f0:f1].copy()
            actual_dur = (f1 - f0) / fps

            chunk_meta = SegmentMeta(
                width=s.tracked.meta.width,
                height=s.tracked.meta.height,
                fps=fps,
                start_sec=s.offset_sec + c.song_start,
                dur_sec=actual_dur,
                n_frames=(f1 - f0),
            )
            out_path = tmpdir / f"chunk_{i:02d}.mp4"
            log_fn(f"[merge] render chunk {i}: src={s.meta.video_id} "
                   f"song[{c.song_start:.1f},{c.song_start+actual_dur:.1f}]s "
                   f"video[{chunk_meta.start_sec:.1f},"
                   f"{chunk_meta.start_sec+actual_dur:.1f}]s "
                   f"→ {out_path.name}")
            cp = color_params[i]
            extract_clip_tracked(
                s.meta.path,
                chunk_meta.start_sec,
                actual_dur,
                out_path,
                traj,
                chunk_meta,
                h_traj=h_t,
                h_anchor=session_h_anchor,
                head_y_traj=head_y_t,
                head_y_conf=head_conf_t,
                head_y_target=(session_head_y
                               if session_head_y is not None else 0.22),
                delogo_corners=None,
                color_gain=(cp[0] if cp is not None else None),
                color_offset=(cp[1] if cp is not None else None),
            )
            temp_paths.append(out_path)
            extract_durs.append(actual_dur)

        if not temp_paths:
            raise RuntimeError("all chunks produced empty slices")

        if len(temp_paths) == 1:
            # After filtering, only one chunk survived.
            shutil.copy(temp_paths[0], dst)
            _mux_canonical_audio(dst, canonical_audio_mp3, clip_dur, log_fn)
            return

        if xfade_dur <= 0:
            # Hard-cut mode: the dancer instantly changes outfits/venue at
            # each transition instead of the 0.5s dissolve. This is the
            # "outfit-swap fancam" look. Requires canonical framing
            # (M3) to really sell it — without pose-aligned crops the
            # hard cut exposes every pixel-level framing jump.
            _hardcut_concat(temp_paths, canonical_audio_mp3, clip_dur, dst,
                            log_fn)
        else:
            _xfade_concat(temp_paths, extract_durs, xfade_dur,
                          canonical_audio_mp3, clip_dur, dst, log_fn)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_single(s: MergeSource, song_start: float, clip_dur: float,
                    dst: Path, log_fn) -> None:
    """Extract a single chunk covering [song_start, song_start+clip_dur]."""
    from match_video import extract_clip_tracked
    fps = s.tracked.meta.fps
    parent_n = s.tracked.meta.n_frames
    f0 = max(0, int(round(song_start * fps)))
    f1 = max(f0, min(parent_n, int(round((song_start + clip_dur) * fps))))
    traj = s.tracked.cxcy[f0:f1].copy()
    h_t = (s.tracked.h_norm[f0:f1].copy()
           if s.tracked.h_norm is not None else None)
    actual_dur = (f1 - f0) / fps
    seg = SegmentMeta(
        width=s.tracked.meta.width,
        height=s.tracked.meta.height,
        fps=fps,
        start_sec=s.offset_sec + song_start,
        dur_sec=actual_dur,
        n_frames=(f1 - f0),
    )
    extract_clip_tracked(s.meta.path, seg.start_sec, actual_dur, dst,
                         traj, seg, h_traj=h_t, delogo_corners=None)


def _hardcut_concat(temp_paths: Sequence[Path], canonical_audio_mp3: Path,
                     clip_dur: float, dst: Path, log_fn) -> None:
    """Hard-cut concat of N chunks — zero-duration transition. Gives the
    'outfit-swap' feel where the dancer instantly changes clothes/venue at
    every cut instead of the 0.5s dissolve that xfade would produce.

    Chunks come from different sources with different fps (e.g. 29.97
    Inkigayo stage + 60 fps Music Bank one-take in the same plan), so we
    normalise to a common fps/timebase BEFORE concat — matching the xfade
    path's choice of 30 fps / 1/30000 timebase.

    Unlike _xfade_concat we don't need per-chunk offset math: each chunk
    was extracted at exactly its unique song duration (no post-roll pad),
    so concat=n=N stitches them end-to-end correctly.
    """
    n = len(temp_paths)
    HARDCUT_FPS = 30
    filters: list[str] = []
    for i in range(n):
        filters.append(
            f"[{i}:v]fps={HARDCUT_FPS},settb=1/{HARDCUT_FPS * 1000}[v{i}n]"
        )
    concat_inputs = "".join(f"[v{i}n]" for i in range(n))
    filters.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")
    filter_complex = ";".join(filters)

    cmd: list[str] = ["ffmpeg", "-y", "-v", "error"]
    for p in temp_paths:
        cmd += ["-i", str(p)]
    cmd += ["-i", str(canonical_audio_mp3)]
    audio_index = n  # canonical audio is the (n+1)-th input, 0-indexed = n
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{audio_index}:a:0",
        "-t", f"{clip_dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(dst),
    ]
    log_fn(f"[merge] hard-cut concat {n} chunks → {dst.name}")
    subprocess.run(cmd, check=True)


def _xfade_concat(temp_paths: Sequence[Path], durs: Sequence[float],
                   xfade_dur: float, canonical_audio_mp3: Path,
                   clip_dur: float, dst: Path, log_fn) -> None:
    """Chain xfade across N >= 2 chunks. Each chunk i (for i<N-1) was
    extracted with xfade_dur of trailing overlap; last chunk has no
    overlap. The transition math:
        xfade offset between chunk i and chunk i+1 = cumulative unique
        duration through chunk i = sum(d_k - xfade for k<=i, except
        the last chunk has no trailing overlap).
    """
    n = len(temp_paths)
    # Unique duration of each chunk = duration minus the trailing overlap
    # (all chunks except the last have a trailing overlap of xfade_dur).
    unique_dur = [
        max(0.0, d - (xfade_dur if i < n - 1 else 0.0))
        for i, d in enumerate(durs)
    ]
    offsets: list[float] = []
    cum = 0.0
    for i in range(n - 1):
        cum += unique_dur[i]
        offsets.append(cum)

    # filter_complex: xfade requires all inputs to share the same
    # framerate and timebase. Chunks come from different source videos
    # which may have different fps (e.g. 29.97 Inkigayo stage + 60 fps
    # Music Bank one-take in the same plan). Force each input through
    # an fps/settb normaliser BEFORE the xfade chain so the chain sees
    # consistent inputs. We pick 30 fps and 1/30000 timebase as a
    # common denominator — 30 is near-universal for K-pop broadcast and
    # keeps file size reasonable; higher-fps sources get decimated.
    XFADE_FPS = 30
    filters: list[str] = []
    for i in range(n):
        filters.append(
            f"[{i}:v]fps={XFADE_FPS},settb=1/{XFADE_FPS * 1000}[v{i}n]"
        )
    prev_label = "v0n"
    for i in range(1, n):
        out_label = "vout" if i == n - 1 else f"vx{i}"
        filters.append(
            f"[{prev_label}][v{i}n]"
            f"xfade=transition=fade:duration={xfade_dur:.3f}:"
            f"offset={offsets[i-1]:.3f}"
            f"[{out_label}]"
        )
        prev_label = out_label
    filter_complex = ";".join(filters)

    cmd: list[str] = ["ffmpeg", "-y", "-v", "error"]
    for p in temp_paths:
        cmd += ["-i", str(p)]
    cmd += ["-i", str(canonical_audio_mp3)]
    audio_index = n  # canonical audio is the (n+1)-th input, 0-indexed = n
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{audio_index}:a:0",
        "-t", f"{clip_dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(dst),
    ]
    log_fn(f"[merge] xfade-concat {n} chunks → {dst.name} "
           f"(offsets={[f'{o:.2f}' for o in offsets]})")
    subprocess.run(cmd, check=True)


def _mux_canonical_audio(mp4: Path, canonical_audio_mp3: Path,
                         clip_dur: float, log_fn) -> None:
    """Replace the audio track of `mp4` with `canonical_audio_mp3`
    (truncated to clip_dur). Runs in-place via a temp file."""
    tmp = mp4.with_suffix(".audiomux.mp4")
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(mp4),
        "-i", str(canonical_audio_mp3),
        "-map", "0:v:0", "-map", "1:a:0",
        "-t", f"{clip_dur:.3f}",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(mp4)
    log_fn(f"[merge] muxed canonical audio into {mp4.name}")
