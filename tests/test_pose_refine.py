"""M5 synthetic regression: _refine_cuts_by_pose snaps cuts within
±N frames to minimize yaw mismatch at the boundary.

Five hand-crafted scenarios — each runs in isolation against the helper
with a stubbed HeadTrack carrying yaw_proxy / yaw_conf series.

Run on Mac mini: ./venv/bin/python tests/test_pose_refine.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge_sources import (  # noqa: E402
    MergeChunk,
    MergeSource,
    _refine_cuts_by_pose,
)
from sources import SourceMeta  # noqa: E402


@dataclass
class StubMeta:
    n_frames: int = 1800
    fps: float = 30.0


@dataclass
class StubTracked:
    meta: StubMeta = field(default_factory=StubMeta)


@dataclass
class StubHead:
    yaw_proxy: np.ndarray
    yaw_conf: np.ndarray
    head_xy_norm: np.ndarray = field(default_factory=lambda: np.zeros((1800, 2), dtype=np.float32))
    head_conf: np.ndarray = field(default_factory=lambda: np.ones(1800, dtype=np.float32))
    n_frames: int = 1800
    probe_elapsed_sec: float = 0.0
    version: int = 2


def make_source(vid: str, yaw: np.ndarray, conf: np.ndarray | None = None,
                n_frames: int = 1800, fps: float = 30.0) -> MergeSource:
    meta = SourceMeta(video_id=vid, path=Path(f"/tmp/{vid}.mp4"),
                      title=vid, duration=60.0,
                      broadcaster="MBC", date="20260101")
    if conf is None:
        conf = np.ones(n_frames, dtype=np.float32)
    src = MergeSource(
        meta=meta,
        offset_sec=0.0,
        tracked=StubTracked(meta=StubMeta(n_frames=n_frames, fps=fps)),
        matched_title="REBEL HEART",
        head=StubHead(
            yaw_proxy=yaw.astype(np.float32),
            yaw_conf=conf.astype(np.float32),
            head_xy_norm=np.zeros((n_frames, 2), dtype=np.float32),
            head_conf=np.ones(n_frames, dtype=np.float32),
            n_frames=n_frames,
        ),
    )
    return src


def collect_logs():
    buf: list[str] = []
    return buf, lambda *a, **k: buf.append(" ".join(str(x) for x in a))


def test_no_op_when_already_aligned():
    """If yaw_A and yaw_B match exactly at delta=0, no snap should fire."""
    n = 1800
    fps = 30.0
    yaw_a = np.full(n, 0.10, dtype=np.float32)
    yaw_b = np.full(n, 0.10, dtype=np.float32)
    a = make_source("AAA", yaw_a)
    b = make_source("BBB", yaw_b)
    chunks = [MergeChunk(0.0, 5.0, 0), MergeChunk(5.0, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3, log_fn=log_fn)
    assert out[0].song_end == 5.0, f"expected 5.0, got {out[0].song_end}"
    print(f"  ✓ no snap when yaw aligned (boundary stays at {out[0].song_end:.3f}s)")


def test_snaps_to_min_yaw_delta():
    """Yaw_A is constant 0.20. Yaw_B has a discontinuity: 0.50 at frame 150,
    0.20 at frame 152. Cut at song time 5.0 → frame 150 → wants delta=+2.
    Expected: cut moves to 5.0 + 2/30 ≈ 5.067."""
    n = 1800
    fps = 30.0
    yaw_a = np.full(n, 0.20, dtype=np.float32)
    yaw_b = np.full(n, 0.50, dtype=np.float32)
    yaw_b[152:] = 0.20  # Sharp transition: from 5.067s onward, yaw_b matches yaw_a
    a = make_source("AAA", yaw_a)
    b = make_source("BBB", yaw_b)
    chunks = [MergeChunk(0.0, 5.0, 0), MergeChunk(5.0, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3, log_fn=log_fn)
    expected_t = 5.0 + 2 / fps
    assert abs(out[0].song_end - expected_t) < 1e-4, \
        f"expected snap to {expected_t:.4f}s, got {out[0].song_end:.4f}s"
    assert out[1].song_start == out[0].song_end, "boundary mismatch"
    print(f"  ✓ snapped to delta=+2f: {out[0].song_end:.4f}s "
          f"(expected {expected_t:.4f}s)")


def test_snap_negative_direction():
    """Yaw_B becomes mismatched at frame 150 onward. Cut at 5.0 → frame 150.
    Should snap delta=-2 to land before the mismatch."""
    n = 1800
    fps = 30.0
    yaw_a = np.full(n, 0.20, dtype=np.float32)
    yaw_b = np.full(n, 0.20, dtype=np.float32)
    yaw_b[148:] = 0.50  # mismatch from frame 148 onward → delta=-3 best
    a = make_source("AAA", yaw_a)
    b = make_source("BBB", yaw_b)
    chunks = [MergeChunk(0.0, 5.0, 0), MergeChunk(5.0, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3, log_fn=log_fn)
    expected_t = 5.0 - 3 / fps  # snap back 3 frames
    assert abs(out[0].song_end - expected_t) < 1e-4, \
        f"expected snap to {expected_t:.4f}s, got {out[0].song_end:.4f}s"
    print(f"  ✓ snapped to delta=-3f: {out[0].song_end:.4f}s")


def test_skip_when_no_head():
    """If a source has head=None, the cut should not be snapped."""
    n = 1800
    yaw_a = np.full(n, 0.20, dtype=np.float32)
    a = make_source("AAA", yaw_a)
    b_meta = SourceMeta(video_id="BBB", path=Path("/tmp/BBB.mp4"),
                        title="BBB", duration=60.0,
                        broadcaster="MBC", date="20260101")
    b = MergeSource(meta=b_meta, offset_sec=0.0,
                    tracked=StubTracked(),
                    matched_title="x", head=None)  # no head
    chunks = [MergeChunk(0.0, 5.0, 0), MergeChunk(5.0, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3, log_fn=log_fn)
    assert out[0].song_end == 5.0, "no head → no snap"
    assert any("skip_no_head=1" in m for m in logs), \
        f"expected log of no-head skip, got {logs}"
    print("  ✓ no-head-track cuts left untouched")


def test_skip_when_low_conf():
    """If yaw_conf is below threshold at all candidate deltas, no snap."""
    n = 1800
    yaw_a = np.full(n, 0.20, dtype=np.float32)
    yaw_b = np.full(n, 0.50, dtype=np.float32)
    conf_zero = np.zeros(n, dtype=np.float32)
    a = make_source("AAA", yaw_a, conf=conf_zero)
    b = make_source("BBB", yaw_b, conf=conf_zero)
    chunks = [MergeChunk(0.0, 5.0, 0), MergeChunk(5.0, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3, log_fn=log_fn)
    assert out[0].song_end == 5.0, "low conf → no snap"
    assert any("skip_no_conf=1" in m for m in logs), \
        f"expected no_conf skip, got {logs}"
    print("  ✓ low-conf yaw correctly skipped")


def test_min_chunk_guard():
    """Snap that would drop a neighbour below min_chunk_sec is rejected.

    Setup: cut at song t=1.05s. round(1.05*30) = 32 (banker's: 31.5→32).
    Make yaw_b match yaw_a perfectly at frame 29 (delta=-3 → improvement
    available), but mismatch at all other frames in [-3..+3] range.
    Then chunk_a after snap = 0.95s < 1.0 min → blocked by guard."""
    n = 1800
    fps = 30.0
    yaw_a = np.full(n, 0.20, dtype=np.float32)
    yaw_b = np.full(n, 0.50, dtype=np.float32)  # mismatch everywhere
    yaw_b[29] = 0.20  # only delta=-3 gives a perfect match
    a = make_source("AAA", yaw_a)
    b = make_source("BBB", yaw_b)
    chunks = [MergeChunk(0.0, 1.05, 0), MergeChunk(1.05, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3, log_fn=log_fn)
    assert out[0].song_end == 1.05, \
        f"min_chunk guard should reject snap; got {out[0].song_end}"
    assert any("skip_min_chunk=1" in m for m in logs), \
        f"expected min_chunk skip, got {logs}"
    print("  ✓ min_chunk_sec guard rejects unsafe snaps")


def test_score_cap_rejects_wild_snap():
    """When baseline is n/a (low conf at boundary frame) and the only
    confident candidates have huge yaw mismatch, do NOT snap. Without
    this cap, the helper would happily move a cut to a frame with
    near-maximal yaw mismatch (yaw² ≈ 2) just because nothing else
    was measurable."""
    n = 1800
    fps = 30.0
    yaw_a = np.full(n, 0.20, dtype=np.float32)
    yaw_b = np.full(n, -0.95, dtype=np.float32)  # near-max mismatch (1.15² ≈ 1.32)
    # Boundary at frame 150 — kill conf there to force baseline=n/a.
    conf_a = np.ones(n, dtype=np.float32)
    conf_b = np.ones(n, dtype=np.float32)
    conf_a[150] = 0.0
    conf_b[150] = 0.0
    a = make_source("AAA", yaw_a, conf=conf_a)
    b = make_source("BBB", yaw_b, conf=conf_b)
    chunks = [MergeChunk(0.0, 5.0, 0), MergeChunk(5.0, 10.0, 1)]
    logs, log_fn = collect_logs()
    out = _refine_cuts_by_pose(chunks, [a, b], clip_dur=10.0,
                                min_chunk_sec=1.0,
                                max_delta_frames=3,
                                max_acceptable_score=0.25,
                                log_fn=log_fn)
    assert out[0].song_end == 5.0, \
        f"score cap should block wild snap; got {out[0].song_end}"
    assert any("adjusted 0/" in m for m in logs), \
        f"expected zero refinements, got {logs}"
    print("  ✓ score cap rejects snap to wildly-mismatched yaw")


def main():
    print("=== M5 synthetic _refine_cuts_by_pose regression ===\n")
    print("[T1] no-op when yaw already aligned at boundary")
    test_no_op_when_already_aligned()
    print("\n[T2] snaps forward to minimize yaw mismatch")
    test_snaps_to_min_yaw_delta()
    print("\n[T3] snaps backward to minimize yaw mismatch")
    test_snap_negative_direction()
    print("\n[T4] skips cut when one source has no HeadTrack")
    test_skip_when_no_head()
    print("\n[T5] skips when yaw_conf below threshold everywhere in window")
    test_skip_when_low_conf()
    print("\n[T6] respects min_chunk_sec guard")
    test_min_chunk_guard()
    print("\n[T7] score cap rejects snap to wildly-mismatched yaw")
    test_score_cap_rejects_wild_snap()
    print("\n=== all 7 tests passed ===")


if __name__ == "__main__":
    main()
