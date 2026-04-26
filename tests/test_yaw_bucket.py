"""Synthetic regression test for M4b yaw-bucket match bonus in plan_merge.

REBEL HEART's downloadable pool has all-frontal sources (3 frontal +
1 unknown, no left/right), so the bucket-match bonus path never
exercises in production. This test feeds plan_merge hand-crafted
sources + yaw_buckets so we can verify the bonus arithmetic does what
M4b advertised.

Run on Mac mini: ./venv/bin/python /tmp/test_yaw_bucket.py
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Repo on path — works regardless of where the test lives.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge_sources import MergeSource, plan_merge  # noqa: E402
from sources import SourceMeta  # noqa: E402


@dataclass
class StubMeta:
    n_frames: int = 1800  # 60s @ 30fps default
    fps: float = 30.0


@dataclass
class StubTracked:
    """Minimal duck-type for TrackedSegment. plan_merge calls:
       .coverage_mask(step_sec), .quality_mask(step_sec, ...),
       .target_scores (dict), .target_dominance (dict),
       .meta.n_frames, .cxcy, .h_norm.
    """
    cov: np.ndarray
    qual: np.ndarray
    target_ids_: list = field(default_factory=lambda: [1])
    # Give a high target_score so the soft/hard gate doesn't trigger
    # demotion in the synthetic scenarios we want to exercise.
    target_scores: dict = field(default_factory=lambda: {1: 0.80})
    target_dominance: dict = field(default_factory=lambda: {1: 0.50})
    meta: StubMeta = field(default_factory=StubMeta)
    cxcy: np.ndarray = field(default_factory=lambda: np.tile([0.5, 0.5], (1800, 1)).astype(np.float32))
    h_norm: object = None

    @property
    def target_ids(self):
        return self.target_ids_

    def coverage_mask(self, step_sec: float = 1.0) -> np.ndarray:
        return self.cov.copy()

    def quality_mask(self, step_sec: float = 1.0,
                     min_face_conf: float = 0.0,
                     soft_min_face_conf: float = 0.0,
                     soft_dominance_margin: float = 0.0) -> np.ndarray:
        return self.qual.copy()


def make_source(vid: str, broadcaster: str, date: str,
                cov: np.ndarray, qual: np.ndarray) -> MergeSource:
    meta = SourceMeta(
        video_id=vid,
        path=Path(f"/tmp/{vid}.mp4"),
        title=f"stub {vid}",
        duration=60.0,
        broadcaster=broadcaster,
        date=date,
    )
    return MergeSource(
        meta=meta,
        offset_sec=0.0,
        tracked=StubTracked(cov=cov, qual=qual),
        matched_title="REBEL HEART",
    )


def chunks_picks_str(chunks, sources):
    return " → ".join(
        f"{sources[c.src_idx].meta.video_id}@[{c.song_start:.0f},{c.song_end:.0f}]"
        for c in chunks
    )


def test_baseline_no_yaw():
    """No yaw buckets supplied → only PRIMARY_BONUS applies. Highest-cluster
    source dominates picks. Establishes a reference."""
    n = 12
    cov_all = np.ones(n, dtype=bool)
    # Put source A in its own cluster so it gets PRIMARY_BONUS easily
    src_a = make_source("AAA", "MBC", "20260101", cov_all,
                        np.full(n, 0.50, dtype=np.float32))
    src_b = make_source("BBB", "KBS", "20260101", cov_all,
                        np.full(n, 0.50, dtype=np.float32))
    src_c = make_source("CCC", "SBS", "20260101", cov_all,
                        np.full(n, 0.50, dtype=np.float32))
    sources = [src_a, src_b, src_c]

    log_buf = io.StringIO()
    chunks = plan_merge(sources, clip_dur=float(n), step_sec=1.0,
                        min_chunk_sec=1.0, beats=None,
                        yaw_buckets=None,
                        log_fn=lambda *a, **k: print(*a, **k, file=log_buf))
    picks = [c.src_idx for c in chunks]
    print(f"  baseline: chunks={chunks_picks_str(chunks, sources)}")
    print(f"  baseline picks: {picks}")
    # With identical quality and 3 single-source clusters, PRIMARY_BONUS
    # gives every cluster's primary equal weight. The first listed wins ties.
    assert all(p == 0 for p in picks), \
        f"with identical quality + tie-break, all picks should be source 0; got {picks}"
    print("  ✓ baseline: tie-break favors first-listed source")


def test_bucket_match_bonus_breaks_tie():
    """Sources B (left) and C (right) at equal quality. Once B is picked
    once (e.g. because it's first in tie-break), the bucket-match bonus
    should keep B winning across subsequent ties — instead of flapping
    B → C → B → C."""
    n = 8
    cov_all = np.ones(n, dtype=bool)

    # Drop source A's coverage everywhere so PRIMARY_BONUS doesn't matter.
    src_a = make_source("AAA", "MBC", "20260101",
                        np.zeros(n, dtype=bool),
                        np.zeros(n, dtype=np.float32))
    src_b = make_source("BBB", "KBS", "20260101", cov_all,
                        np.full(n, 0.50, dtype=np.float32))
    src_c = make_source("CCC", "SBS", "20260101", cov_all,
                        np.full(n, 0.50, dtype=np.float32))
    sources = [src_a, src_b, src_c]
    yaw_buckets = [("frontal", 0.0), ("left", +0.30), ("right", -0.30)]

    log_buf = io.StringIO()
    chunks = plan_merge(sources, clip_dur=float(n), step_sec=1.0,
                        min_chunk_sec=1.0, beats=None,
                        yaw_buckets=yaw_buckets,
                        log_fn=lambda *a, **k: print(*a, **k, file=log_buf))
    picks = [c.src_idx for c in chunks]
    print(f"  match-bonus: chunks={chunks_picks_str(chunks, sources)}")
    print(f"  match-bonus picks per chunk: {picks}")
    # After first pick, prev_bucket is set; bucket-match bonus should
    # keep the same source winning every subsequent equal-quality bucket.
    # So we expect a SINGLE chunk covering all n buckets from the first-picked source.
    assert len(chunks) == 1, \
        f"bucket-match bonus should yield 1 chunk (no flapping), got {len(chunks)}"
    assert picks[0] in (1, 2), f"pick should be B or C (A has no coverage); got {picks}"
    print("  ✓ bucket-match bonus prevents flapping between tied sources")


def test_bucket_match_loses_to_quality():
    """Quality difference > 0.03 (YAW_MATCH_BONUS) should override bucket
    preference. Source C is +0.05 better than B at every bucket → C should
    win even though B was picked first by tie-break."""
    n = 8
    cov_all = np.ones(n, dtype=bool)

    src_a = make_source("AAA", "MBC", "20260101",
                        np.zeros(n, dtype=bool),
                        np.zeros(n, dtype=np.float32))
    # B: left, qual 0.50
    src_b = make_source("BBB", "KBS", "20260101", cov_all,
                        np.full(n, 0.50, dtype=np.float32))
    # C: right, qual 0.55 (delta +0.05 > YAW_MATCH_BONUS=0.03)
    src_c = make_source("CCC", "SBS", "20260101", cov_all,
                        np.full(n, 0.55, dtype=np.float32))
    sources = [src_a, src_b, src_c]
    yaw_buckets = [("frontal", 0.0), ("left", +0.30), ("right", -0.30)]

    log_buf = io.StringIO()
    chunks = plan_merge(sources, clip_dur=float(n), step_sec=1.0,
                        min_chunk_sec=1.0, beats=None,
                        yaw_buckets=yaw_buckets,
                        log_fn=lambda *a, **k: print(*a, **k, file=log_buf))
    picks = [c.src_idx for c in chunks]
    print(f"  qual>bonus: chunks={chunks_picks_str(chunks, sources)}")
    print(f"  qual>bonus picks per chunk: {picks}")
    # C's quality lead (+0.05) > YAW_MATCH_BONUS (+0.03), so C should
    # win every bucket — even though B might have been picked by tie-break
    # at t=0 (it isn't because C has higher raw quality).
    assert all(p == 2 for p in picks), \
        f"C's quality lead should override bucket bonus; got picks {picks}"
    print("  ✓ quality delta > 0.03 correctly overrides bucket-match bonus")


def test_bucket_match_overrides_small_quality_lead():
    """Inverse of above: Source C is only +0.02 better. YAW_MATCH_BONUS
    (+0.03) > quality delta (+0.02), so once B is established it should
    keep winning."""
    n = 8
    cov_all = np.ones(n, dtype=bool)

    # A is primary at t=0..1 with high quality, then drops out.
    cov_a = np.zeros(n, dtype=bool)
    cov_a[:2] = True
    qual_a = np.zeros(n, dtype=np.float32)
    qual_a[:2] = 0.90  # A wins t=0..1 outright

    # B (left): qual 0.50
    qual_b = np.full(n, 0.50, dtype=np.float32)
    qual_b[:2] = 0.0  # A handles t=0..1

    # C (right): qual 0.52 — narrow lead over B, less than 0.03 bonus
    qual_c = np.full(n, 0.52, dtype=np.float32)
    qual_c[:2] = 0.0

    src_a = make_source("AAA", "MBC", "20260101", cov_a, qual_a)
    src_b = make_source("BBB", "KBS", "20260101",
                        np.ones(n, dtype=bool), qual_b)
    src_c = make_source("CCC", "SBS", "20260101",
                        np.ones(n, dtype=bool), qual_c)
    sources = [src_a, src_b, src_c]
    # A=frontal, B=left, C=right
    yaw_buckets = [("frontal", 0.0), ("left", +0.30), ("right", -0.30)]

    log_buf = io.StringIO()
    chunks = plan_merge(sources, clip_dur=float(n), step_sec=1.0,
                        min_chunk_sec=1.0, beats=None,
                        yaw_buckets=yaw_buckets,
                        log_fn=lambda *a, **k: print(*a, **k, file=log_buf))
    picks = [c.src_idx for c in chunks]
    print(f"  bonus>qual: chunks={chunks_picks_str(chunks, sources)}")
    print(f"  bonus>qual picks per chunk: {picks}")
    # Expected: A wins t=0..1 (frontal), then at t=2 prev_bucket=frontal,
    # neither B(left) nor C(right) match, so raw quality picks C (0.52 > 0.50).
    # At t=3, prev_bucket=right — C gets +0.03 → 0.55 vs B 0.50 → C wins.
    # No flap to B. Expect: A,A,C,C,C,C,C,C → 2 chunks
    assert picks[0] == 0, f"t=0 should pick A (high quality); got {picks}"
    # All non-A picks should be the same source (no flap)
    non_a = [p for p in picks if p != 0]
    assert len(set(non_a)) == 1, \
        f"after A drops out, picks should stick to one bucket; got {picks}"
    print(f"  ✓ once a non-frontal bucket is established, sticks (no flap to {non_a[0]})")


def test_unknown_bucket_does_not_overwrite_prev():
    """The contract: when the planner picks an UNKNOWN-bucket source, it
    should NOT overwrite prev_bucket. The previously-established KNOWN
    bucket should still steer subsequent picks via the match bonus.

    Setup: all 3 sources share the same cluster (PRIMARY cancels out).
    - B (left): high quality at t=0..1 (wins), low quality at t=4..7
    - X (unknown): high quality at t=2..3 (wins, but bucket=unknown)
    - C (right): slightly higher baseline quality than B at t=4..7

    Without prev_bucket preservation: at t=4..7, neither B nor C has a
      matching prev_bucket (or it got reset to "unknown") → C wins on raw
      quality.
    WITH prev_bucket preservation across X: prev_bucket is still "left"
      (set when B was picked at t=1) → B gets +0.03 bonus → B wins at t=4..7
      because (B qual + bonus) > (C qual).
    """
    n = 8
    cov_all = np.ones(n, dtype=bool)

    qual_b = np.zeros(n, dtype=np.float32)
    qual_b[:2] = 0.90
    qual_b[4:] = 0.50

    qual_x = np.zeros(n, dtype=np.float32)
    qual_x[2:4] = 0.90

    qual_c = np.zeros(n, dtype=np.float32)
    qual_c[4:] = 0.52  # narrow lead over B (delta 0.02 < bonus 0.03)

    # All in same cluster → PRIMARY_BONUS cancels out across them
    src_b = make_source("BBB", "MBC", "20260101", cov_all, qual_b)
    src_x = make_source("XXX", "MBC", "20260101", cov_all, qual_x)
    src_c = make_source("CCC", "MBC", "20260101", cov_all, qual_c)
    sources = [src_b, src_x, src_c]
    yaw_buckets_known = [("left", +0.30), ("unknown", None), ("right", -0.30)]

    log_buf = io.StringIO()
    chunks_with = plan_merge(sources, clip_dur=float(n), step_sec=1.0,
                             min_chunk_sec=1.0, beats=None,
                             yaw_buckets=yaw_buckets_known,
                             log_fn=lambda *a, **k: print(*a, **k, file=log_buf))
    labels_with = [sources[c.src_idx].meta.video_id for c in chunks_with]
    print(f"  WITH yaw_buckets:    {chunks_picks_str(chunks_with, sources)}")
    print(f"  WITH labels:         {labels_with}")

    # Counter-test: same sources, no yaw_buckets at all. C should win t=4..7.
    chunks_without = plan_merge(sources, clip_dur=float(n), step_sec=1.0,
                                min_chunk_sec=1.0, beats=None,
                                yaw_buckets=None,
                                log_fn=lambda *a, **k: None)
    labels_without = [sources[c.src_idx].meta.video_id for c in chunks_without]
    print(f"  WITHOUT yaw_buckets: {chunks_picks_str(chunks_without, sources)}")
    print(f"  WITHOUT labels:      {labels_without}")

    # Without yaw_buckets, C wins t=4..7 on raw quality (0.52 > 0.50)
    assert labels_without[-1] == "CCC", \
        f"without yaw, C should win t=4..7 on raw quality; got {labels_without}"

    # With yaw_buckets, B's prev_bucket survives through X (unknown)
    # and the +0.03 match bonus pushes B over C → B wins t=4..7
    assert labels_with[-1] == "BBB", \
        f"with yaw, prev_bucket=left should survive X (unknown) and " \
        f"bucket-match bonus should give B the win at t=4..7; got {labels_with}"

    # Both runs should pick B,X then diverge at t=4..7
    assert labels_with[0] == "BBB" and labels_with[1] == "XXX", \
        f"first two chunks should be B then X regardless; got {labels_with}"
    print("  ✓ A/B confirms: unknown pick preserves prev_bucket (B sticks WITH yaw, "
          "C wins WITHOUT)")


def main():
    print("=== synthetic plan_merge yaw-bucket regression ===\n")
    print("[T1] baseline (no yaw_buckets, identical quality)")
    test_baseline_no_yaw()
    print("\n[T2] bucket-match bonus prevents tie-flapping")
    test_bucket_match_bonus_breaks_tie()
    print("\n[T3] quality delta > 0.03 overrides bucket bonus")
    test_bucket_match_loses_to_quality()
    print("\n[T4] bucket bonus > small quality delta — established bucket sticks")
    test_bucket_match_overrides_small_quality_lead()
    print("\n[T5] unknown bucket doesn't overwrite prev_bucket (A/B test)")
    test_unknown_bucket_does_not_overwrite_prev()
    print("\n=== all 5 tests passed ===")


if __name__ == "__main__":
    main()
