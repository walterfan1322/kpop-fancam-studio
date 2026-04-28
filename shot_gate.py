"""TransNetV2 shot-boundary gate.

Detects scene cuts in a downloaded source video so the merge pipeline
can reject multi-cam broadcast edits (typical music-show Performance
Video uploads: 30-80 cuts in a 3-minute song) in favour of single-camera
jikcams, dance-practice recordings, and "one-take" (원테이크) broadcast
clips, which are the only sources that produce a coherent "outfit-swap
fancam" after merging. A hard cut between a jikcam chunk and a multi-cam
chunk looks like a camera-angle change rather than a costume change,
which is exactly the effect we don't want.

Empirical cuts-per-second over a 3-min source:
    single-camera jikcam / dance practice ..... 0.00 – 0.02 cuts/sec
    one-take broadcast (1-2 intro cards)  ..... 0.00 – 0.05 cuts/sec
    multi-cam broadcast edit (rejected)   ..... 0.15 – 0.50 cuts/sec

Gate threshold: 0.10 cuts/sec. Comfortably above the one-take ceiling
and below the multi-cam floor.

Results are cached to `<stem>.shots.json` next to the mp4; TransNetV2
runs at roughly realtime on a modern CPU (~30-150s for 3-4min of 1080p),
too heavy to repeat on every merge planning pass.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


_TRANSNET_CACHE = None  # lazy singleton; type: TransNetV2 | None


def _transnet_model():
    """Load the TransNetV2 model once per process. Weights ship with the
    pip package, so no download. We intentionally stay on CPU: MPS
    produces numerically inconsistent results on Apple silicon for this
    architecture, and the total runtime is dominated by ffmpeg decode
    anyway."""
    global _TRANSNET_CACHE
    if _TRANSNET_CACHE is not None:
        return _TRANSNET_CACHE
    # TransNetV2's decode path shells out to ffmpeg via the ffmpeg-python
    # wrapper. On macOS that needs /opt/homebrew/bin on PATH even when
    # the caller's PATH doesn't have it (e.g. uvicorn started outside
    # Homebrew's shell init). Add it idempotently before the first
    # predict call.
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if extra not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
    from transnetv2_pytorch import TransNetV2  # noqa: PLC0415
    m = TransNetV2()
    m.eval()
    _TRANSNET_CACHE = m
    return m


@dataclass
class ShotInfo:
    """Per-video shot statistics. Serialised to `<stem>.shots.json`.

    `shot_intervals` is a list of (start_sec, end_sec) tuples, one per
    detected shot, in source-time order. Empty list when probing fails
    or the source has zero shots. Downstream consumers (plan_merge's
    shot-aware coverage mask, diagnostics) use this to find long
    contiguous segments inside an otherwise-multicam source.
    """
    num_shots: int
    cuts_per_sec: float
    duration_sec: float
    is_multicam: bool
    cuts_threshold: float
    probe_elapsed_sec: float
    shot_intervals: list[tuple[float, float]] = None  # type: ignore[assignment]
    version: int = 2

    def __post_init__(self) -> None:
        if self.shot_intervals is None:
            self.shot_intervals = []

    def long_shots(self, min_dur: float) -> list[tuple[float, float]]:
        """Return shot intervals with duration >= `min_dur` seconds."""
        return [(s, e) for s, e in self.shot_intervals
                if (e - s) >= min_dur]

    def long_shot_total(self, min_dur: float) -> float:
        """Total duration (seconds) covered by shots >= `min_dur` long."""
        return float(sum(e - s for s, e in self.long_shots(min_dur)))


def _sidecar_for(video: Path) -> Path:
    return video.with_suffix(".shots.json")


def load_cached(video: Path) -> ShotInfo | None:
    """Return cached ShotInfo if the sidecar is present and schema-valid;
    otherwise None. Never raises — corruption just falls back to re-probe."""
    sidecar = _sidecar_for(video)
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Accept both v1 (no intervals) and v2 (with intervals). v1 cached
    # entries return with shot_intervals=[] — downstream consumers must
    # treat "empty intervals + num_shots > 0" as "data unavailable, fall
    # back to aggregate stats only" rather than "zero shots". v1 sidecars
    # will be re-written as v2 the next time probe_shots runs (force=True
    # or threshold change).
    schema_v = data.get("version")
    if schema_v not in (1, 2):
        return None
    try:
        intervals_raw = data.get("shot_intervals") or []
        intervals = [(float(s), float(e)) for s, e in intervals_raw]
        return ShotInfo(
            num_shots=int(data["num_shots"]),
            cuts_per_sec=float(data["cuts_per_sec"]),
            duration_sec=float(data["duration_sec"]),
            is_multicam=bool(data["is_multicam"]),
            cuts_threshold=float(data["cuts_threshold"]),
            probe_elapsed_sec=float(data["probe_elapsed_sec"]),
            shot_intervals=intervals,
        )
    except (KeyError, TypeError, ValueError):
        return None


def probe_shots(video: Path, force: bool = False,
                cuts_per_sec_threshold: float = 0.10,
                log_fn=print) -> ShotInfo:
    """Detect shot boundaries in `video` and return aggregate stats.

    Cached to `<stem>.shots.json`. Pass `force=True` to re-probe (e.g.
    when changing the threshold).

    `cuts_per_sec_threshold` defines the multi-cam gate: any video with
    cut rate at or above this value is flagged `is_multicam=True`.
    Default 0.10 is the empirical sweet spot between broadcast edits
    (~0.15+) and tolerant single-cams with intros/end cards (~0.05).
    """
    cached = None if force else load_cached(video)
    # Schema upgrade: v1 sidecars don't have shot_intervals. If the cache
    # claims num_shots>0 but the intervals list is empty, the data must
    # have come from v1 (or a corrupted v2). Re-probe so downstream
    # shot-aware features have real timestamps to work with.
    if (cached is not None
            and cached.num_shots > 0
            and not cached.shot_intervals):
        log_fn(f"[shot-gate] {video.name}: v1 cache without timestamps — "
               f"re-probing to populate shot_intervals")
        cached = None
    if cached is not None:
        log_fn(f"[shot-gate] cached: {video.name} → "
               f"{cached.num_shots} shots, "
               f"{cached.cuts_per_sec:.3f} cuts/sec, "
               f"multicam={cached.is_multicam}")
        # Re-apply the CURRENT threshold to the cached cut rate so
        # changing the threshold takes effect without re-probing.
        is_mc = cached.cuts_per_sec >= cuts_per_sec_threshold
        if is_mc != cached.is_multicam or (
                abs(cached.cuts_threshold - cuts_per_sec_threshold) > 1e-9):
            cached = ShotInfo(
                num_shots=cached.num_shots,
                cuts_per_sec=cached.cuts_per_sec,
                duration_sec=cached.duration_sec,
                is_multicam=is_mc,
                cuts_threshold=cuts_per_sec_threshold,
                probe_elapsed_sec=cached.probe_elapsed_sec,
                shot_intervals=cached.shot_intervals,
            )
            try:
                _sidecar_for(video).write_text(
                    json.dumps(asdict(cached), indent=2), encoding="utf-8")
            except Exception:
                pass
        return cached

    t0 = time.time()
    model = _transnet_model()
    # detect_scenes() returns a list of {shot_id, start/end frame/time, ...}
    scenes = model.detect_scenes(str(video), threshold=0.5)
    elapsed = time.time() - t0

    num_shots = len(scenes)
    # Pull (start_time, end_time) per scene. Drop any malformed entries
    # quietly — better to miss one boundary than fail the whole probe.
    intervals: list[tuple[float, float]] = []
    for sc in scenes:
        try:
            s = float(sc["start_time"])
            e = float(sc["end_time"])
            if e > s:
                intervals.append((s, e))
        except (KeyError, ValueError, TypeError):
            continue
    # Derive duration from the last scene's end_time. This avoids a second
    # ffprobe roundtrip — TransNetV2 already decoded every frame.
    if intervals:
        duration = intervals[-1][1]
    elif scenes:
        try:
            duration = float(scenes[-1]["end_time"])
        except (KeyError, ValueError, TypeError):
            duration = 0.0
    else:
        duration = 0.0
    # Cuts = shot boundaries = (num_shots - 1). Guard against division by
    # zero on zero-duration / empty inputs.
    cuts = max(0, num_shots - 1)
    cps = (cuts / duration) if duration > 0 else 0.0
    is_mc = cps >= cuts_per_sec_threshold

    info = ShotInfo(
        num_shots=num_shots,
        cuts_per_sec=float(cps),
        duration_sec=float(duration),
        is_multicam=bool(is_mc),
        cuts_threshold=float(cuts_per_sec_threshold),
        probe_elapsed_sec=float(elapsed),
        shot_intervals=intervals,
    )
    try:
        _sidecar_for(video).write_text(
            json.dumps(asdict(info), indent=2), encoding="utf-8")
    except Exception as e:
        log_fn(f"[shot-gate] warn: failed to write sidecar for "
               f"{video.name}: {e}")
    log_fn(f"[shot-gate] probed {video.name}: {num_shots} shots, "
           f"{cps:.3f} cuts/sec over {duration:.1f}s, "
           f"multicam={is_mc} ({elapsed:.1f}s)")
    return info


def main():
    """CLI entry for ad-hoc probing / cache-population.

    Usage: python shot_gate.py <video.mp4> [<video.mp4> ...] [--force]
    """
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", help="mp4 paths to probe")
    ap.add_argument("--force", action="store_true",
                    help="re-probe even when a cached sidecar exists")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="cuts-per-second above which the video is "
                         "flagged multicam (default 0.10)")
    args = ap.parse_args()
    for v in args.videos:
        probe_shots(Path(v), force=args.force,
                    cuts_per_sec_threshold=args.threshold)


if __name__ == "__main__":
    main()
