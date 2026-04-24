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
    """Per-video shot statistics. Serialised to `<stem>.shots.json`."""
    num_shots: int
    cuts_per_sec: float
    duration_sec: float
    is_multicam: bool
    cuts_threshold: float
    probe_elapsed_sec: float
    version: int = 1


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
    if data.get("version") != 1:
        return None
    # Strip any extra fields so future additions don't crash old readers.
    try:
        return ShotInfo(
            num_shots=int(data["num_shots"]),
            cuts_per_sec=float(data["cuts_per_sec"]),
            duration_sec=float(data["duration_sec"]),
            is_multicam=bool(data["is_multicam"]),
            cuts_threshold=float(data["cuts_threshold"]),
            probe_elapsed_sec=float(data["probe_elapsed_sec"]),
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
    # Derive duration from the last scene's end_time. This avoids a second
    # ffprobe roundtrip — TransNetV2 already decoded every frame.
    if scenes:
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
