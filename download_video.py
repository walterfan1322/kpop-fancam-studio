"""Download one or more YouTube (or other yt-dlp supported) videos.

Keeps a sidecar .json with id, title, url, duration, path per video so the
backend can list videos without re-probing. Idempotent: skips download if the
mp4 already exists.

Usage:
    python download_video.py <url> [<url> ...] [--max-height 1080]

Prints `META {...}` per successful video so the backend can parse each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import static_ffmpeg
import yt_dlp

# Force utf-8 so Korean/Japanese titles don't crash on Windows cp950 console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Put our bundled ffmpeg on PATH so yt-dlp can merge video+audio streams.
static_ffmpeg.add_paths()

ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "videos"


def safe_stem(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip()
    return s[:80] or "untitled"


def video_id(url: str, info_id: str | None) -> str:
    if info_id:
        return info_id
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def download(url: str, out_dir: Path, max_height: int = 1080,
             group: str | None = None, song: str | None = None) -> dict:
    # When group/song are provided, file is placed under videos/<group>/<song>/
    # so the Finder browsing stays organised. Otherwise fall back to flat.
    target_dir = out_dir
    if group and song:
        target_dir = out_dir / safe_stem(group) / safe_stem(song)
    target_dir.mkdir(parents=True, exist_ok=True)

    # First pass: extract info only to compute a stable filename.
    # Node is used as the JS runtime for YouTube's signature decryption; without
    # it yt-dlp falls back to incomplete formats and many videos report
    # "Video unavailable".
    probe_opts = {
        "quiet": True, "skip_download": True, "noplaylist": True,
        "js_runtimes": {"node": {}},
        # Fetch the ejs challenge-solver script from yt-dlp's GitHub so recent
        # YouTube n/sig challenges can be resolved. Without this many 2024+
        # uploads come back as "Video unavailable".
        "remote_components": ["ejs:github"],
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    vid = video_id(url, info.get("id"))
    title = info.get("title") or vid
    stem = f"{safe_stem(title)}__{vid}"
    mp4_path = target_dir / f"{stem}.mp4"
    meta_path = target_dir / f"{stem}.json"

    # If this id was already downloaded under the flat layout (or another
    # group/song), reuse that file instead of re-downloading.
    if not mp4_path.exists():
        existing = next((
            p for p in out_dir.rglob(f"*__{vid}.mp4")
            if p.is_file()
        ), None)
        if existing:
            mp4_path = existing
            meta_path = existing.with_suffix(".json")
            stem = existing.stem

    if not mp4_path.exists():
        opts = {
            "quiet": True,
            "noprogress": True,
            "noplaylist": True,
            "js_runtimes": {"node": {}},
        # Fetch the ejs challenge-solver script from yt-dlp's GitHub so recent
        # YouTube n/sig challenges can be resolved. Without this many 2024+
        # uploads come back as "Video unavailable".
        "remote_components": ["ejs:github"],
            # Prefer MP4 container for easy playback + ffmpeg compatibility.
            # Cap at max_height because 4K is ~1 GB/min and we only need 1080p
            # for 9:16 crops.
            "format": (
                f"bv*[ext=mp4][height<={max_height}]+ba[ext=m4a]/"
                f"b[ext=mp4][height<={max_height}]/"
                f"bv*[height<={max_height}]+ba/best[height<={max_height}]/best"
            ),
            "merge_output_format": "mp4",
            "outtmpl": str(target_dir / f"{stem}.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    meta = {
        "id": vid,
        "stem": stem,
        "title": title,
        "url": url,
        "duration": info.get("duration"),
        "path": str(mp4_path.relative_to(ROOT)).replace("\\", "/"),
        "extractor": info.get("extractor"),
    }
    if group:
        meta["group"] = group
    if song:
        meta["song"] = song
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-height", type=int, default=1080)
    ap.add_argument("--group", default=None, help="group name for organised path")
    ap.add_argument("--song", default=None, help="song title for organised path")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ok = 0
    for i, url in enumerate(args.urls, 1):
        print(f"[{i}/{len(args.urls)}] {url}", flush=True)
        try:
            meta = download(url, out_dir, max_height=args.max_height,
                            group=args.group, song=args.song)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            continue
        print(f"  [ok] {meta['stem']}  ({meta.get('duration')}s)", flush=True)
        print("META " + json.dumps(meta, ensure_ascii=False), flush=True)
        ok += 1
    print(f"[done] {ok}/{len(args.urls)} downloaded", flush=True)
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
