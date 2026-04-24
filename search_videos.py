"""Search YouTube for fancam candidates and filter by metadata.

Runs yt-dlp search (`ytsearch{N}:`) on one or more queries, combines + dedupes
results by video id, then filters:
  - duration in [min_dur, max_dur]
  - resolution >= min_height (default 1080)
  - view count >= min_views
  - title contains at least one of --title-any (case-insensitive)

Prints one JSON object to stdout describing candidates; no downloads happen.

Usage:
    python search_videos.py --query "IVE Wonyoung fancam" --query "IVE 장원영 직캠"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).parent


def probe_query(query: str, limit: int) -> list[dict]:
    # extract_flat="in_playlist" returns search entries quickly without hitting
    # each video endpoint, but skips format/resolution info. We need formats,
    # so we pay the cost of a full extract per result.
    opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": False,
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
        # Fetch per-entry info (formats, duration, view_count).
        "extract_flat": False,
        "playlistend": limit,
    }
    results = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        for e in info.get("entries", []) or []:
            if not e:
                continue
            results.append(e)
    return results


def max_height(info: dict) -> int:
    heights = []
    for f in info.get("formats") or []:
        h = f.get("height")
        if isinstance(h, int):
            heights.append(h)
    return max(heights) if heights else (info.get("height") or 0)


def filter_candidates(
    entries: list[dict],
    *,
    min_dur: float,
    max_dur: float,
    min_height: int,
    min_views: int,
    title_any: list[str],
) -> list[dict]:
    title_any_lc = [t.lower() for t in title_any if t.strip()]
    seen: set[str] = set()
    out: list[dict] = []
    for e in entries:
        vid = e.get("id") or ""
        if not vid or vid in seen:
            continue
        seen.add(vid)
        dur = e.get("duration") or 0
        views = e.get("view_count") or 0
        title = e.get("title") or ""
        height = max_height(e)
        reasons: list[str] = []
        if dur < min_dur or dur > max_dur:
            reasons.append(f"duration={dur}")
        if height < min_height:
            reasons.append(f"height={height}")
        if views < min_views:
            reasons.append(f"views={views}")
        if title_any_lc and not any(k in title.lower() for k in title_any_lc):
            reasons.append("no-keyword")
        passed = not reasons
        out.append({
            "id": vid,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title,
            "uploader": e.get("uploader") or e.get("channel"),
            "channel_id": e.get("channel_id"),
            "duration": dur,
            "view_count": views,
            "height": height,
            "passed": passed,
            "reject_reasons": reasons,
        })
    # passing first, then by view_count desc
    out.sort(key=lambda c: (not c["passed"], -(c["view_count"] or 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="append", required=True,
                    help="YouTube search query; may be given multiple times")
    ap.add_argument("--limit", type=int, default=30,
                    help="max results per query before filtering")
    ap.add_argument("--min-dur", type=float, default=90)
    ap.add_argument("--max-dur", type=float, default=330)
    ap.add_argument("--min-height", type=int, default=1080)
    ap.add_argument("--min-views", type=int, default=5000)
    ap.add_argument("--title-any", action="append", default=[],
                    help="title must contain one of these (case-insensitive); "
                         "can repeat. e.g. --title-any fancam --title-any 직캠")
    args = ap.parse_args()

    all_entries: list[dict] = []
    for q in args.query:
        print(f"[search] {q!r}", flush=True)
        try:
            entries = probe_query(q, args.limit)
        except Exception as e:
            print(f"[warn] query failed: {e}", file=sys.stderr)
            continue
        print(f"  -> {len(entries)} raw", flush=True)
        all_entries.extend(entries)

    candidates = filter_candidates(
        all_entries,
        min_dur=args.min_dur, max_dur=args.max_dur,
        min_height=args.min_height, min_views=args.min_views,
        title_any=args.title_any,
    )
    passing = sum(1 for c in candidates if c["passed"])
    print(f"[done] {passing}/{len(candidates)} passed filters", flush=True)

    print("RESULT " + json.dumps({"candidates": candidates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
