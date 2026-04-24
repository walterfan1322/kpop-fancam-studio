"""Fetch a K-pop group's discography from MusicBrainz and merge into groups.yaml.

Usage:
    python fetch_discography.py "IVE"
    python fetch_discography.py "IVE" --artist-id 8f85c938-2d1d-4b35-8fb4-34f8f6b2f8c7
    python fetch_discography.py "IVE" --limit 30

MusicBrainz is free and requires no auth, but rate-limits to 1 request/sec and
requires a User-Agent. Re-running merges into existing yaml entries.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "groups.yaml"

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "kpop_tiktok_audio/0.1 ( https://github.com/local/private )"
RATE_LIMIT_SEC = 1.1  # MB requires <= 1 req/sec; leave margin

NOISE_PATTERNS = [
    r"\(inst(rumental)?\)",
    r"\(off\s*vocal\)",
    r"\(mr\)",
    r"\bremix\b",
    r"\bremaster(ed)?\b",
    r"\blive\b",
    r"\bacoustic\b",
    r"\bsped[- ]up\b",
    r"\bslowed\b",
    r"\bkaraoke\b",
    r"\bdemo\b",
    # Language variants in either (...) or -...- form
    r"[(\-]\s*(japanese|english|chinese|korean|mandarin)\s*ver",
]


def is_noise(title: str) -> bool:
    t = title.lower()
    return any(re.search(p, t) for p in NOISE_PATTERNS)


def canonical(title: str) -> str:
    """Normalise title for dedup: lowercase, strip all paren/dash-inner groups."""
    t = title
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)
    t = re.sub(r"\s*-[^-]+-\s*", " ", t)  # -Japanese ver.-
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


_last_req_at = 0.0


def mb_get(path: str, params: dict) -> dict:
    global _last_req_at
    wait = RATE_LIMIT_SEC - (time.time() - _last_req_at)
    if wait > 0:
        time.sleep(wait)
    params = {**params, "fmt": "json"}
    r = requests.get(
        f"{MB_BASE}/{path}",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    _last_req_at = time.time()
    r.raise_for_status()
    return r.json()


def resolve_artist(query: str) -> dict:
    res = mb_get("artist", {"query": f'artist:"{query}"', "limit": 10})
    items = res.get("artists") or []
    if not items:
        # Loosen query
        res = mb_get("artist", {"query": query, "limit": 10})
        items = res.get("artists") or []
    if not items:
        sys.exit(f"No artist found for query: {query!r}")
    # Prefer exact name, then highest score
    items.sort(key=lambda a: (a["name"].lower() != query.lower(), -a.get("score", 0)))
    return items[0]


def fetch_tracks(artist_id: str, artist_name: str, limit: int | None) -> list[str]:
    """Pull all recordings for this artist, dedupe, filter noise."""
    seen: dict[str, str] = {}  # canonical -> original title
    # Strip "<ArtistName> on " prefix that MB uses for some appearance credits
    prefix_re = re.compile(rf"^{re.escape(artist_name)}\s+on\s+", re.IGNORECASE)
    offset = 0
    page = 100
    while True:
        res = mb_get(
            "recording",
            {"artist": artist_id, "limit": page, "offset": offset},
        )
        recs = res.get("recordings") or []
        if not recs:
            break
        for r in recs:
            if r.get("video"):
                continue
            title = r.get("title") or ""
            title = prefix_re.sub("", title).strip()
            if not title or is_noise(title):
                continue
            key = canonical(title)
            if not key:
                continue
            if key not in seen:
                seen[key] = title
        total = res.get("recording-count", 0)
        offset += len(recs)
        print(f"  fetched {offset}/{total} recordings ({len(seen)} unique)")
        if offset >= total:
            break

    titles = list(seen.values())
    if limit:
        titles = titles[:limit]
    return titles


def load_yaml() -> dict:
    if YAML_PATH.exists():
        with YAML_PATH.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("groups", {})
    return data


def save_yaml(data: dict) -> None:
    with YAML_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", help='Group name e.g. "IVE"')
    ap.add_argument("--artist-id", help="Skip search, use this MusicBrainz artist MBID")
    ap.add_argument("--limit", type=int, help="Keep only the first N unique tracks")
    args = ap.parse_args()

    if args.artist_id:
        artist = mb_get(f"artist/{args.artist_id}", {})
    else:
        artist = resolve_artist(args.group)

    print(f"Resolved artist: {artist['name']} (mbid={artist['id']})")
    titles = fetch_tracks(artist["id"], artist["name"], args.limit)
    print(f"Fetched {len(titles)} unique tracks")

    data = load_yaml()
    existing = data["groups"].get(args.group, {}) or {}
    existing.setdefault("artist_name", artist["name"])
    existing["mb_artist_id"] = artist["id"]
    existing_tracks = existing.get("tracks", []) or []

    def _title_of(t):
        return t if isinstance(t, str) else (t.get("title") or t.get("name") or "")

    existing_keys = {canonical(_title_of(t)) for t in existing_tracks if _title_of(t)}
    merged = list(existing_tracks)
    for title in titles:
        if canonical(title) not in existing_keys:
            merged.append(title)
            existing_keys.add(canonical(title))

    existing["tracks"] = merged
    data["groups"][args.group] = existing
    save_yaml(data)
    print(f"Wrote {YAML_PATH} ({len(merged)} tracks for {args.group})")


if __name__ == "__main__":
    main()
