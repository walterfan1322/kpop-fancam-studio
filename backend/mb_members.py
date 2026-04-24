"""Fetch group members from MusicBrainz.

Uses the `artist-rels` include on the band artist record; each `member of band`
relation points to a member artist. We extract the Latin display name and any
Hangul alias.
"""
from __future__ import annotations

import re
import time

import requests

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "kpop_tiktok_audio/0.1 ( https://github.com/local/private )"
RATE_LIMIT_SEC = 1.1

_last_req_at = 0.0

_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")


def _has_hangul(s: str) -> bool:
    return bool(_HANGUL_RE.search(s or ""))


def _is_latin(s: str) -> bool:
    return bool(s) and not _has_hangul(s) and all(ord(c) < 0x3000 for c in s)


def _get(path: str, params: dict) -> dict:
    global _last_req_at
    wait = RATE_LIMIT_SEC - (time.time() - _last_req_at)
    if wait > 0:
        time.sleep(wait)
    params = {**params, "fmt": "json"}
    r = requests.get(
        f"{MB_BASE}/{path}",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    _last_req_at = time.time()
    r.raise_for_status()
    return r.json()


def _pick_names(artist: dict) -> tuple[str, str]:
    """Return (latin, hangul) from an MB artist object (includes aliases)."""
    name = (artist.get("name") or "").strip()
    sort_name = (artist.get("sort-name") or "").strip()
    aliases = artist.get("aliases") or []

    latin = ""
    hangul = ""

    if _has_hangul(name):
        hangul = name
    elif _is_latin(name):
        latin = name

    for a in aliases:
        n = (a.get("name") or "").strip()
        if not n:
            continue
        if not hangul and _has_hangul(n):
            hangul = n
        if not latin and _is_latin(n) and (a.get("locale") in (None, "", "en") or a.get("type") == "Artist name"):
            latin = n

    if not latin and sort_name and _is_latin(sort_name):
        # "Jang, Wonyoung" -> "Wonyoung Jang"
        parts = [p.strip() for p in sort_name.split(",", 1)]
        latin = " ".join(reversed(parts)) if len(parts) == 2 else sort_name

    return latin, hangul


def fetch_members(mb_artist_id: str) -> list[dict]:
    """Return list of {"latin": str, "hangul": str} for band members."""
    data = _get(f"artist/{mb_artist_id}", {"inc": "artist-rels+aliases"})
    out: list[dict] = []
    seen: set[str] = set()
    for rel in data.get("relations") or []:
        if rel.get("type") != "member of band":
            continue
        art = rel.get("artist") or {}
        art_id = art.get("id")
        if not art_id or art_id in seen:
            continue
        # Fetch with aliases (the nested artist in relations has none by default)
        full = _get(f"artist/{art_id}", {"inc": "aliases"})
        latin, hangul = _pick_names(full)
        if not latin and not hangul:
            continue
        seen.add(art_id)
        out.append({"latin": latin, "hangul": hangul})
    return out
