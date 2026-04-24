"""Broadcaster + date parser for K-pop stage videos.

Used by the multi-source merge feature: we cluster downloaded candidates by
(broadcaster, date) so cuts across the same live performance can be treated
as different angles of the *same* event rather than unrelated clips.

Input is the sidecar `.json` that `download_video.py` writes next to each
`.mp4`. It has these keys:
    id, stem, title, url, duration, path, extractor, group, song

This module reads the `title` and picks out:
  * broadcaster — one of a small keyword list (SBS Inkigayo, MBC Music Core,
    KBS Music Bank, Mnet M Countdown, SBS MTV The Show, MBC M Show
    Champion, MBC Show Champion, MBC every1 Show Champion, Simply K-Pop, …)
  * date        — first plausible date-like token (YYMMDD, YYYYMMDD, or
    YYYY-MM-DD). We keep the normalised YYYYMMDD form.

If neither can be confidently extracted, the candidate is returned as a
"self-cluster": broadcaster=None, date=None → cluster key = a unique
sentinel per video id, so it stays isolated rather than pooling with other
parse-failures (B=self-cluster in the design decision).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keyword list. Order matters: the first match wins. Include both English
# and Hangul forms since Korean titles often use the Hangul show name.
_BROADCASTER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Inkigayo",        re.compile(r"inkigayo|인기가요", re.I)),
    ("Music Bank",      re.compile(r"music\s*bank|뮤직\s*뱅크|뮤뱅", re.I)),
    ("Music Core",      re.compile(r"music\s*core|음악\s*중심|음중", re.I)),
    ("M Countdown",     re.compile(r"m\s*countdown|mcountdown|엠카운트다운|엠카", re.I)),
    ("Show Champion",   re.compile(r"show\s*champion|쇼\s*챔피언|쇼챔", re.I)),
    ("The Show",        re.compile(r"the\s*show|더\s*쇼", re.I)),
    ("Simply K-Pop",    re.compile(r"simply\s*k[-\s]?pop", re.I)),
    ("KBS Open Concert",re.compile(r"open\s*concert|열린\s*음악회", re.I)),
]

# Date patterns, tried in order. All produce a YYYYMMDD string.
#
# YYYY-MM-DD or YYYY.MM.DD or YYYY/MM/DD → "20220826"
# YYYYMMDD                                → "20220826"
# YYMMDD (6 digits)                       → "20220826"  (prefix with 20)
#
# We only accept "reasonable" K-pop dates: year 2010–2029 for the 4-digit
# form, year 10–29 for the 2-digit form. That filters out track counts like
# "220828" appearing accidentally in non-date contexts.
_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?<!\d)(20[12]\d)[-./](\d{2})[-./](\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20[12]\d)(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)([12]\d)(\d{2})(\d{2})(?!\d)"),
]


@dataclass(frozen=True)
class SourceMeta:
    """What we know about a downloaded video for clustering."""
    video_id: str
    path: Path
    title: str
    duration: float | None
    broadcaster: str | None
    date: str | None  # YYYYMMDD or None

    @property
    def cluster_key(self) -> str:
        """(broadcaster, date) tuple serialised. When either is missing, fall
        back to a per-video sentinel so the candidate forms a cluster of one
        (decision B=self-cluster)."""
        if self.broadcaster and self.date:
            return f"{self.broadcaster}|{self.date}"
        return f"__solo__|{self.video_id}"

    @property
    def cluster_label(self) -> str:
        """Human-friendly string for logs."""
        if self.broadcaster and self.date:
            return f"{self.broadcaster} {self.date}"
        return f"solo({self.video_id})"


def parse_broadcaster(title: str) -> str | None:
    for name, pat in _BROADCASTER_PATTERNS:
        if pat.search(title):
            return name
    return None


def parse_date(title: str) -> str | None:
    for pat in _DATE_PATTERNS:
        m = pat.search(title)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 3:
            y, mo, d = groups
            if len(y) == 2:
                # Two-digit year → prepend "20". Only 2010-2029 accepted by the
                # regex so this is unambiguous for the foreseeable future.
                y = "20" + y
            try:
                mo_i = int(mo)
                d_i = int(d)
                if not (1 <= mo_i <= 12 and 1 <= d_i <= 31):
                    continue
            except ValueError:
                continue
            return f"{y}{mo:>02}{d:>02}" if len(mo) == 2 else f"{y}{int(mo):02d}{int(d):02d}"
    return None


def load_source_meta(json_path: Path) -> SourceMeta | None:
    """Load a sidecar `.json` (the one written by download_video.py) and
    return SourceMeta. Returns None if the file is unreadable or has no
    usable title; callers can treat that as a hard parse-failure."""
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    vid = d.get("id") or ""
    title = d.get("title") or ""
    if not vid or not title:
        return None
    mp4 = json_path.with_suffix(".mp4")
    if not mp4.exists():
        return None
    return SourceMeta(
        video_id=vid,
        path=mp4,
        title=title,
        duration=float(d["duration"]) if d.get("duration") else None,
        broadcaster=parse_broadcaster(title),
        date=parse_date(title),
    )


def cluster_by_event(metas: Iterable[SourceMeta]) -> dict[str, list[SourceMeta]]:
    """Group sources by (broadcaster, date); parse-failures become singleton
    clusters (keyed by video_id).

    Returned dict preserves insertion order, so callers can use the first
    cluster key as deterministic tie-break ordering.
    """
    out: dict[str, list[SourceMeta]] = {}
    for m in metas:
        out.setdefault(m.cluster_key, []).append(m)
    return out
