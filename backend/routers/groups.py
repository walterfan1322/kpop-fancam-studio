"""CRUD on groups.yaml + per-track status (has_mp3, index entry)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import INDEX_PATH, OUTPUT_DIR
from ..mb_members import fetch_members
from ..yaml_store import load, save, serialize_track, track_title, track_url

router = APIRouter(prefix="/api/groups", tags=["groups"])


class TrackIn(BaseModel):
    title: str
    music_url: str | None = None
    album: str | None = None


class TrackOut(BaseModel):
    title: str
    music_url: str | None = None
    album: str | None = None
    has_mp3: bool
    mp3_size_kb: int | None = None
    status: str | None = None  # from index.json: ok | no_audio | no_music_match


class GroupSummary(BaseModel):
    name: str
    artist_name: str
    mb_artist_id: str | None = None
    track_count: int
    mp3_count: int


class GroupDetail(GroupSummary):
    tracks: list[TrackOut]


def _safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name[:120] or "untitled"


def _mp3_path(group: str, title: str) -> Path:
    return OUTPUT_DIR / _safe_filename(group) / f"{_safe_filename(title)}.mp3"


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@router.get("", response_model=list[GroupSummary])
def list_groups():
    data = load()
    out = []
    for name, info in (data.get("groups") or {}).items():
        tracks = info.get("tracks") or []
        mp3_count = sum(
            1 for t in tracks if _mp3_path(name, track_title(t)).exists()
        )
        out.append(GroupSummary(
            name=name,
            artist_name=info.get("artist_name", name),
            mb_artist_id=info.get("mb_artist_id"),
            track_count=len(tracks),
            mp3_count=mp3_count,
        ))
    return out


@router.get("/{group}", response_model=GroupDetail)
def get_group(group: str):
    data = load()
    info = (data.get("groups") or {}).get(group)
    if not info:
        raise HTTPException(404, f"Group {group!r} not found")

    index = _load_index().get(group, {})
    tracks_out = []
    for t in info.get("tracks") or []:
        title = track_title(t)
        if not title:
            continue
        mp3 = _mp3_path(group, title)
        idx = index.get(title, {}) or {}
        tracks_out.append(TrackOut(
            title=title,
            music_url=track_url(t) or None,
            album=(t.get("album") if isinstance(t, dict) else None) or None,
            has_mp3=mp3.exists(),
            mp3_size_kb=(mp3.stat().st_size // 1024) if mp3.exists() else None,
            status=idx.get("status"),
        ))
    return GroupDetail(
        name=group,
        artist_name=info.get("artist_name", group),
        mb_artist_id=info.get("mb_artist_id"),
        track_count=len(tracks_out),
        mp3_count=sum(1 for t in tracks_out if t.has_mp3),
        tracks=tracks_out,
    )


class TrackListIn(BaseModel):
    tracks: list[TrackIn]


@router.put("/{group}/tracks")
def replace_tracks(group: str, body: TrackListIn):
    data = load()
    if group not in data.get("groups", {}):
        raise HTTPException(404, f"Group {group!r} not found")
    serialised = []
    seen = set()
    for t in body.tracks:
        title = t.title.strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        serialised.append(serialize_track(title, t.music_url))
    data["groups"][group]["tracks"] = serialised
    save(data)
    return {"ok": True, "count": len(serialised)}


class MemberOut(BaseModel):
    latin: str
    hangul: str
    chinese: str = ""


@router.get("/{group}/members", response_model=list[MemberOut])
def get_members(group: str, refresh: bool = False):
    data = load()
    info = (data.get("groups") or {}).get(group)
    if not info:
        raise HTTPException(404, f"Group {group!r} not found")

    cached = info.get("members")
    if cached and not refresh:
        return [MemberOut(latin=m.get("latin", ""), hangul=m.get("hangul", ""), chinese=m.get("chinese", "")) for m in cached]

    mb_id = info.get("mb_artist_id")
    if not mb_id:
        return []
    try:
        members = fetch_members(mb_id)
    except Exception:
        return [MemberOut(latin=m.get("latin", ""), hangul=m.get("hangul", ""), chinese=m.get("chinese", "")) for m in (cached or [])]

    # Preserve any manually-curated chinese names when refreshing from MB.
    chi_by_latin = {(m.get("latin") or "").lower(): m.get("chinese", "") for m in (cached or [])}
    chi_by_hangul = {m.get("hangul", ""): m.get("chinese", "") for m in (cached or [])}
    for m in members:
        if "chinese" not in m:
            m["chinese"] = chi_by_latin.get((m.get("latin") or "").lower()) or chi_by_hangul.get(m.get("hangul", ""), "")

    info["members"] = members
    save(data)
    return [MemberOut(**m) for m in members]


@router.delete("/{group}")
def delete_group(group: str):
    data = load()
    if group not in data.get("groups", {}):
        raise HTTPException(404)
    del data["groups"][group]
    save(data)
    return {"ok": True}
