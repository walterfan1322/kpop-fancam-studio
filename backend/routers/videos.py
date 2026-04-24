"""List / download / serve videos and matched clips."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import jobs
from ..config import CLIPS_DIR, PROJECT_ROOT, VIDEOS_DIR

router = APIRouter(prefix="/api/videos", tags=["videos"])


class VideoMeta(BaseModel):
    id: str
    stem: str
    title: str
    url: str
    duration: float | None = None
    path: str                       # relative to project root
    size_mb: float | None = None
    has_file: bool = True
    quality: dict | None = None     # contents of <stem>.quality.json if present


class DownloadIn(BaseModel):
    url: str


class DownloadBatchIn(BaseModel):
    urls: list[str]
    max_height: int = 1080


class SearchIn(BaseModel):
    queries: list[str]
    limit: int = 30
    min_dur: float = 90
    max_dur: float = 330
    min_height: int = 1080
    min_views: int = 5000
    title_any: list[str] = []


class Candidate(BaseModel):
    id: str
    url: str
    title: str
    uploader: str | None = None
    channel_id: str | None = None
    duration: float | None = None
    view_count: int | None = None
    height: int | None = None
    passed: bool
    reject_reasons: list[str] = []
    already_downloaded: bool = False


def _safe(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name[:120] or "untitled"


def _load_meta(json_path: Path) -> VideoMeta | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mp4 = PROJECT_ROOT / data.get("path", "")
    size_mb = (mp4.stat().st_size / 1024 / 1024) if mp4.exists() else None
    quality_path = json_path.with_suffix(".quality.json")
    quality = None
    if quality_path.exists():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            quality = None
    # A video's sidecar metadata ends with `.json`; if this *is* a quality
    # sidecar itself, the `path` field will be missing — skip it.
    if not data.get("path"):
        return None
    return VideoMeta(
        id=data.get("id", ""),
        stem=data.get("stem", json_path.stem),
        title=data.get("title", data.get("stem", "")),
        url=data.get("url", ""),
        duration=data.get("duration"),
        path=data.get("path", ""),
        size_mb=round(size_mb, 2) if size_mb else None,
        has_file=mp4.exists(),
        quality=quality,
    )


def _iter_video_jsons():
    """Yield video sidecar .json paths (recursive, skipping .quality.json)."""
    if not VIDEOS_DIR.exists():
        return
    for p in VIDEOS_DIR.rglob("*.json"):
        if p.stem.endswith(".quality"):
            continue
        yield p


@router.get("", response_model=list[VideoMeta])
def list_videos():
    out: list[VideoMeta] = []
    for p in sorted(_iter_video_jsons()):
        m = _load_meta(p)
        if m:
            out.append(m)
    return out


@router.post("/download")
def start_download(body: DownloadIn):
    if not body.url.strip():
        raise HTTPException(400, "url required")
    job = jobs.start_job("download_video", "download_video.py", [body.url.strip()])
    return {
        "id": job.id, "kind": job.kind, "status": job.status,
        "return_code": job.return_code,
        "started_at": job.started_at, "finished_at": job.finished_at,
        "lines_count": len(job.lines),
    }


@router.post("/download-batch")
def start_download_batch(body: DownloadBatchIn):
    urls = [u.strip() for u in body.urls if u.strip()]
    if not urls:
        raise HTTPException(400, "urls empty")
    args = ["--max-height", str(body.max_height), *urls]
    job = jobs.start_job("download_batch", "download_video.py", args)
    return {
        "id": job.id, "kind": job.kind, "status": job.status,
        "return_code": job.return_code,
        "started_at": job.started_at, "finished_at": job.finished_at,
        "lines_count": len(job.lines),
    }


def _downloaded_ids() -> set[str]:
    ids: set[str] = set()
    for p in _iter_video_jsons():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("id"):
                ids.add(data["id"])
        except (OSError, json.JSONDecodeError):
            continue
    return ids


def _find_video(stem: str) -> Path | None:
    """Locate a video mp4 by its stem, searching recursively."""
    safe = _safe(stem)
    matches = list(VIDEOS_DIR.rglob(f"{safe}.mp4"))
    return matches[0] if matches else None


@router.post("/search", response_model=list[Candidate])
def search(body: SearchIn):
    """Search YouTube via yt-dlp, return filtered candidates (no downloads)."""
    # Local import to keep unused path cheap when this endpoint is never hit.
    import yt_dlp  # noqa: PLC0415

    # extract_flat="in_playlist" gives us id/title/duration/view_count quickly
    # (~0.2s per query) but omits format/height. A full extract is ~10s/entry
    # and blocks the request, which is impractical for interactive search.
    # We rely on YouTube's default upload quality being >= 1080p for most
    # fancams in 2024+; the quality probe later verifies pixel-level quality.
    opts = {
        "quiet": True, "skip_download": True, "noplaylist": False,
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
        "extract_flat": "in_playlist",
        "playlistend": body.limit,
    }
    entries: list[dict] = []
    for q in body.queries:
        q = q.strip()
        if not q:
            continue
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch{body.limit}:{q}", download=False)
            entries.extend([e for e in (info.get("entries") or []) if e])
        except Exception as e:
            # surface the error as a synthetic failed candidate so UI can see
            entries.append({"id": f"err:{hash(q)}", "title": f"[query failed: {q}] {e}"})

    have = _downloaded_ids()
    seen: set[str] = set()
    out: list[Candidate] = []
    title_any_lc = [t.lower() for t in body.title_any if t.strip()]

    for e in entries:
        vid = e.get("id") or ""
        if not vid or vid in seen:
            continue
        seen.add(vid)
        dur = e.get("duration") or 0
        views = e.get("view_count") or 0
        title = e.get("title") or ""
        heights = [f.get("height") for f in (e.get("formats") or []) if isinstance(f.get("height"), int)]
        height = max(heights) if heights else (e.get("height") or 0)

        reasons: list[str] = []
        if dur < body.min_dur or dur > body.max_dur:
            reasons.append(f"duration={int(dur)}s")
        # Flat extract doesn't expose height — skip the check; pixel quality
        # is verified post-download by quality_probe.py.
        if height and height < body.min_height:
            reasons.append(f"height={height}p")
        if views < body.min_views:
            reasons.append(f"views={views}")
        if title_any_lc and not any(k in title.lower() for k in title_any_lc):
            reasons.append("no-keyword")

        out.append(Candidate(
            id=vid, url=f"https://www.youtube.com/watch?v={vid}", title=title,
            uploader=e.get("uploader") or e.get("channel"),
            channel_id=e.get("channel_id"),
            duration=dur, view_count=views, height=height,
            passed=not reasons,
            reject_reasons=reasons,
            already_downloaded=(vid in have),
        ))
    out.sort(key=lambda c: (not c.passed, -(c.view_count or 0)))
    return out


class ProbeIn(BaseModel):
    video_stem: str


class OneshotIn(BaseModel):
    group: str
    song: str
    member_lat: str
    member_han: str = ""
    count: int = 3
    delogo_corners: list[str] = []  # any of "tl", "tr", "bl", "br"
    force_landscape: bool = False
    # When >=2, fuse up to N matched source videos into one merged clip
    # that hops between angles second-by-second. 1 = legacy behaviour
    # (one clip per source, --count of them).
    merge_sources: int = 1
    # How to stitch chunks together in a merged clip.
    #   "xfade"    — default, 0.5s cross-dissolve between sources. Hides
    #                framing/angle jumps when sources aren't tightly
    #                aligned. Safe universal default.
    #   "hard_cut" — zero-frame transition, for the "outfit-swap fancam"
    #                look. Only makes sense when merge_sources>=2 AND
    #                sources are same-angle same-framing jikcams.
    #                Ignored when merge_sources==1.
    merge_style: str = "xfade"
    # M3: run RTMPose-m on the target bbox per frame to anchor the
    # dancer's head at a fixed y-fraction across sources. Kills head-pop
    # on hard-cut merges. Ignored when merge_sources==1 (single-source
    # clips don't have cross-source head-position drift).
    use_pose: bool = False


@router.post("/oneshot")
def start_oneshot(body: OneshotIn):
    if not body.group.strip() or not body.song.strip() or not body.member_lat.strip():
        raise HTTPException(400, "group, song, member_lat all required")
    args = [
        "--group", body.group,
        "--song", body.song,
        "--member-lat", body.member_lat,
        "--count", str(max(1, min(10, body.count))),
    ]
    if body.member_han.strip():
        args += ["--member-han", body.member_han.strip()]
    valid_corners = [c for c in body.delogo_corners if c in ("tl", "tr", "bl", "br", "auto")]
    if valid_corners:
        args += ["--delogo-corners", ",".join(valid_corners)]
    if body.force_landscape:
        args += ["--force-landscape"]
    merge_n = max(1, min(6, int(body.merge_sources)))
    if merge_n >= 2:
        args += ["--merge-sources", str(merge_n)]
        # merge_style only has meaning when we actually have >1 source
        # to concat. Validate to reject anything unexpected rather than
        # letting oneshot_fancam choke on argparse choices=.
        style = (body.merge_style or "xfade").strip()
        if style not in ("xfade", "hard_cut"):
            raise HTTPException(
                400, f"merge_style must be 'xfade' or 'hard_cut', got: {style!r}")
        args += ["--merge-style", style]
        if body.use_pose:
            args += ["--pose"]
    job = jobs.start_job("oneshot", "oneshot_fancam.py", args)
    return {
        "id": job.id, "kind": job.kind, "status": job.status,
        "return_code": job.return_code,
        "started_at": job.started_at, "finished_at": job.finished_at,
        "lines_count": len(job.lines),
    }


@router.post("/probe")
def start_probe(body: ProbeIn):
    mp4 = VIDEOS_DIR / f"{body.video_stem}.mp4"
    if not mp4.exists():
        raise HTTPException(404, f"video not found: {mp4.name}")
    job = jobs.start_job("probe", "quality_probe.py", ["--video", str(mp4)])
    return {
        "id": job.id, "kind": job.kind, "status": job.status,
        "return_code": job.return_code,
        "started_at": job.started_at, "finished_at": job.finished_at,
        "lines_count": len(job.lines),
    }


@router.delete("/{stem}")
def delete_video(stem: str):
    mp4 = _find_video(stem)
    if not mp4:
        raise HTTPException(404)
    removed = []
    for suffix in (".mp4", ".json", ".quality.json", ".bboxes.npz"):
        p = mp4.with_suffix("").with_suffix(suffix) if suffix != ".mp4" else mp4
        # For .quality.json / .bboxes.npz the suffix chain matters — use stem.
        p = mp4.parent / f"{mp4.stem}{suffix}"
        if p.exists():
            p.unlink()
            removed.append(suffix)
    return {"ok": True, "removed": removed}


@router.get("/{stem}/file")
def serve_video(stem: str):
    mp4 = _find_video(stem)
    if not mp4:
        raise HTTPException(404)
    p = mp4.resolve()
    if not str(p).startswith(str(VIDEOS_DIR.resolve())):
        raise HTTPException(400)
    return FileResponse(str(p), media_type="video/mp4")


# ---------------- clips ----------------

class ClipOut(BaseModel):
    group: str
    title: str
    song: str
    path: str
    size_mb: float | None = None
    # Unix timestamp so the UI can show most-recent-first without parsing
    # filenames. 0.0 when stat fails — callers should tolerate it.
    mtime: float = 0.0


@router.get("/clips", response_model=list[ClipOut])
def list_clips():
    """List clips under clips/<Group>/<Song>/<file>.mp4.

    Clips with a `.mp4.hidden` sidecar are omitted — the frontend's
    'dismiss' (X) button writes that sidecar when the user wants a clip
    hidden from the Recent Clips panel permanently. The mp4 file itself
    is left alone (still retrievable via the admin view, and still
    subject to the normal sweep/.keep policy)."""
    if not CLIPS_DIR.exists():
        return []
    out: list[ClipOut] = []
    for group_dir in sorted(p for p in CLIPS_DIR.iterdir() if p.is_dir()):
        for song_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for mp4 in sorted(song_dir.glob("*.mp4")):
                if mp4.with_suffix(mp4.suffix + ".hidden").exists():
                    continue
                st = mp4.stat()
                size_mb = st.st_size / 1024 / 1024
                out.append(ClipOut(
                    group=group_dir.name,
                    song=song_dir.name,
                    title=mp4.stem,
                    path=str(mp4.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    size_mb=round(size_mb, 2),
                    mtime=st.st_mtime,
                ))
    return out


@router.get("/clips/{group}/{song}/{title}")
def serve_clip(group: str, song: str, title: str, keep: int = 0):
    """Serve the clip. If `?keep=1`, touch a `.keep` sidecar so the sweeper
    that runs at every oneshot start won't delete it. This is how the
    frontend's download button tells the server 'the user actually wanted
    this one' — otherwise it'll be cleaned up within N minutes."""
    p = (CLIPS_DIR / _safe(group) / _safe(song) / f"{_safe(title)}.mp4").resolve()
    if not str(p).startswith(str(CLIPS_DIR.resolve())):
        raise HTTPException(400)
    if not p.exists():
        raise HTTPException(404)
    if keep:
        keep_path = p.with_suffix(p.suffix + ".keep")
        try:
            keep_path.touch(exist_ok=True)
        except OSError:
            pass  # best-effort; serving the file is more important
    return FileResponse(str(p), media_type="video/mp4")


@router.post("/clips/{group}/{song}/{title}/keep")
def mark_clip_kept(group: str, song: str, title: str):
    """Explicit 'mark as kept' endpoint. Idempotent; used by the frontend if
    we want to flag a clip without triggering a download."""
    p = (CLIPS_DIR / _safe(group) / _safe(song) / f"{_safe(title)}.mp4").resolve()
    if not str(p).startswith(str(CLIPS_DIR.resolve())):
        raise HTTPException(400)
    if not p.exists():
        raise HTTPException(404)
    p.with_suffix(p.suffix + ".keep").touch(exist_ok=True)
    return {"ok": True}


@router.post("/clips/{group}/{song}/{title}/hide")
def mark_clip_hidden(group: str, song: str, title: str):
    """'Hide from listing' endpoint. Creates a `.mp4.hidden` sidecar next
    to the clip; list_clips skips anything with that sidecar. The mp4
    itself stays on disk so the sweep/keep policy is unchanged — this is
    purely a UI-level 'I don't want to see this in Recent Clips' flag
    that persists across browser refreshes and tabs. Idempotent."""
    p = (CLIPS_DIR / _safe(group) / _safe(song) / f"{_safe(title)}.mp4").resolve()
    if not str(p).startswith(str(CLIPS_DIR.resolve())):
        raise HTTPException(400)
    if not p.exists():
        raise HTTPException(404)
    p.with_suffix(p.suffix + ".hidden").touch(exist_ok=True)
    return {"ok": True}


@router.delete("/clips/{group}/{song}/{title}")
def delete_clip(group: str, song: str, title: str):
    p = CLIPS_DIR / _safe(group) / _safe(song) / f"{_safe(title)}.mp4"
    if not p.exists():
        raise HTTPException(404)
    p.unlink()
    keep = p.with_suffix(p.suffix + ".keep")
    if keep.exists():
        keep.unlink()
    return {"ok": True}
