"""Kick off match_video.py jobs against a downloaded video."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import jobs
from ..config import VIDEOS_DIR

router = APIRouter(prefix="/api/match", tags=["match"])


class MatchIn(BaseModel):
    group: str
    video_stem: str
    threshold: float = 0.6
    margin: float = 0.03
    extract: bool = True


@router.post("")
def start_match(body: MatchIn):
    mp4 = VIDEOS_DIR / f"{body.video_stem}.mp4"
    if not mp4.exists():
        raise HTTPException(404, f"video not found: {mp4.name}")
    args = [
        "--group", body.group,
        "--video", str(mp4),
        "--threshold", str(body.threshold),
        "--margin", str(body.margin),
    ]
    if not body.extract:
        args.append("--no-extract")
    job = jobs.start_job("match", "match_video.py", args)
    return {
        "id": job.id, "kind": job.kind, "status": job.status,
        "return_code": job.return_code,
        "started_at": job.started_at, "finished_at": job.finished_at,
        "lines_count": len(job.lines),
    }
