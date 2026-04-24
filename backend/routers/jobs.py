"""Start and stream background jobs (fetch / download / resolve)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    return_code: int | None
    started_at: float
    finished_at: float
    lines_count: int
    # Script + args let the UI render a useful title ("IVE · After LIKE ·
    # Wonyoung") for oneshot jobs without having to parse log lines.
    script: str
    args: list[str]


def _to_out(j: jobs.Job) -> JobOut:
    return JobOut(
        id=j.id, kind=j.kind, status=j.status, return_code=j.return_code,
        started_at=j.started_at, finished_at=j.finished_at,
        lines_count=len(j.lines),
        script=j.script, args=list(j.args),
    )


class FetchIn(BaseModel):
    group: str
    limit: int | None = 40
    artist_id: str | None = None


@router.post("/fetch", response_model=JobOut)
def start_fetch(body: FetchIn):
    args = [body.group]
    if body.artist_id:
        args += ["--artist-id", body.artist_id]
    if body.limit:
        args += ["--limit", str(body.limit)]
    job = jobs.start_job("fetch", "fetch_discography.py", args)
    return _to_out(job)


class DownloadIn(BaseModel):
    group: str | None = None
    delay: float = 3.0
    headed: bool = False


@router.post("/download", response_model=JobOut)
def start_download(body: DownloadIn):
    args: list[str] = ["--delay", str(body.delay)]
    if body.group:
        args += ["--group", body.group]
    if body.headed:
        args += ["--headed"]
    job = jobs.start_job("download", "download_tiktok.py", args)
    return _to_out(job)


class ResolveIn(BaseModel):
    group: str
    urls: list[str]
    threshold: float = 0.85
    dry_run: bool = False


@router.post("/resolve", response_model=JobOut)
def start_resolve(body: ResolveIn):
    if not body.urls:
        raise HTTPException(400, "urls is empty")
    args = ["--group", body.group, "--threshold", str(body.threshold)]
    if body.dry_run:
        args += ["--dry-run"]
    args += body.urls
    job = jobs.start_job("resolve", "resolve_video_urls.py", args)
    return _to_out(job)


@router.get("", response_model=list[JobOut])
def list_recent():
    return [_to_out(j) for j in jobs.list_jobs()]


@router.get("/{job_id}", response_model=JobOut)
def get(job_id: str):
    j = jobs.get_job(job_id)
    if not j:
        raise HTTPException(404)
    return _to_out(j)


@router.get("/{job_id}/logs")
def get_logs(job_id: str, since: int = 0):
    j = jobs.get_job(job_id)
    if not j:
        raise HTTPException(404)
    return {
        "status": j.status,
        "return_code": j.return_code,
        "next_offset": len(j.lines),
        "lines": j.lines[since:],
    }


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    j = jobs.get_job(job_id)
    if not j:
        raise HTTPException(404)
    return StreamingResponse(
        jobs.stream_job(j),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
