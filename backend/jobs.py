"""Subprocess job runner with SSE-friendly log streaming.

Each job runs one of the existing CLI scripts (fetch_discography.py,
download_tiktok.py, resolve_video_urls.py) as a child process and captures
stdout/stderr line by line. The REST layer exposes:
  - POST to start a job   -> returns job_id
  - GET /jobs/{id}/stream -> Server-Sent Events with each log line
  - GET /jobs/{id}        -> current status snapshot
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import PROJECT_ROOT, PYTHON_EXE


@dataclass
class Job:
    id: str
    kind: str           # "fetch" | "download" | "resolve"
    args: list[str]     # script args (excluding python + script path)
    script: str         # e.g. "fetch_discography.py"
    status: str = "pending"  # pending | running | done | failed
    return_code: int | None = None
    lines: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    # internal
    _cond: threading.Condition = field(default_factory=threading.Condition)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _run_subprocess(job: Job) -> None:
    job.status = "running"
    job.started_at = time.time()

    cmd = [PYTHON_EXE, job.script, *job.args]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        with job._cond:
            job.lines.append(f"[launch failed] {e}")
            job.status = "failed"
            job.return_code = -1
            job.finished_at = time.time()
            job._cond.notify_all()
        return

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        with job._cond:
            job.lines.append(line)
            job._cond.notify_all()

    proc.wait()
    with job._cond:
        job.return_code = proc.returncode
        job.status = "done" if proc.returncode == 0 else "failed"
        job.finished_at = time.time()
        job._cond.notify_all()


def start_job(kind: str, script: str, args: list[str]) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, script=script, args=args)
    with _jobs_lock:
        _jobs[job.id] = job
    t = threading.Thread(target=_run_subprocess, args=(job,), daemon=True)
    t.start()
    return job


def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs(limit: int = 50) -> list[Job]:
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: -j.started_at)
        return items[:limit]


async def stream_job(job: Job) -> AsyncIterator[str]:
    """Yield SSE-formatted lines as new stdout arrives, then a final 'done'."""
    last = 0
    while True:
        # Copy out new lines under the lock, then release before yielding
        with job._cond:
            current = len(job.lines)
            new = job.lines[last:current]
            terminal = job.status in ("done", "failed")
            rc = job.return_code
        for line in new:
            yield _sse("log", line)
        last = current
        if terminal:
            yield _sse("status", f'{{"status":"{job.status}","return_code":{rc}}}')
            return
        await asyncio.sleep(0.25)


def _sse(event: str, data: str) -> str:
    # Escape newlines inside data per SSE spec
    data = data.replace("\r\n", "\n").replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {data}\n\n"
