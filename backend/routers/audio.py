"""Serve downloaded mp3 files."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import OUTPUT_DIR

router = APIRouter(prefix="/api/audio", tags=["audio"])


def _safe(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name[:120] or "untitled"


def _resolve(group: str, title: str) -> Path:
    safe_group = _safe(group)
    safe_title = _safe(title)
    path = (OUTPUT_DIR / safe_group / f"{safe_title}.mp3").resolve()
    # Defence in depth: refuse paths that escape OUTPUT_DIR
    if not str(path).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(400, "bad path")
    return path


@router.get("/{group}/{title}.mp3")
def get_mp3(group: str, title: str):
    path = _resolve(group, title)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path), media_type="audio/mpeg", filename=path.name)
