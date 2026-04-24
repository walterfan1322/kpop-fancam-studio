"""FastAPI entry. Run with:

    uvicorn backend.main:app --reload --port 8770

Or use ./run_backend.bat
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS
from .routers import audio, groups, jobs, match, videos

app = FastAPI(title="kpop_tiktok_audio", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(groups.router)
app.include_router(jobs.router)
app.include_router(audio.router)
app.include_router(videos.router)
app.include_router(match.router)


@app.get("/api/health")
def health():
    return {"ok": True}


DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST_DIR / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST_DIR / "index.html")
