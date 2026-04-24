"""Runtime configuration for the kpop_tiktok_audio backend.

All paths and ports can be overridden via env vars so the same code runs
on the dev box and on the deployment target.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root (where groups.yaml, output/, and the scripts live)
PROJECT_ROOT = Path(os.environ.get(
    "KPOP_PROJECT_ROOT",
    Path(__file__).resolve().parent.parent,
))

YAML_PATH = PROJECT_ROOT / "groups.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
INDEX_PATH = OUTPUT_DIR / "index.json"
FAILURES_PATH = OUTPUT_DIR / "failures.tsv"
VIDEOS_DIR = PROJECT_ROOT / "videos"
CLIPS_DIR = PROJECT_ROOT / "clips"

# Python interpreter to use for subprocess jobs. Windows venvs put it under
# Scripts/python.exe; POSIX venvs under bin/python.
_VENV = PROJECT_ROOT / "venv"
_DEFAULT_PY = (_VENV / "Scripts" / "python.exe") if sys.platform == "win32" \
    else (_VENV / "bin" / "python")
PYTHON_EXE = os.environ.get("KPOP_PYTHON", str(_DEFAULT_PY))

# Server
HOST = os.environ.get("KPOP_HOST", "127.0.0.1")
PORT = int(os.environ.get("KPOP_PORT", "8770"))

# CORS origins allowed for the frontend (comma-sep)
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "KPOP_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]
