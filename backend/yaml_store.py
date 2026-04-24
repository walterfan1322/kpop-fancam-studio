"""Read/write groups.yaml with a tiny file lock so concurrent writes don't corrupt."""
from __future__ import annotations

import threading
from typing import Any

import yaml

from .config import YAML_PATH

_lock = threading.RLock()


def load() -> dict[str, Any]:
    with _lock:
        if not YAML_PATH.exists():
            return {"groups": {}}
        data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
        data.setdefault("groups", {})
        return data


def save(data: dict[str, Any]) -> None:
    with _lock:
        YAML_PATH.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=2),
            encoding="utf-8",
        )


def track_title(t: Any) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        return t.get("title") or t.get("name") or ""
    return ""


def track_url(t: Any) -> str:
    if isinstance(t, dict):
        return t.get("music_url") or ""
    return ""


def serialize_track(title: str, music_url: str | None) -> Any:
    title = title.strip()
    if music_url:
        return {"title": title, "music_url": music_url.strip()}
    return title
