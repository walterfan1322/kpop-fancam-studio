"""Download the 60-second TikTok-cut audio for every track in groups.yaml.

Flow per track:
  1. TikTok music search: artist + title -> top music card -> music_id + music URL
  2. Open music page, intercept the auto-played audio/mp4 clip from TikTok CDN
  3. ffmpeg transcode the .m4a bytes to MP3 at output/<group>/<safe_title>.mp3

This gets the authoritative 60-second cut (same clip the app plays on the music
page), not the audio of whatever random video happens to use the sound.

Skips files that already exist. Uses a persistent browser context under
./browser_data so a one-time manual login (if TikTok prompts) sticks.

Usage:
    python download_tiktok.py                # all groups in groups.yaml
    python download_tiktok.py --group IVE    # just one group
    python download_tiktok.py --headed       # watch what it's doing (debug)
    python download_tiktok.py --delay 4      # seconds between songs (default 3)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import yaml
from playwright.sync_api import (
    BrowserContext,
    Page,
    TimeoutError as PWTimeout,
    sync_playwright,
)

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "groups.yaml"
OUT_DIR = ROOT / "output"
BROWSER_DATA = ROOT / "browser_data"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# TikTok music CDN patterns (observed: sf16-ies-music-sg.tiktokcdn.com/obj/tos-...)
AUDIO_URL_HINT = re.compile(r"tiktokcdn\.com/obj/(tos|musically)")

MUSIC_HREF_RE = re.compile(r"/music/[^/?#]*?-(\d{15,25})")


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name[:120] or "untitled"


def load_groups(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("groups", {}) or {}


def launch_context(pw, headless: bool) -> BrowserContext:
    BROWSER_DATA.mkdir(exist_ok=True)
    return pw.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_DATA),
        headless=headless,
        user_agent=UA,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        args=["--disable-blink-features=AutomationControlled"],
    )


def find_music_url(page: Page, artist: str, title: str) -> tuple[str, str] | None:
    """Search TikTok music and return (music_id, music_page_url) or None."""
    q = urllib.parse.quote(f"{artist} {title}")
    url = f"https://www.tiktok.com/search/music?q={q}"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except PWTimeout:
        return None
    try:
        page.wait_for_selector("a[href*='/music/']", timeout=15000)
    except PWTimeout:
        return None

    hrefs = page.eval_on_selector_all(
        "a[href*='/music/']",
        "els => els.map(e => e.getAttribute('href'))",
    )
    for href in hrefs:
        if not href:
            continue
        m = MUSIC_HREF_RE.search(href)
        if m:
            full = href if href.startswith("http") else "https://www.tiktok.com" + href
            return m.group(1), full.split("?")[0]
    return None


def capture_music_audio(ctx: BrowserContext, music_url: str, wait_ms: int = 12000) -> bytes | None:
    """Open the music page, observe the audio URL the player fetches, then
    re-download it through the browser context (which carries cookies).
    Returns audio bytes or None."""
    page = ctx.new_page()
    audio_url_holder: dict[str, str] = {}

    def on_request(req):
        if "url" in audio_url_holder:
            return
        url = req.url
        if AUDIO_URL_HINT.search(url) or url.endswith(".m4a") or url.endswith(".mp3"):
            audio_url_holder["url"] = url

    page.on("request", on_request)
    try:
        page.goto(music_url, wait_until="domcontentloaded", timeout=30000)
    except PWTimeout:
        page.close()
        return None

    deadline = time.time() + wait_ms / 1000
    while time.time() < deadline and "url" not in audio_url_holder:
        page.wait_for_timeout(400)

    audio_url = audio_url_holder.get("url")
    if not audio_url:
        page.close()
        return None

    # Re-download with the context's cookies (can't read resp.body from a
    # streaming media response mid-playback).
    try:
        resp = ctx.request.get(audio_url, headers={"Referer": music_url})
        if resp.status != 200:
            page.close()
            return None
        data = resp.body()
    except Exception:
        page.close()
        return None
    finally:
        page.close()

    if len(data) < 50_000:
        return None
    return data


def _ffmpeg() -> str:
    import shutil
    import static_ffmpeg
    static_ffmpeg.add_paths()
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("static_ffmpeg did not provide ffmpeg")
    return exe


def transcode_to_mp3(raw: bytes, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            _ffmpeg(),
            "-y",
            "-loglevel", "error",
            "-i", str(tmp_path),
            "-vn",
            "-c:a", "libmp3lame",
            "-q:a", "2",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.stderr.write(f"    ffmpeg failed: {res.stderr.strip()[:300]}\n")
            return False
        return out_path.exists() and out_path.stat().st_size > 0
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_track(t) -> tuple[str, str | None]:
    """Accept either 'Title' or {'title': 'x', 'music_url': 'https://...'}."""
    if isinstance(t, str):
        return t, None
    if isinstance(t, dict):
        return t.get("title") or t.get("name"), t.get("music_url") or t.get("url")
    raise ValueError(f"Unsupported track entry: {t!r}")


def process_group(
    ctx: BrowserContext,
    group: str,
    info: dict,
    delay: float,
    index_path: Path,
    failures_path: Path,
) -> None:
    artist = info.get("artist_name", group)
    tracks = info.get("tracks") or []
    out_group = OUT_DIR / safe_filename(group)
    out_group.mkdir(parents=True, exist_ok=True)

    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    index.setdefault(group, {})

    search_page = ctx.new_page()
    for entry in tracks:
        title, override_url = _parse_track(entry)
        if not title:
            continue
        safe = safe_filename(title)
        mp3_path = out_group / f"{safe}.mp3"
        if mp3_path.exists():
            print(f"  [skip] {title}")
            continue

        print(f"  [get ] {title}")
        if override_url:
            m = MUSIC_HREF_RE.search(override_url)
            music_id = m.group(1) if m else "override"
            music_url = override_url.split("?")[0]
        else:
            found = find_music_url(search_page, artist, title)
            if not found:
                print("    search miss -> logged to failures.tsv")
                index[group][title] = {"status": "no_music_match"}
                _append_failure(failures_path, group, artist, title)
                _save_index(index_path, index)
                continue
            music_id, music_url = found

        raw = capture_music_audio(ctx, music_url)
        if not raw:
            print(f"    music_id={music_id} but no audio captured")
            index[group][title] = {"status": "no_audio", "music_id": music_id, "music_url": music_url}
            _append_failure(failures_path, group, artist, title)
            _save_index(index_path, index)
            continue

        ok = transcode_to_mp3(raw, mp3_path)
        index[group][title] = {
            "status": "ok" if ok else "transcode_failed",
            "music_id": music_id,
            "music_url": music_url,
            "bytes_in": len(raw),
        }
        if ok:
            print(f"    -> {mp3_path.name} ({mp3_path.stat().st_size // 1024} KB)")

        _save_index(index_path, index)
        time.sleep(delay + random.uniform(0, 2))

    search_page.close()


def _save_index(path: Path, index: dict) -> None:
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_failure(path: Path, group: str, artist: str, title: str) -> None:
    search_url = (
        "https://www.tiktok.com/search/music?q="
        + urllib.parse.quote(f"{artist} {title}")
    )
    header = "group\tartist\ttitle\ttiktok_search_url\n"
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{group}\t{artist}\t{title}\t{search_url}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", help="Only process this group name")
    ap.add_argument("--headed", action="store_true", help="Show browser window")
    ap.add_argument("--delay", type=float, default=3.0, help="Base seconds between songs")
    args = ap.parse_args()

    groups = load_groups(YAML_PATH)
    if not groups:
        sys.exit(f"No groups in {YAML_PATH}. Run fetch_discography.py first.")
    if args.group:
        if args.group not in groups:
            sys.exit(f"Group {args.group!r} not in yaml. Available: {list(groups)}")
        groups = {args.group: groups[args.group]}

    OUT_DIR.mkdir(exist_ok=True)
    index_path = OUT_DIR / "index.json"
    failures_path = OUT_DIR / "failures.tsv"

    with sync_playwright() as pw:
        ctx = launch_context(pw, headless=not args.headed)
        try:
            for group, info in groups.items():
                print(f"\n=== {group} ===")
                process_group(ctx, group, info, args.delay, index_path, failures_path)
        finally:
            ctx.close()

    if failures_path.exists():
        print(f"\n{failures_path.name} lists songs that need a music_url override")
        print("Paste the music URL into groups.yaml like:")
        print("  tracks:")
        print("    - title: BLACKHOLE")
        print("      music_url: https://www.tiktok.com/music/BLACKHOLE-7603042795515709456")


if __name__ == "__main__":
    main()
