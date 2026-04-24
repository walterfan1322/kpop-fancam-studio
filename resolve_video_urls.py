"""Resolve TikTok video URLs -> (music_title, music_url) and optionally
merge into groups.yaml by fuzzy-matching the title against an existing group.

Usage:
    # Pipe URLs via stdin (one per line)
    python resolve_video_urls.py --group IVE < urls.txt

    # Or pass inline
    python resolve_video_urls.py --group IVE \\
        https://vt.tiktok.com/xxx/ https://vt.tiktok.com/yyy/

    # Just print, don't modify yaml
    python resolve_video_urls.py --dry-run ...
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import yaml
from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright

from download_tiktok import (
    BROWSER_DATA,
    MUSIC_HREF_RE,
    UA,
    YAML_PATH,
)

ROOT = Path(__file__).parent


def launch_ctx(pw, headless: bool):
    BROWSER_DATA.mkdir(exist_ok=True)
    return pw.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_DATA),
        headless=headless,
        user_agent=UA,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        args=["--disable-blink-features=AutomationControlled"],
    )


def resolve_one(ctx, url: str) -> tuple[str, str] | None:
    """Open a TikTok share URL (video OR music page), return
    (music_title, canonical_music_url)."""
    page = ctx.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)  # let redirects settle
        final_url = page.url
        title_text = ""
        # Case A: the short link redirected to a music page already
        if "/music/" in final_url and MUSIC_HREF_RE.search(final_url):
            music_url = final_url.split("?")[0]
            # Page title looks like "IVE.official - After LIKE | TikTok"
            try:
                ptitle = page.title()
                m = re.match(r"^[^-|]+-\s*(.+?)\s*\|\s*TikTok", ptitle)
                if m:
                    title_text = m.group(1).strip()
            except Exception:
                pass
            if not title_text:
                slug_match = re.search(r"/music/([^/?#]+)-\d{10,}", music_url)
                if slug_match:
                    title_text = slug_match.group(1).replace("-", " ")
            return title_text, music_url

        # Case B: video page -> look for music link in DOM
        try:
            page.wait_for_selector("a[href*='/music/']", timeout=10000)
        except PWTimeout:
            return None
        pairs = page.eval_on_selector_all(
            "a[href*='/music/']",
            "els => els.map(e => [e.getAttribute('href'), e.textContent])",
        )
    finally:
        page.close()

    for href, text in pairs:
        if not href:
            continue
        m = MUSIC_HREF_RE.search(href)
        if not m:
            continue
        full = href if href.startswith("http") else "https://www.tiktok.com" + href
        music_url = full.split("?")[0]
        title = (text or "").strip()
        if not title or title.lower().startswith("original sound"):
            slug_match = re.search(r"/music/([^/?#]+)-\d{10,}", music_url)
            if slug_match:
                title = slug_match.group(1).replace("-", " ")
        return title, music_url
    return None


def best_match(title: str, candidates: list[str]) -> tuple[str, float]:
    title_norm = title.lower().replace("-", " ").replace("_", " ")
    best = ("", 0.0)
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, title_norm, c.lower()).ratio()
        if ratio > best[1]:
            best = (c, ratio)
    return best


def load_yaml() -> dict:
    with YAML_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict) -> None:
    with YAML_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


def existing_title_of(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("title") or entry.get("name") or ""
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*", help="TikTok video URLs")
    ap.add_argument("--group", required=True, help="Group name to attach matches to")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify yaml")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Min fuzzy match ratio to auto-assign (default 0.45)")
    args = ap.parse_args()

    urls = list(args.urls)
    if not urls and not sys.stdin.isatty():
        urls = [line.strip() for line in sys.stdin if line.strip()]
    if not urls:
        sys.exit("No URLs provided (pass as args or pipe via stdin)")

    data = load_yaml()
    group_info = (data.get("groups") or {}).get(args.group)
    if not group_info:
        sys.exit(f"Group {args.group!r} not found in groups.yaml")
    tracks = group_info.get("tracks") or []
    titles = [existing_title_of(t) for t in tracks if existing_title_of(t)]

    results = []
    with sync_playwright() as pw:
        ctx = launch_ctx(pw, headless=not args.headed)
        try:
            for i, url in enumerate(urls, 1):
                print(f"[{i}/{len(urls)}] {url}")
                got = resolve_one(ctx, url)
                if not got:
                    print("    ! could not resolve")
                    results.append((url, None, None, None, 0.0))
                    continue
                music_title, music_url = got
                match, ratio = best_match(music_title, titles)
                print(f"    music_title = {music_title!r}")
                print(f"    music_url   = {music_url}")
                print(f"    best match  = {match!r} (ratio={ratio:.2f})")
                results.append((url, music_title, music_url, match, ratio))
        finally:
            ctx.close()

    print("\n=== Summary ===")
    assigned = []
    unmatched = []
    for url, mtitle, murl, match, ratio in results:
        if not murl:
            unmatched.append((url, "resolve failed", None))
            continue
        if ratio >= args.threshold and match:
            assigned.append((match, murl, ratio, mtitle))
        else:
            unmatched.append((url, mtitle, murl))

    for t, u, r, mt in assigned:
        print(f"  OK  [{r:.2f}] {t}  <-  {mt}  ->  {u}")
    for url, mt, murl in unmatched:
        print(f"  ??  {url}  -> {mt}  ({murl})")

    if args.dry_run:
        print("\n(dry-run: yaml not modified)")
        return

    # Apply assignments
    if not assigned:
        print("\nNothing matched above threshold. yaml not modified.")
        return

    new_tracks = []
    assigned_map = {t: u for t, u, r, mt in assigned}
    applied = 0
    for entry in tracks:
        title = existing_title_of(entry)
        if title in assigned_map:
            new_tracks.append({"title": title, "music_url": assigned_map[title]})
            applied += 1
        else:
            new_tracks.append(entry)
    data["groups"][args.group]["tracks"] = new_tracks
    save_yaml(data)
    print(f"\nUpdated {applied} entries in groups.yaml")


if __name__ == "__main__":
    main()
