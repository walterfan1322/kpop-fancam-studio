"""One-shot: given a group, song, member name and N, produce fancam clips.

Pipeline:
  1. Read groups.yaml to get artist_name + verify song has a downloaded mp3.
  2. yt-dlp search "{artist} {member} fancam" (+Hangul query if provided).
  3. Pick top-N passing candidates, preferring not-already-downloaded; if fewer
     than N, reuse already-downloaded ones.
  4. Download each (skips if mp4 already exists, by design of download_video.py).
  5. Run match_video.run on each with --only-title so only the chosen song
     is considered. Margin guard still applies (via absolute threshold).
  6. Print per-video MATCH blobs + a final ONESHOT summary.

Usage:
    python oneshot_fancam.py --group IVE --song "LOVE DIVE" \
        --member-lat Wonyoung --member-han 장원영 --count 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "groups.yaml"
VIDEOS_DIR = ROOT / "videos"
CLIPS_DIR = ROOT / "clips"


def log(msg: str) -> None:
    print(msg, flush=True)


def get_artist(group: str) -> str:
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    info = (data.get("groups") or {}).get(group)
    if not info:
        raise SystemExit(f"group {group!r} not in {YAML_PATH}")
    return info.get("artist_name") or group


def yt_search(queries: list[str], limit: int) -> list[dict]:
    """Return flat-extract entries across all queries, deduped by id."""
    import yt_dlp  # noqa: PLC0415

    opts = {
        "quiet": True, "skip_download": True, "noplaylist": False,
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
        "extract_flat": "in_playlist",
        "playlistend": limit,
    }
    seen: set[str] = set()
    entries: list[dict] = []
    for q in queries:
        log(f"[search] {q}")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch{limit}:{q}", download=False)
            for e in (info.get("entries") or []):
                if not e or not e.get("id") or e["id"] in seen:
                    continue
                seen.add(e["id"])
                entries.append(e)
        except Exception as e:
            log(f"[search:error] {q}: {e}")
    return entries


def _iter_video_jsons():
    if not VIDEOS_DIR.exists():
        return
    for p in VIDEOS_DIR.rglob("*.json"):
        if p.stem.endswith(".quality"):
            continue
        yield p


def downloaded_ids() -> set[str]:
    ids: set[str] = set()
    for p in _iter_video_jsons():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("id"):
                ids.add(d["id"])
        except Exception:
            continue
    return ids


def path_for_id(vid: str) -> Path | None:
    """Return the mp4 Path for a downloaded video id, or None."""
    for p in _iter_video_jsons():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("id") == vid:
            mp4 = p.with_suffix(".mp4")
            return mp4 if mp4.exists() else None
    return None


_FANCAM_KW = ("fancam", "직캠", "focus", "focuscam", "focus cam", "페이스캠")
_STAGE_KW = ("stage", "무대", "performance", "dance practice", "댄스 연습",
             "music bank", "inkigayo", "m countdown", "mcountdown",
             "show champion", "the show")


def _classify_source(title_lc: str, member_lat: str, member_han: str) -> str:
    """solo_fancam if the title looks like a single-person direct cam,
    otherwise group_stage. Used both for ranking and for labelling clips in
    the output summary so the UI can show where each clip came from."""
    mem_lat = (member_lat or "").lower().strip()
    mem_han = (member_han or "").lower().strip()
    fancam_hit = any(k in title_lc for k in _FANCAM_KW)
    member_hit = (mem_lat and mem_lat in title_lc) or (mem_han and mem_han in title_lc)
    if fancam_hit and member_hit:
        return "solo_fancam"
    if fancam_hit:
        # Direct cam but title didn't name our target — still a solo shot of
        # *someone*, which won't suit us for tracking. Treat as lowest prio.
        return "solo_other"
    return "group_stage"


def filter_candidates(entries: list[dict], min_dur: float, max_dur: float,
                      min_views: int, title_any: list[str],
                      song: str = "",
                      member_lat: str = "", member_han: str = "",
                      force_landscape: bool = False) -> list[dict]:
    """force_landscape drops solo-fancam-style titles (MPD직캠 and similar are
    portrait in 2024+), keeping only team-stage / performance titles which are
    reliably 16:9. Orientation can't be filtered from yt-dlp flat metadata
    (extract_flat doesn't return width/height), so we key off title cues."""
    title_any_lc = [t.lower() for t in title_any if t.strip()]
    song_lc = song.lower().strip()
    out: list[dict] = []
    for e in entries:
        dur = e.get("duration") or 0
        views = e.get("view_count") or 0
        title = (e.get("title") or "").lower()
        if dur and (dur < min_dur or dur > max_dur):
            continue
        if views and views < min_views:
            continue
        if title_any_lc and not any(k in title for k in title_any_lc):
            continue
        e["_source_type"] = _classify_source(title, member_lat, member_han)
        if force_landscape and e["_source_type"] in ("solo_fancam", "solo_other"):
            # Direct-cam titles are almost always portrait — skip entirely
            # when the user asked for landscape.
            continue
        out.append(e)
    # Rank: song in title > source type > view count. In force_landscape mode
    # only group_stage remains, so the priority dict only matters for the
    # default path.
    if force_landscape:
        prio = {"group_stage": 0}
    else:
        prio = {"solo_fancam": 0, "group_stage": 1, "solo_other": 2}
    def key(e):
        title = (e.get("title") or "").lower()
        song_hit = 1 if (song_lc and song_lc in title) else 0
        return (-song_hit, prio.get(e.get("_source_type", ""), 3),
                -(e.get("view_count") or 0))
    out.sort(key=key)
    return out


def run_download(urls: list[str], group: str, song: str) -> None:
    """Invoke download_video.py as a subprocess so we inherit its streaming logs."""
    cmd = [sys.executable, str(ROOT / "download_video.py"),
           "--group", group, "--song", song, *urls]
    subprocess.run(cmd, check=False)


def _kept_video_ids(group: str, song: str) -> set[str]:
    """YouTube IDs whose clip has been *kept* (user pressed the download
    button) under clips/<group>/<song>/. A kept clip has a `.mp4.keep`
    sidecar; clips without one are ephemeral and get swept on each run.

    Clip filenames are `<video_id>.mp4` (see match_video.run suffix=vid).
    """
    sys.path.insert(0, str(ROOT))
    from match_video import safe_filename  # noqa: PLC0415
    song_dir = CLIPS_DIR / safe_filename(group) / safe_filename(song)
    if not song_dir.exists():
        return set()
    return {p.stem[:-len(".mp4")] for p in song_dir.glob("*.mp4.keep")}


# Backwards-compat alias (nothing else imports it, but keep the name).
_used_video_ids = _kept_video_ids


CLIP_SWEEP_SEC = 30 * 60  # clips older than this without `.keep` are deleted


def _sweep_unkept_clips(group: str, song: str) -> int:
    """Delete un-kept clips in clips/<group>/<song>/ older than CLIP_SWEEP_SEC.
    Returns how many files were removed. Runs at oneshot start so the user
    can re-generate a video they previously skipped."""
    sys.path.insert(0, str(ROOT))
    from match_video import safe_filename  # noqa: PLC0415
    song_dir = CLIPS_DIR / safe_filename(group) / safe_filename(song)
    if not song_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for mp4 in song_dir.glob("*.mp4"):
        if mp4.with_suffix(mp4.suffix + ".keep").exists():
            continue
        try:
            age = now - mp4.stat().st_mtime
        except OSError:
            continue
        if age < CLIP_SWEEP_SEC:
            continue
        try:
            mp4.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def run_match(group: str, video_path: Path, only_title: str,
              threshold: float, margin: float, suffix: str,
              delogo_corners: list[str] | None = None,
              track_crop: bool = False,
              member_lat: str = "", member_han: str = "") -> dict:
    """Import match_video.run directly so we can capture the dict result."""
    sys.path.insert(0, str(ROOT))
    import match_video  # noqa: PLC0415
    return match_video.run(group, video_path, threshold, margin,
                           extract=True, only_title=only_title, suffix=suffix,
                           delogo_corners=delogo_corners,
                           track_crop=track_crop,
                           member_lat=member_lat, member_han=member_han)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--song", required=True)
    ap.add_argument("--member-lat", required=True)
    ap.add_argument("--member-han", default="")
    ap.add_argument("--count", type=int, default=3,
                    help="target number of SUCCESSFUL clips (not videos tried)")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="hard cap on videos to download & match. "
                         "0 = auto (count*5, at least 8)")
    ap.add_argument("--min-dur", type=float, default=90.0)
    ap.add_argument("--max-dur", type=float, default=330.0)
    ap.add_argument("--min-views", type=int, default=5000)
    ap.add_argument("--search-limit", type=int, default=20,
                    help="yt-dlp results per query (then filtered down to count)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--delogo-corners", default=None,
                    help="comma-separated corners (tl,tr,bl,br) to apply ffmpeg delogo")
    ap.add_argument("--force-landscape", action="store_true",
                    help="prefer 16:9 team-stage videos; drop portrait direct-cam "
                         "titles (fancam / 직캠 / focus cam)")
    ap.add_argument("--merge-sources", type=int, default=1,
                    help="when >=2, combine up to N matched source videos "
                         "into a single merged clip, hopping between angles "
                         "second-by-second to maximise target-on-screen time "
                         "(default 1 = existing single-clip-per-video behaviour)")
    ap.add_argument("--no-shot-gate", action="store_true",
                    help="disable TransNetV2 multi-cam gate. Only matters "
                         "in merge mode (merge-sources>=2); without the "
                         "gate, multi-cam broadcast edits can enter the "
                         "pool and ruin hard-cut outfit-swap transitions "
                         "by introducing within-source cuts that look "
                         "like merge cuts.")
    ap.add_argument("--shot-gate-threshold", type=float, default=0.10,
                    help="cuts-per-second threshold for the shot gate. "
                         "Sources with this rate or higher are labeled "
                         "multi-cam and rejected from the merge pool. "
                         "Default 0.10 separates broadcast edits "
                         "(~0.15-0.50) from jikcams / dance practice / "
                         "one-take broadcast clips (~0.00-0.05).")
    ap.add_argument("--merge-style", choices=["xfade", "hard_cut"],
                    default="xfade",
                    help="how to join merged chunks. 'xfade' (default) does "
                         "a 0.5s cross-dissolve between sources, safer when "
                         "framing/angle across sources isn't tightly aligned. "
                         "'hard_cut' does a zero-frame transition for the "
                         "'outfit-swap fancam' look — the dancer appears to "
                         "instantly change outfits at each cut. Only worth "
                         "enabling when sources are same-angle same-framing "
                         "jikcams (music-show solo cams from different weeks).")
    ap.add_argument("--pose", action="store_true",
                    help="run RTMPose-m on the target bbox per frame and "
                         "derive a session-level head-y anchor so the "
                         "dancer's head lands at the same vertical fraction "
                         "across every source. Adds ~30s of CPU inference "
                         "per source (cached per (video,start,dur)), worth "
                         "it to kill head-pop on hard-cut merges. "
                         "Only used when --merge-sources >= 2.")
    args = ap.parse_args()
    corners = None
    if args.delogo_corners:
        corners = [c.strip() for c in args.delogo_corners.split(",")
                   if c.strip() in ("tl", "tr", "bl", "br", "auto")]

    t0 = time.time()
    artist = get_artist(args.group)
    swept = _sweep_unkept_clips(args.group, args.song)
    if swept:
        log(f"[sweep] removed {swept} un-kept clip(s) older than "
            f"{CLIP_SWEEP_SEC//60}min for {args.group}/{args.song}")
    used_vids = _kept_video_ids(args.group, args.song)
    if used_vids:
        log(f"[oneshot] {len(used_vids)} kept clip(s) on disk for "
            f"{args.group}/{args.song}; those videos will be excluded from the picks")
    log(f"[oneshot] group={args.group} artist={artist} song={args.song!r}")
    log(f"[oneshot] member_lat={args.member_lat!r} member_han={args.member_han!r}"
        f" count={args.count}")

    # Two-track search: solo fancams (narrow) + group stages (broad).
    # For B-side songs solo fancams often don't exist, so group stages fill
    # the gap — they're landscape 16:9 so track-crop auto-engages and
    # person_track uses the face library to follow `member_lat` into 9:16.
    if args.force_landscape:
        # Skew heavily toward music-show / performance terms, and drop any
        # queries that would surface MPD직캠-style portrait cams.
        queries = [
            f"{artist} {args.song} stage",
            f"{artist} {args.song} performance",
            f"{artist} {args.song} music bank",
            f"{artist} {args.song} inkigayo",
            f"{artist} {args.song} m countdown",
            f"{artist} {args.song} show champion",
            f"{artist} {args.song} dance practice",
            f"{artist} {args.song} live",
        ]
        if args.member_han.strip():
            queries.insert(1, f"{artist} {args.song} 무대")
    else:
        queries = [
            # Solo-fancam angle — song first so YouTube pre-filters for us.
            f"{artist} {args.member_lat} {args.song} fancam",
            f"{artist} {args.song} {args.member_lat} fancam",
            # Group-stage / performance angle (no member name → catches B-sides).
            f"{artist} {args.song} stage",
            f"{artist} {args.song} performance",
            f"{artist} {args.song} dance practice",
            # Looser solo-fancam fallback (no song — used only if above come up thin).
            f"{artist} {args.member_lat} fancam",
            f"{artist} {args.member_lat} focus cam",
        ]
        if args.member_han.strip():
            han = args.member_han.strip()
            queries.insert(1, f"{artist} {han} {args.song} 직캠")
            queries.insert(5, f"{artist} {args.song} 무대")
            queries.append(f"{artist} {han} 직캠")

    entries = yt_search(queries, args.search_limit)
    log(f"[search] {len(entries)} unique entries")

    # Accept either a fancam-style title OR a team-stage title. Audio
    # margin-guard downstream still enforces the chosen song. In
    # force_landscape mode we tighten the keyword gate to stage-only terms
    # so the title_any pre-filter doesn't let portrait titles through.
    if args.force_landscape:
        title_any = [
            "stage", "무대", "performance", "dance practice", "댄스 연습",
            "music bank", "inkigayo", "countdown", "show champion", "live",
        ]
    else:
        title_any = [
            "fancam", "직캠", "focus", args.member_lat, args.member_han,
            "stage", "무대", "performance", "dance practice", "댄스 연습",
        ]
    filt = filter_candidates(entries, args.min_dur, args.max_dur,
                             args.min_views, title_any, song=args.song,
                             member_lat=args.member_lat,
                             member_han=args.member_han,
                             force_landscape=args.force_landscape)
    log(f"[filter] {len(filt)} pass metadata filter")
    if used_vids:
        before = len(filt)
        filt = [e for e in filt if e["id"] not in used_vids]
        log(f"[filter] excluded {before - len(filt)} previously-produced video(s)")

    # Respect filter_candidates' ranking (song-in-title > source-type > views).
    # Already-downloaded entries get a tiebreak bump *within* the same tier so
    # we save bandwidth when possible, but never at the cost of picking a
    # wrong-song candidate over a right-song one.
    have = downloaded_ids()
    song_lc = args.song.lower().strip()
    if args.force_landscape:
        prio = {"group_stage": 0}
    else:
        prio = {"solo_fancam": 0, "group_stage": 1, "solo_other": 2}
    def _pool_key(e):
        title = (e.get("title") or "").lower()
        song_hit = 1 if (song_lc and song_lc in title) else 0
        have_hit = 1 if e.get("id") in have else 0
        return (-song_hit, prio.get(e.get("_source_type", ""), 3),
                -have_hit, -(e.get("view_count") or 0))
    pool = sorted(filt, key=_pool_key)
    max_attempts = args.max_attempts or max(args.count * 8, 16)
    pool = pool[:max_attempts]
    log(f"[pick] pool={len(pool)} (cap={max_attempts}); target={args.count} success(es)")

    merge_n = max(1, int(args.merge_sources))
    if merge_n >= 2:
        summary = _run_merge_mode(args, artist, pool, have, corners, merge_n, t0)
        print("ONESHOT " + json.dumps(summary, ensure_ascii=False))
        log(f"[done] merge={summary.get('merged_clip') is not None} "
            f"sources_used={summary['matched']} in {summary['elapsed_sec']}s")
        return

    # Match one at a time against the chosen song; keep going until we have
    # `count` clips or the pool is exhausted.
    clips: list[dict] = []
    matched_ok = 0
    for i, e in enumerate(pool, 1):
        if matched_ok >= args.count:
            log(f"[stop] target {args.count} reached after {i - 1} attempts")
            break
        vid = e["id"]
        if vid not in have:
            log(f"[download] ({i}/{len(pool)}) {vid} — {(e.get('title') or '')[:60]}")
            run_download([f"https://www.youtube.com/watch?v={vid}"],
                         args.group, args.song)
        mp4 = path_for_id(vid)
        if not mp4:
            log(f"[match] ({i}/{len(pool)}) {vid}: no mp4 after download, skip")
            continue
        stem = mp4.stem
        log(f"[match] ({i}/{len(pool)}) {stem}  ({matched_ok}/{args.count} matched so far)")
        try:
            r = run_match(args.group, mp4, args.song, args.threshold, args.margin,
                          suffix=vid, delogo_corners=corners,
                          track_crop=True,
                          member_lat=args.member_lat, member_han=args.member_han)
        except Exception as ex:
            log(f"[match:error] {stem}: {ex}")
            continue
        matched = "clip_path" in r
        if matched:
            matched_ok += 1
        clips.append({
            "video_id": vid,
            "video_stem": stem,
            "video_title": e.get("title"),
            "view_count": e.get("view_count"),
            "source_type": e.get("_source_type"),
            "matched": matched,
            "skip_reason": r.get("skip_reason"),
            "score": r.get("matched_score") or (r["candidates"][0]["score"] if r.get("candidates") else None),
            "clip_path": r.get("clip_path"),
            "crop_mode": r.get("crop_mode"),
        })

    ok = [c for c in clips if c["matched"]]
    summary = {
        "group": args.group,
        "song": args.song,
        "member_lat": args.member_lat,
        "member_han": args.member_han,
        "requested": args.count,
        "considered": len(clips),
        "matched": len(ok),
        "clips": clips,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("ONESHOT " + json.dumps(summary, ensure_ascii=False))
    log(f"[done] {len(ok)}/{args.count} target clips produced after "
        f"{len(clips)} attempt(s) in {summary['elapsed_sec']}s")


def _run_merge_mode(args, artist: str, pool: list[dict], have: set[str],
                     corners: list[str] | None, merge_n: int,
                     t0: float) -> dict:
    """Multi-source merge path: accept up to `merge_n` source videos, run
    audio-align + tracking on each, then fuse them into one 9:16 clip that
    hops between angles second-by-second to maximise target-on-screen time.

    Produces a single output at `clips/<Group>/<Song>/merged_<hash>.mp4`.
    The source per-clip extraction is skipped entirely; only the merged
    clip is written.
    """
    sys.path.insert(0, str(ROOT))
    import match_video  # noqa: PLC0415
    import merge_sources  # noqa: PLC0415
    import shot_gate  # noqa: PLC0415
    from sources import load_source_meta  # noqa: PLC0415
    import hashlib  # noqa: PLC0415

    log(f"[oneshot:merge] target sources={merge_n} (max pool={len(pool)})")

    # Audio-align + track up to `merge_n` candidates. Unlike single-clip
    # mode we need the full TrackedSegment per candidate, so we can't just
    # treat 'matched' as a binary count.
    merge_sources_list: list[merge_sources.MergeSource] = []
    clips: list[dict] = []
    canonical_mp3: Path | None = None
    for i, e in enumerate(pool, 1):
        if len(merge_sources_list) >= merge_n:
            log(f"[oneshot:merge] reached target {merge_n} sources after "
                f"{i - 1} attempts")
            break
        vid = e["id"]
        if vid not in have:
            log(f"[download] ({i}/{len(pool)}) {vid} — "
                f"{(e.get('title') or '')[:60]}")
            run_download([f"https://www.youtube.com/watch?v={vid}"],
                         args.group, args.song)
        mp4 = path_for_id(vid)
        if not mp4:
            log(f"[oneshot:merge] ({i}/{len(pool)}) {vid}: no mp4 after download, skip")
            continue
        stem = mp4.stem
        log(f"[oneshot:merge] ({i}/{len(pool)}) {stem} "
            f"({len(merge_sources_list)}/{merge_n} usable sources so far)")
        # Shot gate (M2a): reject multi-cam broadcast edits before spending
        # ~30-60s on person tracking. A 43-cut Performance Video merged
        # hard-cut against a 1-cut jikcam looks like "editor flipped cams",
        # not "dancer changed outfits", so we refuse to include it in the
        # merge pool. Skip the gate entirely when NOT merging (legacy
        # one-clip-per-source behaviour is fine with broadcast edits —
        # their own cuts just become part of the clip).
        if merge_n >= 2 and not args.no_shot_gate:
            try:
                shot_info = shot_gate.probe_shots(
                    mp4,
                    cuts_per_sec_threshold=args.shot_gate_threshold,
                    log_fn=log,
                )
                if shot_info.is_multicam:
                    log(f"[oneshot:merge] ({i}/{len(pool)}) {stem}: "
                        f"rejected by shot gate "
                        f"({shot_info.num_shots} shots, "
                        f"{shot_info.cuts_per_sec:.3f} cuts/sec "
                        f">= {args.shot_gate_threshold:.3f}) — multi-cam "
                        f"broadcast edit, unsuitable for outfit-swap merge")
                    clips.append({
                        "video_id": vid,
                        "video_stem": stem,
                        "video_title": e.get("title"),
                        "view_count": e.get("view_count"),
                        "source_type": e.get("_source_type"),
                        "matched": False,
                        "skip_reason": (f"multicam_{shot_info.num_shots}shots_"
                                        f"{shot_info.cuts_per_sec:.2f}cps"),
                        "score": None,
                        "crop_mode": None,
                    })
                    continue
            except Exception as ex:
                # Gate failure is non-fatal — log and continue. Better to
                # try tracking than to silently drop a maybe-good source.
                log(f"[oneshot:merge] shot-gate error on {stem}: {ex!r} — "
                    f"continuing without gate")
        try:
            out, ts = match_video.match_and_track(
                args.group, mp4, args.threshold, args.margin,
                only_title=args.song,
                member_lat=args.member_lat, member_han=args.member_han,
                # merge planner already picks per-bucket source on quality,
                # so each source should keep its face-tracked trajectory
                # instead of collapsing to static-centre on low coverage.
                disable_low_cov_fallback=True,
            )
        except Exception as ex:
            log(f"[oneshot:merge:error] {stem}: {ex}")
            continue
        usable = (ts is not None) and (out.get("skip_reason") is None)
        clips.append({
            "video_id": vid,
            "video_stem": stem,
            "video_title": e.get("title"),
            "view_count": e.get("view_count"),
            "source_type": e.get("_source_type"),
            "matched": usable,
            "skip_reason": out.get("skip_reason"),
            "score": out.get("matched_score"),
            "crop_mode": out.get("crop_mode"),
        })
        if not usable:
            continue
        # Canonical mp3: construct the path directly from args.song.
        # DON'T use load_group_tracks(only_title=…) — its `only_title` arg
        # is silently ignored (see its docstring; it always returns every
        # track so the matcher can cross-compare). Taking `[0]` from that
        # result would pick the first track in groups.yaml order (for IVE
        # that's BLACKHOLE), producing clips with completely wrong audio.
        # We already know the exact song the user asked for — look it up
        # by deterministic filename instead.
        if canonical_mp3 is None:
            try:
                from match_video import OUTPUT_DIR, safe_filename  # noqa: PLC0415
                cand = (OUTPUT_DIR / safe_filename(args.group)
                        / f"{safe_filename(args.song)}.mp3")
                if cand.exists():
                    canonical_mp3 = cand
                else:
                    log(f"[oneshot:merge] canonical mp3 not found at "
                        f"{cand} — falling back to source audio")
            except Exception as ex:
                log(f"[oneshot:merge] couldn't resolve canonical mp3: {ex}")
        # Build SourceMeta (broadcaster/date from sidecar .json).
        sidecar = mp4.with_suffix(".json")
        smeta = load_source_meta(sidecar) if sidecar.exists() else None
        if smeta is None:
            # .json is missing or corrupt — synthesise a self-cluster entry
            # so the video still participates.
            from sources import SourceMeta  # noqa: PLC0415
            smeta = SourceMeta(
                video_id=vid, path=mp4, title=e.get("title") or stem,
                duration=float(e.get("duration") or 0.0) or None,
                broadcaster=None, date=None,
            )
        merge_sources_list.append(merge_sources.MergeSource(
            meta=smeta,
            offset_sec=float(out["matched_offset_sec"]),
            tracked=ts,
            matched_title=out["matched_title"],
        ))
        # Expose target-member visibility per source so it's obvious when a
        # merge pool has no source that actually features the target — e.g.
        # a pool of group-stage / one-take / rotating-closeup cams will all
        # pass tracking but none may feature Wonyoung for long. Without this
        # the user sees a silently-mediocre merge and doesn't know why.
        try:
            q = ts.quality_mask(step_sec=1.0)
            cov = ts.coverage_mask(step_sec=1.0)
            scores = {k: round(float(v), 3)
                      for k, v in (ts.target_scores or {}).items()}
            dom = {k: round(float(v), 3)
                   for k, v in (ts.target_dominance or {}).items()}
            log(f"[oneshot:merge]   target-vis: "
                f"cov={float(cov.mean()):.2%} "
                f"q_mean={float(q.mean()):.3f} q_max={float(q.max()):.3f} "
                f"scores={scores} dom={dom} "
                f"cluster={smeta.cluster_label}")
        except Exception as ex:
            log(f"[oneshot:merge]   target-vis: unavailable ({ex})")

    if not merge_sources_list:
        log("[oneshot:merge] no usable sources — aborting")
        return {
            "group": args.group, "song": args.song,
            "member_lat": args.member_lat, "member_han": args.member_han,
            "requested": merge_n, "considered": len(clips),
            "matched": 0, "clips": clips,
            "merged_clip": None, "mode": "merge",
            "elapsed_sec": round(time.time() - t0, 1),
        }

    # All sources must share the same clip_dur; pick the min so no source
    # runs past its aligned segment.
    clip_dur = min(float(s.tracked.dur_sec) for s in merge_sources_list)
    log(f"[oneshot:merge] chosen clip_dur={clip_dur:.2f}s across "
        f"{len(merge_sources_list)} source(s)")

    # Beat detection on the canonical audio — chunk boundaries will
    # snap to the nearest beat (1-2 frames before) so cuts land
    # musically. Falls back to no-snap if librosa is missing or the
    # track has no clear pulse.
    beat_src = (canonical_mp3 if canonical_mp3 and canonical_mp3.exists()
                else merge_sources_list[0].meta.path)
    beats = merge_sources.detect_beats(beat_src, log_fn=log)
    if len(beats) > 0:
        import numpy as np  # noqa: PLC0415
        beats = beats[(beats >= 0.0) & (beats <= clip_dur)]

    # M4a/b: precompute per-source head keypoint tracks when --pose is
    # set. The sidecar cache makes the second inference pass inside
    # merge_clip free. We need yaw buckets BEFORE plan_merge so the
    # bucket-match bonus can influence source selection.
    yaw_buckets = None
    if bool(args.pose):
        try:
            import pose_track  # noqa: PLC0415
        except Exception as e:
            log(f"[oneshot:merge] pose precompute: import failed ({e!r})")
        else:
            pre_head_tracks = []
            for s in merge_sources_list:
                try:
                    ht = pose_track.track_head_keypoints(
                        s.meta.path,
                        tracks=s.tracked.tracks,
                        target_ids=s.tracked.target_ids,
                        start_sec=s.tracked.meta.start_sec,
                        dur_sec=s.tracked.meta.dur_sec,
                        meta=s.tracked.meta,
                        log_fn=log)
                except Exception as e:
                    log(f"[oneshot:merge] pose precompute failed for "
                        f"{s.meta.video_id}: {e!r}")
                    ht = None
                s.head = ht
                pre_head_tracks.append(ht)
            yaw_buckets = pose_track.session_yaw_bucket(pre_head_tracks)

    # Plan + render.
    chunks = merge_sources.plan_merge(merge_sources_list, clip_dur,
                                       step_sec=1.0, min_chunk_sec=1.5,
                                       beats=beats, beat_lead_sec=0.05,
                                       yaw_buckets=yaw_buckets,
                                       log_fn=log)
    # Output path: hash of sorted video ids for stable naming.
    sorted_ids = sorted(s.meta.video_id for s in merge_sources_list)
    vid_hash = hashlib.sha1("|".join(sorted_ids).encode("utf-8")).hexdigest()[:10]
    from match_video import safe_filename  # noqa: PLC0415
    dst = (CLIPS_DIR / safe_filename(args.group) / safe_filename(args.song)
           / f"merged_{vid_hash}.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if canonical_mp3 is None or not canonical_mp3.exists():
        log(f"[oneshot:merge] canonical mp3 not resolved — "
            f"falling back to first source's audio via its own extraction")

    # xfade=0.5s is the default safe dissolve — hides framing jumps between
    # sources that weren't shot from identical angles. hard_cut=0.0 is the
    # "outfit-swap" look: dancer appears to instantly change clothes/venue.
    # Only picks the hard cut when the user opted in — M3's canonical
    # framing is what actually makes hard cuts tolerable, so the default
    # stays on xfade.
    merge_xfade_dur = 0.0 if args.merge_style == "hard_cut" else 0.5
    log(f"[oneshot:merge] merge_style={args.merge_style} "
        f"xfade_dur={merge_xfade_dur:.2f}s")

    merge_sources.merge_clip(
        chunks, merge_sources_list,
        canonical_audio_mp3=(canonical_mp3
                             if canonical_mp3 and canonical_mp3.exists()
                             else merge_sources_list[0].meta.path),
        clip_dur=clip_dur,
        dst=dst,
        xfade_dur=merge_xfade_dur,
        use_pose=bool(args.pose),
        log_fn=log,
    )
    try:
        rel = str(dst.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel = str(dst).replace("\\", "/")
    log(f"[ok] merged clip -> {rel}")

    # Compose summary. Include per-source stats so UI/user can see which
    # angles contributed.
    per_src_summary = [
        {
            "video_id": s.meta.video_id,
            "cluster": s.meta.cluster_label,
            "offset_sec": s.offset_sec,
            "coverage": float(s.tracked.coverage_mask(step_sec=1.0).mean()),
            "mode": s.tracked.mode,
            "target_ids": list(s.tracked.target_ids),
        }
        for s in merge_sources_list
    ]
    per_chunk_summary = [
        {"song_start": c.song_start, "song_end": c.song_end,
         "src_video_id": merge_sources_list[c.src_idx].meta.video_id,
         "src_cluster": merge_sources_list[c.src_idx].meta.cluster_label}
        for c in chunks
    ]
    return {
        "group": args.group, "song": args.song,
        "member_lat": args.member_lat, "member_han": args.member_han,
        "requested": merge_n, "considered": len(clips),
        "matched": len(merge_sources_list), "clips": clips,
        "merged_clip": rel, "mode": "merge",
        "merge_style": args.merge_style,
        "use_pose": bool(args.pose),
        "sources": per_src_summary,
        "chunks": per_chunk_summary,
        "clip_dur": clip_dur,
        "elapsed_sec": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    main()
