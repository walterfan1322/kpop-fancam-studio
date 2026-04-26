# kpop-fancam-studio

Turn multiple fancams of the same K-pop performance into a single **outfit-swap
style** edit — hard-cut between sources so the dancer appears to instantly
change outfit or venue, while staying framed consistently (head lands at the
same screen position across cuts).

Not a general-purpose video editor. Opinionated pipeline:
- pick a member
- pull every fancam of every song on YouTube
- reject multi-camera broadcast edits (TransNetV2 shot gate)
- audio-fingerprint each source to align to the official TikTok cut
- face-ID + BoT-SORT to follow the member when occluded by peers
- RTMPose keypoints to lock head position + bucket sources by yaw angle
- beat-aligned hard cuts, pose-refined transitions

## Architecture

```
┌──────────────────────── User (browser) ────────────────────────┐
│  React + Vite + TypeScript  →  http://host:8000/               │
│   Quick Mode | Groups | Videos | Jobs | Recent Clips           │
└──────────┬────────────────────────────────────────────▲────────┘
           │ REST / polling                             │ MP4 download
           ▼                                            │
┌──────────── FastAPI backend (backend/main.py) ────────┴────────┐
│   /api/videos/search      → yt-dlp metadata search             │
│   /api/videos/download-batch                                    │
│   /api/videos/oneshot     → spawn oneshot_fancam.py            │
│   /api/jobs/{id}, /logs   → tail subprocess output             │
│   /api/groups/*           → groups.yaml CRUD                   │
│   /api/videos/clips       → serve finished MP4s                │
└───────────────────┬────────────────────────────────────────────┘
                    │ subprocess
                    ▼
┌────────── oneshot_fancam.py (CLI / orchestrator) ──────────────┐
│  for each source in pool:                                       │
│    shot_gate.probe_shots()        ← TransNetV2 cuts/sec         │
│    chroma fingerprint → song offset                             │
│    person_track (BoT-SORT + InsightFace face-ID)                │
│    quality_probe   (stability, height ratio, person count)      │
│    pose_track.track_head_keypoints() ← RTMPose-m ONNX           │
│  merge_sources.plan_merge()                                     │
│    session anchors (h, head_y, tempo, beats)                    │
│    yaw bucket per source (frontal / left / right / unknown)     │
│    assign chunks to best source, prefer same-bucket transitions │
│  extract_clip_tracked() per chunk (pose-locked head y)          │
│  concat (hard_cut) or xfade (soft)  → merged_*.mp4              │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline stages (milestones)

Status as of last session:

| Stage | What it does                                                         | Status |
|-------|----------------------------------------------------------------------|--------|
| M0    | Install & smoke-test RTMPose-m ONNX / Beat This! / TransNetV2 / Pr-VIPE | ✅ |
| M1a   | Plumb `xfade_dur` through API; add `hard_cut` via concat demuxer     | ✅ |
| M1b   | Force identical `target_fill` across all sources in a session        | ✅ |
| M1c   | Swap `librosa.beat_track` → Beat This! for K-pop tempo/downbeat      | ✅ |
| M1d   | Validate M1 end-to-end on REBEL HEART reference set                  | ✅ |
| M2a   | Integrate TransNetV2 shot-gate to reject multi-cam broadcasts        | ✅ |
| M2b   | Re-download a pool with gate active; confirm only jikcams pass       | ✅ |
| M3a   | RTMPose-m ONNX runner + per-source head keypoint track               | ✅ |
| M3b   | Re-crop so head lands at fixed y-fraction regardless of source       | ✅ |
| M3c   | Validation: run M1+M3 together, compare to reference                 | ✅ |
| M4a   | Keypoint-based yaw estimation → bucket frontal/left/right            | ✅ |
| M4b   | `plan_merge` honors bucket — prefers same-bucket transitions         | ✅ |
| M5    | Yaw-mismatch ±3-frame cut refinement (pose-refine pass)              | ✅ |
| M6    | M6-candidate observability — flag stubborn cuts in the merge log     | ✅ |
| M6+   | RIFE frame interpolation for flagged cuts                            | ❌ rejected (visual eval) |
| M7    | Expose merge_style / use_pose flags in API + UI                      | ✅ |

Main pipeline (M0–M4 + M7) is functionally complete. M4b's cross-bucket
preference can't be exercised end-to-end on REBEL HEART (the entire
downloadable pool falls in `frontal`/`unknown`), so it's covered by a
synthetic regression in `tests/test_yaw_bucket.py` instead — five
hand-crafted scenarios verify the bucket-match bonus arithmetic, with
an A/B counter-test to prove the bonus actually changes picks.

No remaining work — M6+ was evaluated and rejected.

**M6+ (RIFE interpolator) — rejected after visual evaluation.** The
plan was to build RIFE-style frame interpolation on top of the
M6-candidate flags so stubborn cuts (low yaw-conf, or best yaw² above
the acceptance cap) wouldn't read as jumps. The README's stated
trigger criterion was "visual inspection of rendered MP4s confirms
the flagged cuts are perceptibly jarring."

Visual inspection of the REBEL HEART reference render
(`merged_e5e4b023b9.mp4`, 5 cuts, 4 of them M6-candidates) inverted
that premise:

| Cut | t (s) | What's actually happening | Verdict |
|-----|-------|---------------------------|---------|
| 0   | 7.870 | Same dance-practice studio, micro angle shift, same costume | Not jarring |
| 1   | 10.210 | Practice video → MV close-up. Total scene/costume change | **Intentional outfit-swap** |
| 2   | 13.030 | Music show #1 (cream jacket, daylight) → Music show #2 (gray jacket, purple stage) | **Intentional outfit-swap** |
| 4   | 58.010 | Same music show, near-identical pose, micro angle shift | Not jarring |

Cuts 1 and 2 are *exactly* the effect this project's tagline describes:
"the dancer appears to instantly change outfit or venue." RIFE-morphing
those cuts would erase the snap that creates the outfit-swap aesthetic.
M5's pose-refine and M6's score-cap are still correct (they prevent the
planner from snapping a cut onto a wildly-mismatched yaw frame within
the *same* source-pair window), but the resulting "M6-candidate" flag
turned out to mark cuts where the helper *correctly chose to stop*, not
broken cuts that need smoothing.

Cuts 0 and 4 are minor enough that no smoothing is warranted either.

The M6-candidate logging stays in the codebase as observability for
future songs whose pool composition might produce genuinely-bad cuts,
but the interpolator track is closed: building RIFE for this aesthetic
would be a regression, not an improvement.

## Tech stack

| Layer           | Tool                                                      |
|-----------------|-----------------------------------------------------------|
| Shot boundary   | TransNetV2 (ONNX on CPU; MPS skipped for numerical drift) |
| Person tracking | Ultralytics YOLOv8n + BoT-SORT                            |
| Face ID         | InsightFace `buffalo_l`                                   |
| Pose            | RTMPose-m body7 (192×256, COCO-17) via rtmlib             |
| Yaw proxy       | `(nose_x - eye_midpoint_x) / \|eye_span\|`                |
| Audio align     | chroma fingerprint matching → per-source offset           |
| Beats / tempo   | Beat This! (default) + librosa fallback                   |
| Backend         | FastAPI + uvicorn                                         |
| Frontend        | React 18 + Vite + TypeScript                              |
| Video ops       | ffmpeg (filter_complex for xfade; concat demuxer for hard_cut) |

## Directory layout

```
kpop-fancam-studio/
├── backend/              # FastAPI app
│   ├── main.py           # app factory + static mount
│   ├── jobs.py           # subprocess job manager (polled via /api/jobs)
│   ├── yaml_store.py     # groups.yaml read/write
│   ├── mb_members.py     # MusicBrainz member lookup
│   └── routers/          # audio / groups / jobs / match / videos
├── frontend/             # React UI
│   └── src/              # QuickMode is the primary interface
├── oneshot_fancam.py     # CLI orchestrator (used by /api/videos/oneshot)
├── merge_sources.py      # multi-source merge planning + chunk rendering
├── pose_track.py         # RTMPose wrapper + yaw bucketing + sidecar cache
├── person_track.py       # YOLOv8 + BoT-SORT + InsightFace face ID
├── shot_gate.py          # TransNetV2 shot-boundary probe + cache
├── match_video.py        # chroma fingerprint → song offset
├── quality_probe.py      # stability / height / people-count scoring
├── search_videos.py      # yt-dlp metadata search
├── download_video.py     # per-URL downloader + metadata sidecar
├── download_tiktok.py    # (legacy) TikTok 60s official-cut ripper
├── fetch_discography.py  # (legacy) Spotify → groups.yaml
├── groups.yaml           # canonical per-group member + track roster
├── requirements.txt
└── .env.example          # Spotify creds for the legacy discography helper
```

## Setup

Tested on macOS (Apple Silicon). Linux should work. Windows is not the primary
target for the pipeline (ffmpeg / RTMPose paths assume homebrew-style). Use
the UI over HTTP from any browser.

```bash
# Clone
git clone https://github.com/<you>/kpop-fancam-studio.git
cd kpop-fancam-studio

# Python
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install \
  ultralytics \
  insightface onnxruntime \
  rtmlib \
  transnetv2-pytorch

# System deps (macOS)
brew install ffmpeg

# Model weights (not in git — too large)
mkdir -p models
# RTMPose-m body7:
curl -L -o models/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx
# YOLOv8n (ultralytics auto-downloads on first run; persist with:)
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# Frontend
cd frontend
npm install
npm run build      # or `npm run dev` for live reload on :5173
cd ..
```

`.env` is only needed for the legacy TikTok-audio helper (`download_tiktok.py`,
`fetch_discography.py`). The fancam pipeline does not require it.

## Running

```bash
# Backend (serves the built frontend at /)
./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Open http://<host>:8000/ in a browser
# Quick Mode tab → pick group / song / member / count → Go
```

CLI mode (the UI's "Go" button is just a wrapper around this):

```bash
./venv/bin/python oneshot_fancam.py \
  --group IVE \
  --song 'REBEL HEART' \
  --member-lat Yujin \
  --count 1 \
  --force-landscape \
  --merge-sources 3 \
  --merge-style hard_cut \
  --pose
```

Output: `clips/<group>/<song>/merged_*.mp4` (1080×1920, 30 fps, h264+aac).

## How the outfit-swap effect actually works

1. **Gate**: `shot_gate` runs TransNetV2 over every candidate. Cuts/sec
   > 0.10 → multi-camera broadcast edit → rejected. Real single-camera
   fancams sit at 0.005–0.02 cuts/sec. Broadcast stages run 0.22–0.36.

2. **Align**: chroma fingerprint against the official TikTok 60 s cut gives a
   per-source offset. All sources get cropped to the same 60 s window of the
   performance.

3. **Track**: YOLOv8 + BoT-SORT finds every person per frame. InsightFace
   scores each track against the target member's gallery vs. peer members —
   tracks below a dominance threshold are rejected ("peer confused" log). The
   surviving track is the chosen member.

4. **Pose-anchor**: RTMPose-m pulls COCO-17 keypoints. The **session head_y
   anchor** = median of per-source median head-y positions. Every clip is
   re-cropped so the head lands at this y-fraction (`target=0.30` in current
   builds) — that is the "same framing" illusion across cuts.

5. **Yaw bucket**: per-frame yaw proxy `(nose_x - eye_mid_x) / |eye_span|`
   ∈ [-1, 1]. Source median into frontal / left / right buckets (threshold
   ±0.10). `plan_merge` adds a bonus to candidates whose bucket matches the
   previous chunk's bucket — reduces flip-flopping between angles.

6. **Cut points**: beats from Beat This! define chunk boundaries. Each chunk
   gets assigned the highest-quality source that has target coverage there.
   Hard-cut = `ffmpeg concat demuxer`, xfade = `filter_complex`.

7. **Pose-refine snap**: for each cut, the planner evaluates ±3 frames
   around the beat-snapped boundary and picks the offset that minimizes
   `(yaw_outgoing - yaw_incoming)²`. Skipped when either side has low yaw
   confidence at every candidate frame, when the snap would shrink a
   neighbour below `min_chunk_sec`, or when the best candidate is still
   above the `max_acceptable_score` cap (yaw² > 0.25 ≈ yaw delta > 0.5
   means the cap kicks in and the cut stays at its beat-snapped position
   rather than landing on a wildly mismatched frame).

8. **M6-candidate flags**: cuts that the pose-refine pass refused to
   touch (low yaw-conf in the entire ±3-frame window, or best candidate
   above the score cap) are logged as `[merge] M6-candidate: cut K
   t=…s (vidA→vidB) …`. These are the cuts most likely to look jarring
   on screen and the natural inputs for a future RIFE interpolation
   pass. On the REBEL HEART reference run, 4 of 5 cuts get flagged
   because the pool sits almost entirely in the `frontal` bucket with
   noisy yaw confidence — exactly the case where pose-refine has
   nothing to grip on.

## Observed numbers (REBEL HEART session)

- Session head_y anchor: `0.304`
- Yaw buckets for 3 sources: 2 frontal (`+0.022`, `-0.033`) + 1 unknown
- Pose-anchor success: 189/189, 141/141, 199/225, 455/1033, 127/176 frames
  used head keypoint (remainder fell back to bbox center)
- Output: `1080×1920 @ 30fps`, `60.0s`, `h264+aac`, `~41 MB`, `5.5 Mbps`

## Known issues / caveats

- **얼빡직캠 (extreme close-up fancams)** get caught by the shot-gate because
  they tend to edit-zoom frequently (0.2+ cuts/sec). Consider relaxing the
  threshold if you want to include them.
- **Yaw proxy** requires all 3 of `nose`, `l_eye`, `r_eye` ≥ conf 0.30. Hair
  occlusion or motion blur → `unknown` bucket (works correctly in planner).
- **DMHead head-pose estimation** was investigated and abandoned — produced
  implausible Euler angles across multiple preprocessing strategies. Keypoint
  yaw proxy gave cleaner signal with no new dependencies.
- **TransNetV2 on MPS** (Apple Silicon GPU) has numerical drift vs. CPU —
  shot boundaries shift by ±1 frame. Pipeline forces CPU.

## Bonus: legacy TikTok audio ripper

The original goal of this repo was "rip the authoritative 60-second TikTok-cut
audio" for every song of a K-pop group. Those tools are still here and still
work — they feed the chroma fingerprint used above:

```bash
python fetch_discography.py IVE         # Spotify → groups.yaml
python download_tiktok.py --group IVE   # Playwright + ffmpeg → output/
```

See `download_tiktok.py` docstring for captcha / manual-URL fallback.

## License

MIT. See LICENSE.
