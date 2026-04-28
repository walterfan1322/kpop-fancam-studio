"""Match a downloaded video against a group's reference mp3s and extract the
60-second segment that aligns with the best-matching reference.

Algorithm (Path A — studio audio):
  1. Extract mono 22.05 kHz audio from the video via ffmpeg -> tmp wav bytes.
  2. Compute chroma CENS (12 x T, ~5 frames/sec) for the video and each mp3.
  3. Cross-correlate (FFT-based) to find offset with peak mean cosine-sim.
  4. If score > --threshold, ffmpeg-cut `<offset> + <ref_duration>` (capped at
     60s) into clips/<group>/<safe_song>.mp4 (re-encoded for frame accuracy).

Prints line-buffered progress and one final JSON `MATCH {...}` blob per run.

Usage:
    python match_video.py --group IVE --video videos/<stem>.mp4 [--threshold 0.6]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import static_ffmpeg
import yaml
from scipy.signal import fftconvolve

static_ffmpeg.add_paths()

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "groups.yaml"
OUTPUT_DIR = ROOT / "output"
VIDEOS_DIR = ROOT / "videos"
CLIPS_DIR = ROOT / "clips"

SR = 22050
HOP = 4096          # ~5.4 frames/sec @ 22.05 kHz -> enough for alignment
CLIP_MAX_SEC = 60.0


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name[:120] or "untitled"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- audio loading ----------

def ffmpeg_to_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Decode any media to mono 22.05 kHz float32 via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(SR), "-f", "wav", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    y, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def chroma(y: np.ndarray, sr: int) -> np.ndarray:
    # CENS chroma is robust to timbre/loudness changes, columns pre-normalised.
    c = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP)
    # Ensure per-frame L2 normalisation (CENS is already, but belt-and-braces).
    norms = np.linalg.norm(c, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return c / norms


# ---------- alignment ----------

@dataclass
class MatchResult:
    title: str
    ref_path: str
    score: float
    offset_sec: float
    ref_duration_sec: float


def best_offset(v_chroma: np.ndarray, r_chroma: np.ndarray) -> tuple[float, int]:
    """Return (score, frame_offset) of peak cross-correlation.

    Score is mean cosine similarity over ref frames at the peak offset.
    """
    T_r = r_chroma.shape[1]
    T_v = v_chroma.shape[1]
    if T_v < T_r:
        return 0.0, 0

    total = None
    for p in range(12):
        # Convolve video signal with time-reversed ref -> cross-correlation.
        c = fftconvolve(v_chroma[p], r_chroma[p, ::-1], mode="valid")
        total = c if total is None else total + c
    sim = total / T_r              # mean over ref frames; each frame ≤1
    idx = int(np.argmax(sim))
    return float(sim[idx]), idx


# ---------- ffmpeg clip extraction ----------

def _probe_size(video: Path) -> tuple[int, int] | None:
    """Return (width, height) via ffprobe, or None on failure."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(video),
        ], text=True).strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception:
        return None


_LAMA = None
_LAMA_FAILED = False


def _get_lama():
    """Lazy-load SimpleLama onto the best available device (MPS > CUDA > CPU).
    Returns None if the import fails so callers can fall back to Gaussian blur.
    """
    global _LAMA, _LAMA_FAILED
    if _LAMA is not None or _LAMA_FAILED:
        return _LAMA
    try:
        import warnings  # noqa: PLC0415
        # MPS backend emits UserWarnings about output-tensor resizing for every
        # inference call. Harmless but spams the job log.
        warnings.filterwarnings(
            "ignore",
            message=r"An output with one or more elements was resized.*",
            category=UserWarning,
        )
        import torch  # noqa: PLC0415
        from simple_lama_inpainting import SimpleLama  # noqa: PLC0415
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
        log(f"[lama] loading model on {dev}...")
        _LAMA = SimpleLama(device=dev)
        log("[lama] model ready")
    except Exception as e:
        log(f"[lama] unavailable ({e}); falling back to Gaussian blur")
        _LAMA_FAILED = True
        _LAMA = None
    return _LAMA


def _inpaint_rects_lama(frame_bgr: np.ndarray,
                        rects: list[tuple[int, int, int, int]],
                        padding: int = 48,
                        pixel_masks: list[np.ndarray] | None = None,
                        alpha_masks: list[np.ndarray] | None = None,
                        ) -> np.ndarray | None:
    """Run LaMa inpainting on a padded ROI around each rect and paste back.

    pixel_masks[i] (uint8 HxW, same shape as rect i): if provided, LaMa only
    fills those pixels (logo-shaped). Otherwise the whole rect is filled.
    alpha_masks[i] (float32 HxW in [0,1]): if provided, the inpainted result
    is alpha-blended back onto the original (1 = use LaMa output, 0 = keep
    original). Used to softly feather the mask edge so there's no visible seam.
    """
    lama = _get_lama()
    if lama is None:
        return None
    try:
        import cv2  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except Exception:
        return None
    H, W = frame_bgr.shape[:2]
    for i, (x, y, bw, bh) in enumerate(rects):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(W, x + bw + padding)
        y1 = min(H, y + bh + padding)
        sub = frame_bgr[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        mask = np.zeros(sub.shape[:2], dtype=np.uint8)
        mx0, my0 = x - x0, y - y0
        if pixel_masks is not None and i < len(pixel_masks) and pixel_masks[i] is not None:
            pm = pixel_masks[i]
            if pm.shape == (bh, bw):
                mask[my0:my0 + bh, mx0:mx0 + bw] = pm
        else:
            mask[my0:my0 + bh, mx0:mx0 + bw] = 255
        sub_rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
        out_pil = lama(Image.fromarray(sub_rgb), Image.fromarray(mask))
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        # LaMa pads to modulo 8 — trim back to the original crop size.
        out_bgr = out_bgr[:sub.shape[0], :sub.shape[1]]
        if alpha_masks is not None and i < len(alpha_masks) and alpha_masks[i] is not None:
            am = alpha_masks[i]  # HxWx1 float32 for the rect (not the padded ROI)
            # Build a full-ROI alpha: 0 outside the rect, `am` inside. Then
            # LaMa output replaces only the logo area, with smooth falloff.
            roi_alpha = np.zeros((sub.shape[0], sub.shape[1], 1), dtype=np.float32)
            roi_alpha[my0:my0 + bh, mx0:mx0 + bw] = am
            blended = (roi_alpha * out_bgr.astype(np.float32)
                       + (1.0 - roi_alpha) * sub.astype(np.float32))
            frame_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            frame_bgr[y0:y1, x0:x1] = out_bgr
    return frame_bgr


def _delogo_rect(video: Path, corner: str) -> tuple[int, int, int, int] | None:
    """Rough watermark box for a given corner, scaled to video size.

    Sized to catch broadcast-station bugs (e.g. MBC "예능연구소", MPD직캠),
    which often span >15% of the shorter side. Portrait videos need a taller
    box because logos like MPD's stacked "MPD / 직캠" mark scale with width
    but extend tall — a landscape-tuned 14% box is too short.
    """
    size = _probe_size(video)
    if not size:
        return None
    w, h = size
    side = min(w, h)
    portrait = h > w
    if portrait:
        bw = max(80, int(side * 0.25))
        bh = max(100, int(side * 0.30))
    else:
        bw = max(60, int(side * 0.22))
        bh = max(40, int(side * 0.14))
    margin = max(4, int(side * 0.01))
    if corner == "tl":
        return margin, margin, bw, bh
    if corner == "tr":
        return w - bw - margin, margin, bw, bh
    if corner == "bl":
        return margin, h - bh - margin, bw, bh
    if corner == "br":
        return w - bw - margin, h - bh - margin, bw, bh
    return None


def detect_watermark_corners(video: Path, static_frac_thresh: float = 0.12,
                             pixel_std_thresh: float = 15.0,
                             edge_spot_thresh: float = 0.15,
                             n_frames: int = 24) -> list[str]:
    """Return corners likely to hold a static overlay (watermark).

    Combines two signals per corner, each thresholded independently:

    1. Pixel-static fraction (+ sub-window spot) — catches SOLID logos where
       most of the logo region is truly constant across frames (e.g. the M2
       fill of "M COUNTDOWN").
    2. Edge persistence — per-pixel fraction of sampled frames where Canny
       reports an edge, thresholded at 70% of frames. OUTLINE logos with
       transparent interiors (e.g. MPD직캠's hollow "MPD/직캠" wordmark) have
       very low pixel-static frac but high edge persistence.

    A corner fires if either signal's sub-window peak exceeds its threshold.
    """
    size = _probe_size(video)
    if not size:
        return []
    try:
        import cv2  # noqa: PLC0415
    except Exception as e:
        log(f"[auto-delogo] cv2 unavailable: {e}")
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 2:
        cap.release()
        return []
    sample_idx = np.linspace(max(2, int(total * 0.05)),
                             max(3, int(total * 0.95)),
                             num=n_frames, dtype=int)
    corners = ("tl", "tr", "bl", "br")
    static_stacks: dict[str, list[np.ndarray]] = {c: [] for c in corners}
    edge_stacks: dict[str, list[np.ndarray]] = {c: [] for c in corners}
    for fi in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        for c in corners:
            rect = _delogo_rect(video, c)
            if not rect:
                continue
            x, y, bw, bh = rect
            patch = gray[y:y + bh, x:x + bw]
            epatch = edges[y:y + bh, x:x + bw]
            if patch.size:
                static_stacks[c].append(patch.astype(np.float32))
                edge_stacks[c].append((epatch > 0).astype(np.float32))
    cap.release()
    out: list[str] = []
    # Solid-logo "spot" threshold (existing): a filled logo only needs one
    # sub-window to be mostly static.
    static_spot_thresh = max(0.35, static_frac_thresh * 2.5)

    def _best_spot(mask: np.ndarray) -> float:
        hh, ww = mask.shape
        sh, sw = max(1, hh // 3), max(1, ww // 3)
        best = 0.0
        for iy in range(0, hh - sh + 1, max(1, sh // 2)):
            for ix in range(0, ww - sw + 1, max(1, sw // 2)):
                sub = mask[iy:iy + sh, ix:ix + sw]
                f = float(sub.mean())
                if f > best:
                    best = f
        return best

    for c in corners:
        if len(static_stacks[c]) < 3:
            continue
        arr = np.stack(static_stacks[c], axis=0)
        per_pixel_std = arr.std(axis=0)
        static_mask = per_pixel_std < pixel_std_thresh
        static_frac = float(static_mask.mean())
        static_spot = _best_spot(static_mask.astype(np.float32))
        # Edge persistence: at each pixel, fraction of frames where Canny
        # reports an edge. Logo outlines persist across nearly all frames.
        estack = np.stack(edge_stacks[c], axis=0)
        edge_persist = estack.mean(axis=0)  # 0..1 per pixel
        edge_mask = (edge_persist > 0.7).astype(np.float32)
        edge_spot = _best_spot(edge_mask)
        log(f"[auto-delogo] {video.name} {c}: "
            f"static={static_frac:.2f}/{static_spot:.2f} "
            f"edge_spot={edge_spot:.2f}")
        if (static_frac >= static_frac_thresh
                or static_spot >= static_spot_thresh
                or edge_spot >= edge_spot_thresh):
            out.append(c)
    # Guard against static-camera / static-backdrop shoots (e.g. dance
    # practice): if every corner looks "static", the scene just has a calm
    # background, not logos. Reject all and let the clip pass through clean.
    if len(out) >= 3:
        log(f"[auto-delogo] {len(out)}/4 corners flagged → looks like static "
            f"backdrop, skipping delogo")
        return []
    return out


def _build_logo_masks(video: Path,
                      rects: list[tuple[int, int, int, int]],
                      offset_sec: float, duration_sec: float,
                      dilate_px: int = 10, feather_px: int = 8,
                      n_frames: int = 20,
                      ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """For each rect, build a precise logo mask from edge persistence.

    Returns (pixel_masks, alpha_masks):
      pixel_masks[i]: uint8 HxW, 255 where LaMa should fill, 0 elsewhere.
                     Dilated to cover anti-aliasing and shadow halos.
      alpha_masks[i]: float32 HxWx1 in [0,1], the feathered version used to
                     alpha-blend LaMa output onto the original frame. Feather
                     extends outside the dilated mask so there is no hard
                     seam; interior stays 1 so the logo is fully removed.

    Hugging the logo shape (instead of a rectangle) is what makes the
    watermark removal look natural — the fill follows the logo's outline,
    non-logo pixels inside the detection rect stay untouched.
    """
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        # Fallback: solid rect mask
        solid = [np.full((bh, bw), 255, dtype=np.uint8) for _x, _y, bw, bh in rects]
        alpha = [np.ones((bh, bw, 1), dtype=np.float32) for _x, _y, bw, bh in rects]
        return solid, alpha
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start = int(round(offset_sec * fps))
    end = int(round((offset_sec + duration_sec) * fps))
    idxs = np.linspace(start, max(start + 1, end - 1), n_frames, dtype=int)
    edge_stacks: list[list[np.ndarray]] = [[] for _ in rects]
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        for i, (x, y, bw, bh) in enumerate(rects):
            patch = edges[y:y + bh, x:x + bw]
            if patch.shape == (bh, bw):
                edge_stacks[i].append((patch > 0).astype(np.float32))
    cap.release()
    pixel_masks: list[np.ndarray] = []
    alpha_masks: list[np.ndarray] = []
    for i, (x, y, bw, bh) in enumerate(rects):
        if len(edge_stacks[i]) < 3:
            # Not enough samples: fall back to solid
            pixel_masks.append(np.full((bh, bw), 255, dtype=np.uint8))
            alpha_masks.append(np.ones((bh, bw, 1), dtype=np.float32))
            continue
        persist = np.stack(edge_stacks[i], axis=0).mean(axis=0)
        logo = (persist > 0.5).astype(np.uint8) * 255
        # Dilate so the mask covers anti-aliasing, shadows, and the FILL inside
        # outline-only logos (solid punch-out).
        kd = 2 * dilate_px + 1
        dilated = cv2.dilate(logo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        # Fill enclosed holes so outline logos (like MPD's hollow "M") become
        # solid — otherwise LaMa leaves the interior untouched and the logo
        # silhouette is still visible.
        filled = dilated.copy()
        ff = np.zeros((filled.shape[0] + 2, filled.shape[1] + 2), np.uint8)
        cv2.floodFill(filled, ff, (0, 0), 255)
        filled = cv2.bitwise_not(filled)  # inverted: 255 where holes were
        solid = cv2.bitwise_or(dilated, filled)
        # Small extra dilate to get a generous paint area before feathering.
        solid = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        pixel_masks.append(solid)
        # Feathered alpha: 1 inside solid mask, smooth falloff outside.
        alpha = solid.astype(np.float32) / 255.0
        kf = 2 * feather_px + 1
        alpha = cv2.GaussianBlur(alpha, (kf, kf), feather_px / 2.0)
        alpha = np.clip(alpha, 0, 1)[..., None]
        alpha_masks.append(alpha)
    return pixel_masks, alpha_masks


def _extract_clip_lama_pipe(video: Path, offset_sec: float, duration_sec: float,
                            dst: Path, delogo_rects: list[tuple[int, int, int, int]]) -> None:
    """No-crop extraction that pipes per-frame LaMa-inpainted BGR24 to ffmpeg.
    Used when delogo_corners were detected and SimpleLama is available.
    """
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = int(round(offset_sec * fps))
    total_frames = int(round(duration_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    W2, H2 = W - (W & 1), H - (H & 1)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W2}x{H2}", "-r", f"{fps:.6f}",
        "-i", "-",
        "-ss", f"{max(0.0, offset_sec):.3f}",
        "-i", str(video),
        "-t", f"{duration_sec:.3f}",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # Build per-rect precise logo masks ONCE (edge-persistence across N frames
    # of the actual clip window). These hug the logo shape with a feathered
    # halo, so inpainting follows the logo outline instead of a hard rectangle.
    pixel_masks, alpha_masks = _build_logo_masks(
        video, delogo_rects, offset_sec, duration_sec)
    log(f"[auto-delogo] built precise masks for {len(delogo_rects)} rect(s)")
    # Re-inpaint every 3 frames — stage lights flicker fast enough that a
    # 5-frame stale fill shows as a pulsing patch against the surrounding
    # brightness. With a feathered alpha mask the cost is worth it.
    reinpaint_every = 3
    last_inpainted_frame: np.ndarray | None = None
    try:
        for i in range(total_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if i % reinpaint_every == 0:
                inpainted = _inpaint_rects_lama(
                    frame.copy(), delogo_rects,
                    pixel_masks=pixel_masks, alpha_masks=alpha_masks)
                if inpainted is not None:
                    last_inpainted_frame = inpainted
                else:
                    last_inpainted_frame = None
            if last_inpainted_frame is not None:
                # Stale-fill path: copy just the inpainted rect regions onto
                # the current frame, alpha-blended so the edge dissolves.
                for (x, y, bw, bh), alpha in zip(delogo_rects, alpha_masks):
                    if i % reinpaint_every == 0:
                        continue  # inpainted already holds the full result
                    fill = last_inpainted_frame[y:y + bh, x:x + bw]
                    patch = frame[y:y + bh, x:x + bw]
                    if fill.shape[:2] != patch.shape[:2]:
                        continue
                    blended = (alpha * fill.astype(np.float32)
                               + (1.0 - alpha) * patch.astype(np.float32))
                    frame[y:y + bh, x:x + bw] = np.clip(blended, 0, 255).astype(np.uint8)
                if i % reinpaint_every == 0:
                    frame = last_inpainted_frame
            if frame.shape[0] != H2 or frame.shape[1] != W2:
                frame = frame[:H2, :W2]
            proc.stdin.write(frame.tobytes())
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()


def _inflate_rect(video: Path, rect: tuple[int, int, int, int],
                  factor_w: float = 1.30, factor_h: float = 1.15
                  ) -> tuple[int, int, int, int]:
    """Enlarge a delogo rect for inpainting, clamped to frame bounds.

    Detection thresholds drop if the detection rect is too large (logo gets
    diluted by moving background), but inpainting needs to cover the *full*
    logo including subtext and decorations that sit slightly outside the
    detection box. So we detect with a tight rect and wipe with a larger one.
    """
    size = _probe_size(video)
    if not size:
        return rect
    W, H = size
    x, y, bw, bh = rect
    cx, cy = x + bw / 2.0, y + bh / 2.0
    new_bw = int(round(bw * factor_w))
    new_bh = int(round(bh * factor_h))
    # If the rect is anchored to a corner, grow it away from that corner
    # rather than symmetrically — avoids wasting inpaint area past the edge.
    left_anchored = x < W - (x + bw)
    top_anchored = y < H - (y + bh)
    if left_anchored:
        new_x = x
    else:
        new_x = x + bw - new_bw
    if top_anchored:
        new_y = y
    else:
        new_y = y + bh - new_bh
    new_x = max(0, min(W - 1, new_x))
    new_y = max(0, min(H - 1, new_y))
    new_bw = min(new_bw, W - new_x)
    new_bh = min(new_bh, H - new_y)
    return new_x, new_y, new_bw, new_bh


def extract_clip(video: Path, offset_sec: float, duration_sec: float, dst: Path,
                 delogo_corners: list[str] | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rects: list[tuple[int, int, int, int]] = []
    if delogo_corners:
        for corner in delogo_corners:
            rect = _delogo_rect(video, corner)
            if rect:
                rects.append(_inflate_rect(video, rect))
    # If LaMa is available and we have corners to wipe, pipe through LaMa for
    # clean inpainting instead of ffmpeg's delogo (which blurs with neighbor
    # pixels but can't remove high-contrast text bugs).
    if rects and _get_lama() is not None:
        _extract_clip_lama_pipe(video, offset_sec, duration_sec, dst, rects)
        return

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{max(0.0, offset_sec):.3f}",
        "-i", str(video),
        "-t", f"{duration_sec:.3f}",
    ]
    if rects:
        filters = [f"delogo=x={x}:y={y}:w={bw}:h={bh}" for x, y, bw, bh in rects]
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def extract_clip_tracked(video: Path, offset_sec: float, duration_sec: float,
                         dst: Path, trajectory: np.ndarray, meta,
                         out_w: int = 1080, out_h: int = 1920,
                         delogo_corners: list[str] | None = None,
                         h_traj: np.ndarray | None = None,
                         target_fill: float = 0.62,
                         h_anchor: float | None = None,
                         head_y_traj: np.ndarray | None = None,
                         head_y_conf: np.ndarray | None = None,
                         head_y_target: float = 0.22,
                         head_y_min_conf: float = 0.30,
                         color_gain: np.ndarray | None = None,
                         color_offset: np.ndarray | None = None) -> None:
    """Portrait-crop a landscape clip along `trajectory` (normalised cx,cy per
    output frame). When `h_traj` is supplied, the crop window SIZE varies per
    frame so the target person fills `target_fill` of the output height
    (~2/3 by default). When `h_traj` is None, the crop is a fixed 9:16 using
    the full source height — same behaviour as before.

    Because ffmpeg rawvideo needs a fixed input size per frame, we always
    pre-scale each variable crop to (out_w, out_h) in Python via Lanczos
    before piping. ffmpeg just muxes + encodes; the `-vf scale=` filter is
    therefore no longer needed.

    Audio is muxed from the source at the same offset/duration.
    """
    import cv2  # noqa: PLC0415

    dst.parent.mkdir(parents=True, exist_ok=True)
    W, H = meta.width, meta.height
    n = int(trajectory.shape[0])

    # ---- pre-compute per-frame crop dimensions ------------------------------
    # base_ch / base_cw: "fixed" 9:16 using full source height, the fallback
    # when we have no confident person bbox or h_traj is missing.
    base_ch = H - (H & 1)
    base_cw = int(round(base_ch * out_w / out_h)) & ~1
    if base_cw > W:
        # Source is already portrait/square — widen to full W, shrink height.
        base_cw = W - (W & 1)
        base_ch = int(round(base_cw * out_h / out_w)) & ~1

    # Per-frame crop sizes. If h_traj is available, derive crop_h so the
    # bbox fills `target_fill` of the crop height. EMA-smooth h_traj first
    # so the zoom breathes gently rather than pumping on every bbox jitter.
    MIN_FILL_RATIO = 0.35  # aesthetic floor: don't crop below 35% of source
    MAX_FILL_RATIO = 1.00  # crop can't exceed source height
    # Quality floor — cap the final upscale ratio (out_h / crop_h) at
    # MAX_UPSCALE. A 1080p source output to 1920x1080 portrait already
    # costs 1.78x upscale; zooming further quickly goes blurry. 2.0x is
    # the "soft but watchable" boundary. So:
    #   1080p source: min crop_h = 1920/2.0 = 960 (very little zoom room)
    #   4K source:    min crop_h = 960 (plenty of zoom headroom — 2160->960)
    # This gives source-resolution-aware behaviour for free: low-res
    # uploads zoom less, high-res uploads zoom aggressively.
    MAX_UPSCALE = 2.0
    quality_floor_h = int(round(out_h / MAX_UPSCALE))
    # Absolute floor is max of aesthetic + quality constraints.
    abs_floor_h = max(int(round(MIN_FILL_RATIO * H)), quality_floor_h)
    # Don't let the floor exceed the source — on 1080p with MAX_UPSCALE=2.0,
    # the floor (960) is below H (1080) so there's a sliver of zoom range.
    abs_floor_h = min(abs_floor_h, H)
    crop_h_arr = np.full(n, base_ch, dtype=np.int32)
    crop_w_arr = np.full(n, base_cw, dtype=np.int32)

    # Compute h_smooth: per-frame smoothed bbox-height in normalised units
    # (ratio of source height). Three paths:
    #   1) h_anchor   — session-level constant, all frames get the same zoom.
    #   2) h_traj     — EMA-smooth the observed per-frame bbox height.
    #   3) neither    — fall back to fixed full-height crop.
    h_smooth: np.ndarray | None
    if h_anchor is not None and 0.0 < float(h_anchor) <= 1.0:
        # Session-fixed zoom path: every frame uses the SAME crop_h, derived
        # from the session-level h_anchor. Used during multi-source merges so
        # the on-screen person fills the same fraction of the frame regardless
        # of which source the current chunk came from — required for hard-cut
        # "outfit-swap" cuts to read as costume changes rather than zoom
        # changes.
        h_smooth = np.full(n, float(h_anchor), dtype=np.float32)
        log(f"[track-crop] session-anchor zoom: h_anchor={h_anchor:.3f} "
            f"(all {n} frames use constant crop_h)")
    elif h_traj is not None and h_traj.size == n:
        # EMA smoothing on the raw bbox-height trajectory. Alpha=0.10 is
        # gentler than the cxcy alpha (0.18) — we want the zoom to lag
        # behind the person's position a little so it reads as deliberate
        # rather than jittery.
        alpha_h = 0.10
        h_smooth = np.zeros(n, dtype=np.float32)
        s = 0.0
        seeded = False
        for i in range(n):
            hv = float(h_traj[i])
            if hv > 0:
                if not seeded:
                    s = hv            # jump to first valid observation
                    seeded = True
                else:
                    s = alpha_h * hv + (1.0 - alpha_h) * s
            # else: freeze s (missing frame — carry last)
            h_smooth[i] = s
    else:
        h_smooth = None

    if h_smooth is not None:
        # Derive raw per-frame crop_h, then apply a secondary gentle EMA on
        # crop_h itself (alpha=0.08) so a single big bbox doesn't punch the
        # zoom in for one beat. With h_anchor, all h_smooth values are
        # identical so the secondary EMA is a no-op — kept for shared code
        # path.
        crop_h_raw = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if h_smooth[i] <= 0.0:
                ch = base_ch
            else:
                ratio = h_smooth[i] / target_fill  # desired crop_h / H
                ratio = max(MIN_FILL_RATIO, min(MAX_FILL_RATIO, ratio))
                ch = int(round(ratio * H))
                # Quality floor — never zoom so tight that the final
                # upscale ratio exceeds MAX_UPSCALE.
                if ch < abs_floor_h:
                    ch = abs_floor_h
            crop_h_raw[i] = ch
        alpha_c = 0.08
        ch_s = float(crop_h_raw[0])
        for i in range(n):
            ch_s = alpha_c * float(crop_h_raw[i]) + (1.0 - alpha_c) * ch_s
            ch = int(round(ch_s)) & ~1
            cw = int(round(ch * out_w / out_h)) & ~1
            if cw > W:
                cw = W - (W & 1)
                ch = int(round(cw * out_h / out_w)) & ~1
            crop_h_arr[i] = ch
            crop_w_arr[i] = cw
        mean_upscale = float(out_h) / float(crop_h_arr.mean())
        max_upscale_seen = float(out_h) / float(crop_h_arr.min())
        floor_hits = int(np.sum(crop_h_arr <= abs_floor_h + 1))
        log(f"[track-crop] dynamic zoom: crop_h "
            f"min={int(crop_h_arr.min())} max={int(crop_h_arr.max())} "
            f"mean={int(round(float(crop_h_arr.mean())))} "
            f"(src_H={H}, out_h={out_h}, target_fill={target_fill})")
        log(f"[track-crop] upscale: mean={mean_upscale:.2f}x max={max_upscale_seen:.2f}x "
            f"floor={abs_floor_h}px (hit {floor_hits}/{n} frames)")
        if floor_hits > n * 0.7:
            # Source is too low-res relative to output — we'd be upscaling
            # {MAX_UPSCALE}x across most frames. Revert to full-height fixed
            # crop: person appears smaller than the 2/3 target but quality
            # stays at baseline (out_h/H upscale only). Honesty > fake zoom.
            log(f"[track-crop] quality-floor clamped {floor_hits}/{n} frames "
                f"— reverting to fixed crop. Source too low-res for zoom "
                f"at target_fill={target_fill:.2f}.")
            crop_h_arr = np.full(n, base_ch, dtype=np.int32)
            crop_w_arr = np.full(n, base_cw, dtype=np.int32)
            log(f"[track-crop] fixed crop: {base_cw}x{base_ch} from {W}x{H} "
                f"(baseline upscale={out_h/H:.2f}x)")
    else:
        log(f"[track-crop] fixed crop: {base_cw}x{base_ch} from {W}x{H} "
            f"(baseline upscale={out_h/H:.2f}x)")

    # ---- M3b: pose-based vertical anchor ---------------------------------
    # When head_y_traj + head_y_conf are supplied, override the per-frame
    # crop y-origin so the dancer's head lands at `head_y_target` of the
    # output crop (default 0.22, upper third). Body-bbox top edges are
    # unstable (hair/hats/raised arms) and differ between sources shot at
    # different camera heights, which pops the head vertically across
    # hard cuts. RTMPose's nose+eyes midpoint is rock-steady.
    #
    # Fallback: frames with head_y_conf < head_y_min_conf keep the
    # cy-based y1 from `trajectory` (the bbox head-bias approximation).
    use_pose_y = (head_y_traj is not None
                  and head_y_conf is not None
                  and head_y_traj.size == n
                  and head_y_conf.size == n)
    pose_frames_used = 0

    # Pre-compute delogo rects once (in source-frame coords).
    delogo_rects: list[tuple[int, int, int, int]] = []
    if delogo_corners:
        for c in delogo_corners:
            r = _delogo_rect(video, c)
            if r:
                delogo_rects.append(_inflate_rect(video, r))

    # ffmpeg stdin is now the FINAL OUTPUT size (we pre-scale per frame).
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        # Video: rawvideo BGR24 on stdin, already at (out_w, out_h).
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}", "-r", f"{meta.fps:.6f}",
        "-i", "-",
        # Audio: from source at offset
        "-ss", f"{max(0.0, offset_sec):.3f}",
        "-i", str(video),
        "-t", f"{duration_sec:.3f}",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(str(video))
    start_frame = int(round(offset_sec * meta.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # Refresh the LaMa inpaint every ~5 frames and reuse the fill in between.
    # The watermark is static so this trades negligible quality for ~5x speed.
    reinpaint_every = 5
    cached_fills: list[np.ndarray] | None = None
    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if delogo_rects:
                if i % reinpaint_every == 0:
                    inpainted = _inpaint_rects_lama(frame.copy(), delogo_rects)
                    if inpainted is not None:
                        cached_fills = [inpainted[y:y + bh, x:x + bw].copy()
                                        for x, y, bw, bh in delogo_rects]
                    else:
                        cached_fills = None
                if cached_fills is not None:
                    # LaMa path: paste the cached inpainted patches over the
                    # watermark region on each frame.
                    for (x, y, bw, bh), fill in zip(delogo_rects, cached_fills):
                        frame[y:y + bh, x:x + bw] = fill
                else:
                    # Fallback: wide Gaussian blur covers high-contrast text.
                    for x, y, bw, bh in delogo_rects:
                        roi = frame[y:y + bh, x:x + bw]
                        if roi.size:
                            blurred = cv2.GaussianBlur(roi, (91, 91), 0)
                            blurred = cv2.GaussianBlur(blurred, (91, 91), 0)
                            frame[y:y + bh, x:x + bw] = blurred

            crop_h = int(crop_h_arr[i])
            crop_w = int(crop_w_arr[i])
            cx_n, cy_n = float(trajectory[i, 0]), float(trajectory[i, 1])
            cx = int(cx_n * W)
            cy = int(cy_n * H)
            x1 = max(0, min(W - crop_w, cx - crop_w // 2))
            if use_pose_y and float(head_y_conf[i]) >= head_y_min_conf:
                # Pose-anchored y: position crop so head lands at
                # head_y_target of the output. head_y_traj is in [0,1]
                # image coords; head_y_target is in [0,1] of the crop.
                head_y_px = float(head_y_traj[i]) * H
                y1 = int(round(head_y_px - head_y_target * crop_h))
                y1 = max(0, min(H - crop_h, y1))
                pose_frames_used += 1
            else:
                y1 = max(0, min(H - crop_h, cy - crop_h // 2))
            crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
                # Edge-of-frame clamp: re-cut with whatever we actually have.
                crop = cv2.resize(crop, (crop_w, crop_h))
            # Unconditional Lanczos resize to final output size — guarantees
            # ffmpeg stdin is the fixed size we promised it.
            if (crop_w, crop_h) != (out_w, out_h):
                crop = cv2.resize(crop, (out_w, out_h),
                                  interpolation=cv2.INTER_LANCZOS4)
            # Per-chunk color correction. Reinhard-style per-channel
            # (BGR) mean/std matching pre-computed by merge_sources so
            # chunks cut from different broadcasts harmonise in white
            # balance / contrast. No-op when params are None (single
            # source / reference chunk).
            if color_gain is not None and color_offset is not None:
                cropf = crop.astype(np.float32)
                cropf = cropf * color_gain + color_offset
                crop = np.clip(cropf, 0.0, 255.0).astype(np.uint8)
            proc.stdin.write(crop.tobytes())
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg (tracked) failed with rc={proc.returncode}")
    if use_pose_y:
        log(f"[track-crop] pose-anchor y: {pose_frames_used}/{n} frames "
            f"used head keypoint (target={head_y_target:.2f}, "
            f"min_conf={head_y_min_conf:.2f})")


# ---------- top-level ----------

def load_group_tracks(group: str, only_title: str | None = None) -> list[tuple[str, Path]]:
    """Return [(title, mp3)] for every track in the group that has an mp3 on disk.

    `only_title` is intentionally ignored here — we always load all references
    so the matcher can compare the target song against its siblings and catch
    false positives (K-pop tracks by the same artist often share key/tempo).
    The `only_title` guard lives in `run()` instead.
    """
    _ = only_title  # kept for backward-compat; see docstring
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    info = (data.get("groups") or {}).get(group)
    if not info:
        raise SystemExit(f"Group {group!r} not found in {YAML_PATH}")
    out: list[tuple[str, Path]] = []
    for t in info.get("tracks") or []:
        title = t if isinstance(t, str) else (t.get("title") or "")
        if not title:
            continue
        mp3 = OUTPUT_DIR / safe_filename(group) / f"{safe_filename(title)}.mp3"
        if mp3.exists():
            out.append((title, mp3))
    return out


def _match_audio(group: str, video: Path, threshold: float, margin: float,
                  only_title: str | None
                  ) -> tuple[dict, "MatchResult | None", str | None,
                             "MatchResult | None"]:
    """Shared front-end for `run()` (single-source extract) and
    `match_and_track()` (multi-source merge planning).

    Runs the chroma audio alignment across the group library, picks the
    best candidate, and applies the variant-accept / wrong-song /
    below-threshold / ambiguous guards. Does NOT extract a clip.

    Returns (out, best, reason, second) where `out` is the partial result
    dict containing video/group/candidates; `best` is the MatchResult the
    caller should use (may have been swapped to the requested variant by
    the variant-accept logic); `reason` is a string when we should skip
    this video, else None; `second` is the runner-up (for logging).
    """
    refs = load_group_tracks(group, only_title=only_title)
    if not refs:
        raise SystemExit(f"No downloaded mp3s for group {group!r}.")
    log(f"[load] video: {video.name}")
    y_v, _ = ffmpeg_to_mono_wav(video)
    v_chroma = chroma(y_v, SR)
    log(f"[chroma] video frames: {v_chroma.shape[1]}  ({y_v.size / SR:.1f}s)")

    results: list[MatchResult] = []
    for i, (title, mp3) in enumerate(refs, 1):
        try:
            y_r, _ = ffmpeg_to_mono_wav(mp3)
            r_chroma = chroma(y_r, SR)
        except Exception as e:
            log(f"[skip] {title}: {e}")
            continue
        score, idx = best_offset(v_chroma, r_chroma)
        offset_sec = idx * HOP / SR
        ref_dur = y_r.size / SR
        results.append(MatchResult(title, str(mp3), score, offset_sec, ref_dur))
        log(f"[{i:>2}/{len(refs)}] {title:<40} score={score:.3f} offset={offset_sec:6.2f}s")

    results.sort(key=lambda r: -r.score)
    best = results[0] if results else None
    second = results[1] if len(results) > 1 else None
    gap = (best.score - second.score) if (best and second) else (best.score if best else 0.0)

    try:
        rel_video = str(video.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        # video sits outside ROOT (e.g. symlinked data drive).
        rel_video = str(video).replace("\\", "/")
    out: dict = {
        "video": rel_video,
        "group": group,
        "threshold": threshold,
        "margin": margin,
        "margin_actual": gap,
        "candidates": [
            {"title": r.title, "score": r.score, "offset_sec": r.offset_sec,
             "ref_duration_sec": r.ref_duration_sec}
            for r in results[:5]
        ],
    }

    reason: str | None = None
    if not best:
        reason = "no_candidates"
    elif best.score < threshold:
        reason = "below_threshold"
    elif only_title and best.title.lower() != only_title.lower():
        # Oneshot: caller asked for song X but track Y scored higher.
        # Check this BEFORE `ambiguous` — a clearly-different-song video is
        # more informative as "wrong_song" than as "ambiguous".
        #
        # Variant tolerance: K-pop releases often ship an album cut alongside
        # a "(Short)" / "(Inst.)" / "(Intro)" variant sharing ~95% of the
        # chord structure. When both are in the reference library, the
        # "(Short)" edit can nudge 0.001-0.02 above the main cut — not
        # because the video is a different song, but because the short edit
        # aligns more tightly with the stage performance length. If the
        # requested title is in the top results within 0.02 of best, treat
        # them as the same song and extract as the requested title so the
        # clip lands in the right folder.
        requested_hit = next(
            (r for r in results if r.title.lower() == only_title.lower()), None
        )
        if requested_hit and (best.score - requested_hit.score) <= 0.02:
            log(f"[variant-accept] best={best.title}({best.score:.3f}) "
                f"≈ requested={only_title}({requested_hit.score:.3f}) "
                f"— treating as same-song variant")
            best = requested_hit
            out["candidates"] = [
                {"title": r.title, "score": r.score, "offset_sec": r.offset_sec,
                 "ref_duration_sec": r.ref_duration_sec}
                for r in results[:5]
            ]
        else:
            reason = "wrong_song"
            req_score = f"{requested_hit.score:.3f}" if requested_hit else "n/a"
            log(f"[skip:wrong-song] best={best.title}({best.score:.3f}) "
                f"requested={only_title}({req_score})")
    elif gap < margin and not only_title:
        # Ambiguity guard matters only when we're sweeping the full library
        # blind. In oneshot (only_title set), the caller has ALREADY told us
        # which song they want and best is that song — K-pop album tracks
        # often share chord structures so the runner-up scores close, but
        # that's not a reason to reject a correctly-identified clip.
        reason = "ambiguous"
    return out, best, reason, second


def match_and_track(group: str, video: Path, threshold: float, margin: float,
                     only_title: str | None,
                     member_lat: str, member_han: str = "",
                     disable_low_cov_fallback: bool = False,
                     ) -> tuple[dict, "object | None"]:
    """Audio-align + track-prepare one source for the multi-source merge.

    Returns (out, tracked_segment_or_None). On skip, `out['skip_reason']`
    is set and the second tuple element is None. On success, out contains
    matched_title / matched_score / matched_offset_sec / clip_dur and the
    second tuple element is a `person_track.TrackedSegment` ready to hand
    to `merge_sources.MergeSource`.

    Differs from `run(extract=True)` in two ways: never extracts a clip,
    and returns the tracked segment dataclass. Merge accepts both
    landscape sources (full track-crop + 9:16 reframe) AND portrait sources
    (solo-fancam direct cams; already 9:16 so we passthrough — tracking
    still runs to verify the target member). Square / unparseable aspect
    is returned with skip_reason 'not_landscape'; missing member with
    'no_member'.
    """
    out, best, reason, _ = _match_audio(group, video, threshold, margin,
                                         only_title)
    if reason:
        out["skip_reason"] = reason
        if best and reason == "below_threshold":
            log(f"[skip:below-threshold] best={best.title} "
                f"score={best.score:.3f} < {threshold}")
        return out, None
    assert best is not None
    clip_dur = min(CLIP_MAX_SEC, best.ref_duration_sec)
    size = _probe_size(video)
    is_landscape = bool(size and size[0] > size[1] * 1.2)
    is_portrait = bool(size and size[1] > size[0] * 1.2)
    if not (is_landscape or is_portrait):
        out["skip_reason"] = "not_landscape"
        log(f"[merge:skip] {video.name}: aspect not 16:9 or 9:16 (size={size})")
        return out, None
    if is_portrait:
        log(f"[merge:portrait] {video.name}: native 9:16 source "
            f"(size={size}) — track-crop will passthrough")
    if not member_lat:
        out["skip_reason"] = "no_member"
        log(f"[merge:skip] {video.name}: no member_lat provided")
        return out, None
    import person_track  # noqa: PLC0415
    face_root = ROOT / "face_library"
    yolo_weights = ROOT / "yolov8n.pt"
    try:
        ts = person_track.tracked_segment(
            video, best.offset_sec, clip_dur,
            group=group, member_lat=member_lat, member_han=member_han,
            face_root=face_root, yolo_weights=yolo_weights,
            disable_low_cov_fallback=disable_low_cov_fallback,
        )
    except Exception as e:
        out["skip_reason"] = "track_error"
        log(f"[merge:skip] {video.name}: track error: {e}")
        return out, None

    out["matched_title"] = best.title
    out["matched_score"] = best.score
    out["matched_offset_sec"] = best.offset_sec
    out["clip_dur"] = clip_dur
    out["crop_mode"] = ts.mode
    return out, ts


def run(group: str, video: Path, threshold: float, margin: float, extract: bool,
        only_title: str | None = None, suffix: str | None = None,
        delogo_corners: list[str] | None = None,
        track_crop: bool = False,
        member_lat: str = "", member_han: str = "") -> dict:
    out, best, reason, second = _match_audio(group, video, threshold, margin,
                                              only_title)
    gap = out["margin_actual"]

    if best and not reason and extract:
        clip_dur = min(CLIP_MAX_SEC, best.ref_duration_sec)
        fname = safe_filename(suffix) if suffix else "main"
        dst = CLIPS_DIR / safe_filename(group) / safe_filename(best.title) / f"{fname}.mp4"
        log(f"[extract] {best.title}  offset={best.offset_sec:.2f}s  dur={clip_dur:.2f}s"
            f"  margin={gap:.3f}")
        corners = delogo_corners
        if corners and "auto" in corners:
            detected = detect_watermark_corners(video)
            log(f"[auto-delogo] detected corners: {detected or 'none'}")
            corners = detected
        crop_mode = None
        # Auto: landscape sources get a 9:16 tracked portrait crop; videos
        # that are already portrait/square skip tracking entirely.
        size = _probe_size(video)
        is_landscape = bool(size and size[0] > size[1] * 1.2)
        use_track = track_crop and is_landscape and bool(member_lat)
        if track_crop and not use_track:
            log(f"[track-crop] skipping — source size={size} member={member_lat!r}")
        if use_track:
            try:
                import person_track  # noqa: PLC0415
                face_root = ROOT / "face_library"
                yolo_weights = ROOT / "yolov8n.pt"
                trajectory, h_traj, meta, crop_mode = person_track.plan_crop(
                    video, best.offset_sec, clip_dur,
                    group=group, member_lat=member_lat, member_han=member_han,
                    face_root=face_root, yolo_weights=yolo_weights,
                )
                log(f"[track-crop] mode={crop_mode} frames={meta.n_frames} "
                    f"src={meta.width}x{meta.height} "
                    f"{'dynamic-zoom=on' if h_traj is not None else 'dynamic-zoom=off'}")
                extract_clip_tracked(video, best.offset_sec, clip_dur, dst,
                                     trajectory, meta, delogo_corners=corners,
                                     h_traj=h_traj)
            except Exception as e:
                log(f"[track-crop:error] {e} — falling back to plain crop")
                crop_mode = "fallback"
                extract_clip(video, best.offset_sec, clip_dur, dst,
                             delogo_corners=corners)
        else:
            extract_clip(video, best.offset_sec, clip_dur, dst, delogo_corners=corners)
        try:
            out["clip_path"] = str(dst.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            out["clip_path"] = str(dst).replace("\\", "/")
        out["matched_title"] = best.title
        out["matched_score"] = best.score
        out["matched_offset_sec"] = best.offset_sec
        if crop_mode:
            out["crop_mode"] = crop_mode
        log(f"[ok] clip -> {out['clip_path']}")
    elif best and reason == "below_threshold":
        log(f"[skip:below-threshold] best={best.title} score={best.score:.3f} < {threshold}")
    elif best and reason == "ambiguous":
        runner = second.title if second else "?"
        log(f"[skip:ambiguous] best={best.title}({best.score:.3f}) vs "
            f"{runner}({second.score:.3f})  gap={gap:.3f} < margin {margin}")
    if reason:
        out["skip_reason"] = reason
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--video", required=True, help="path to downloaded video file")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="absolute score floor (rarely the deciding factor)")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="min gap between top1 and top2 scores; guards against "
                         "ambiguous matches since K-pop chord progressions overlap")
    ap.add_argument("--no-extract", action="store_true",
                    help="only report scores, do not write clip")
    ap.add_argument("--only-title", default=None,
                    help="restrict matching to a single song title (case-insensitive)")
    ap.add_argument("--delogo-corners", default=None,
                    help="comma-separated corners (tl,tr,bl,br) to apply ffmpeg delogo")
    ap.add_argument("--track-crop", action="store_true",
                    help="YOLO+face-ID follow the target member; output 9:16 portrait")
    ap.add_argument("--member-lat", default="",
                    help="latin member name for face-ID target (track-crop)")
    ap.add_argument("--member-han", default="",
                    help="hangul member name (optional)")
    args = ap.parse_args()
    corners = None
    if args.delogo_corners:
        corners = [c.strip() for c in args.delogo_corners.split(",")
                   if c.strip() in ("tl", "tr", "bl", "br", "auto")]

    video = Path(args.video)
    if not video.is_absolute():
        video = (ROOT / video).resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    result = run(args.group, video, args.threshold, args.margin,
                 extract=not args.no_extract, only_title=args.only_title,
                 delogo_corners=corners,
                 track_crop=args.track_crop,
                 member_lat=args.member_lat, member_han=args.member_han)
    print("MATCH " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
