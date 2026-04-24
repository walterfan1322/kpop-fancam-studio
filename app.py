"""Streamlit UI for the kpop_tiktok_audio pipeline.

Run:
    streamlit run app.py

Or use ./run_ui.bat (Windows).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "groups.yaml"
OUT_DIR = ROOT / "output"
INDEX_PATH = OUT_DIR / "index.json"

st.set_page_config(page_title="K-pop TikTok 音訊抓取", page_icon="🎵", layout="wide")


# ---------- yaml helpers ----------
def load_yaml() -> dict:
    if not YAML_PATH.exists():
        return {"groups": {}}
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    data.setdefault("groups", {})
    return data


def save_yaml(data: dict) -> None:
    YAML_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=2),
        encoding="utf-8",
    )


def load_index() -> dict:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {}


def track_title(t) -> str:
    if isinstance(t, str):
        return t
    return t.get("title") or t.get("name") or ""


def track_url(t) -> str:
    if isinstance(t, dict):
        return t.get("music_url") or ""
    return ""


def mp3_path(group: str, title: str) -> Path:
    safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in title).strip()[:120]
    return OUT_DIR / group / f"{safe}.mp3"


# ---------- subprocess streaming ----------
def run_stream(cmd: list[str], log_placeholder) -> int:
    """Run a command, streaming stdout lines into a Streamlit placeholder."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        # Show last ~40 lines
        log_placeholder.code("\n".join(lines[-40:]), language="text")
    proc.wait()
    return proc.returncode


# ---------- sidebar: add group via Spotify ----------
with st.sidebar:
    st.header("🎤 加入團體")
    st.caption("從 MusicBrainz 抓 discography 寫進 groups.yaml")
    new_group = st.text_input("團名", placeholder="IVE / aespa / NewJeans / LE SSERAFIM")
    limit = st.number_input("歌曲上限 (0 = 不限)", 0, 200, 30, step=5)
    if st.button("🔍 抓歌單", type="primary", width="stretch"):
        if not new_group.strip():
            st.error("填團名")
        else:
            args = [sys.executable, "fetch_discography.py", new_group.strip()]
            if limit:
                args += ["--limit", str(limit)]
            with st.status(f"MusicBrainz: {new_group}", expanded=True) as status:
                log = st.empty()
                rc = run_stream(args, log)
                if rc == 0:
                    status.update(label=f"✅ 已加入 {new_group}", state="complete")
                    st.rerun()
                else:
                    status.update(label="❌ 失敗", state="error")

    st.divider()
    st.header("⚙️ 全域設定")
    delay = st.slider("每首間隔秒數", 1.0, 10.0, 3.0, 0.5)
    headed = st.checkbox("顯示瀏覽器 (第一次建議打開解 captcha)", value=False)


# ---------- main: list groups ----------
st.title("🎵 K-pop TikTok 音訊抓取")

data = load_yaml()
index = load_index()
groups = data["groups"]

if not groups:
    st.info("左邊還沒加團體。輸入團名（如 IVE）按「抓歌單」開始。")
    st.stop()

# Top-level metrics
total_tracks = sum(len(g.get("tracks") or []) for g in groups.values())
total_mp3s = sum(1 for g in groups for t in (groups[g].get("tracks") or [])
                 if mp3_path(g, track_title(t)).exists())
cols = st.columns(3)
cols[0].metric("團數", len(groups))
cols[1].metric("歌曲總數", total_tracks)
cols[2].metric("已抓音訊", f"{total_mp3s} / {total_tracks}")

# Select which group to view
group_names = list(groups.keys())
tab_objs = st.tabs(group_names)

for tab, group in zip(tab_objs, group_names):
    info = groups[group]
    tracks = info.get("tracks") or []
    with tab:
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.caption(f"Artist: `{info.get('artist_name', group)}` "
                     f"(mbid `{info.get('mb_artist_id', info.get('spotify_artist_id', '?'))}`)")

        # Build table
        rows = []
        for t in tracks:
            title = track_title(t)
            url = track_url(t)
            mp3 = mp3_path(group, title)
            entry = index.get(group, {}).get(title, {})
            if mp3.exists():
                status = "✅ 已下載"
            elif entry.get("status") == "ok":
                status = "✅"
            elif entry.get("status") == "no_music_match":
                status = "❌ 搜尋不到"
            elif entry.get("status") == "no_audio":
                status = "⚠ 無音訊"
            elif url:
                status = "🔗 有 URL,待抓"
            else:
                status = "○ 待處理"
            rows.append({
                "title": title,
                "status": status,
                "music_url": url,
            })
        df = pd.DataFrame(rows)

        edited = st.data_editor(
            df,
            key=f"editor_{group}",
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "title": st.column_config.TextColumn("歌名", required=True, width="medium"),
                "status": st.column_config.TextColumn("狀態", disabled=True, width="small"),
                "music_url": st.column_config.LinkColumn(
                    "TikTok music URL (空=自動搜尋)",
                    width="large",
                    validate=r"^(https://www\.tiktok\.com/music/.+)?$",
                ),
            },
        )

        btns = st.columns([1, 1, 1, 3])
        if btns[0].button("💾 存回 yaml", key=f"save_{group}"):
            new_tracks = []
            for _, row in edited.iterrows():
                title = (row.get("title") or "").strip()
                if not title:
                    continue
                url = (row.get("music_url") or "").strip()
                new_tracks.append({"title": title, "music_url": url} if url else title)
            data["groups"][group]["tracks"] = new_tracks
            save_yaml(data)
            st.success("已存檔")
            st.rerun()

        if btns[1].button("⬇️ 下載這團", key=f"dl_{group}", type="primary"):
            args = [sys.executable, "download_tiktok.py",
                    "--group", group, "--delay", str(delay)]
            if headed:
                args.append("--headed")
            with st.status(f"下載 {group}", expanded=True) as status:
                log = st.empty()
                rc = run_stream(args, log)
                if rc == 0:
                    status.update(label=f"✅ {group} 完成", state="complete")
                    st.rerun()
                else:
                    status.update(label="❌ 失敗", state="error")

        if btns[2].button("🗑 刪除此團", key=f"del_{group}"):
            st.session_state[f"confirm_del_{group}"] = True
        if st.session_state.get(f"confirm_del_{group}"):
            st.warning(f"確定要刪除 **{group}** 嗎？（只刪 yaml 條目，不會刪 mp3）")
            c1, c2 = st.columns(2)
            if c1.button("確認刪除", key=f"del2_{group}"):
                del data["groups"][group]
                save_yaml(data)
                st.session_state.pop(f"confirm_del_{group}", None)
                st.rerun()
            if c2.button("取消", key=f"cancel_{group}"):
                st.session_state.pop(f"confirm_del_{group}", None)
                st.rerun()

        # Audio preview section
        downloaded = [t for t in tracks
                      if mp3_path(group, track_title(t)).exists()]
        if downloaded:
            with st.expander(f"🎧 試聽 ({len(downloaded)})"):
                for t in downloaded:
                    title = track_title(t)
                    p = mp3_path(group, title)
                    st.write(f"**{title}** — `{p.stat().st_size // 1024} KB`")
                    st.audio(str(p))
