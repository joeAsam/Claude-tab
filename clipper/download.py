"""Downloads a past stream (VOD) and its chat replay with yt-dlp.

For YouTube, the live chat replay is saved too (as <name>.live_chat.json), which
gives the clipper real chat reactions with timestamps, just like a live stream.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".mkv", ".webm")


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/videos/|/live/|/shorts/)([\w-]{6,})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-24:]


def download_vod(url: str, out_dir: Path, first_minutes: float | None, max_height: int = 1080) -> tuple[Path, Path | None]:
    """Download the video (only the first `first_minutes` if given) and its chat replay.
    Returns (video path, chat replay path or None)."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed: pip install 'yt-dlp[default]'")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_id(url) + (f"_first{first_minutes:g}m" if first_minutes else "")
    template = str(out_dir / f"{stem}.%(ext)s")

    existing = next((p for p in (out_dir / f"{stem}{e}" for e in VIDEO_EXTS) if p.exists()), None)
    if existing:
        log.info("Using the earlier download %s", existing)
        video = existing
    else:
        cmd = [
            "yt-dlp", "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
            "--merge-output-format", "mp4", "-o", template, "--no-playlist",
        ]
        if first_minutes:
            cmd += ["--download-sections", f"*0-{int(first_minutes * 60)}"]
        log.info("Downloading %s%s...", url, f" (first {first_minutes:g} minutes)" if first_minutes else "")
        subprocess.run(cmd + [url], check=True)
        video = next((p for p in (out_dir / f"{stem}{e}" for e in VIDEO_EXTS) if p.exists()), None)
        if not video:
            raise RuntimeError(f"yt-dlp finished but no video was found in {out_dir}")

    chat = out_dir / f"{stem}.live_chat.json"
    if not chat.exists():
        log.info("Downloading the chat replay (if the stream has one)...")
        # The chat replay always covers the whole stream; the clipper ignores messages
        # past the end of the downloaded part.
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
             "-o", template, "--no-playlist", url],
        )
        if result.returncode != 0 or not chat.exists():
            log.info("No chat replay available; detecting moments from audio, video and speech.")
            return video, None
    return video, chat
