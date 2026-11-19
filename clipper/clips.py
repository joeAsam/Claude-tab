"""Cuts clips out of the segment buffer with ffmpeg."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .recorder import StreamRecorder

VERTICAL_FILTER = "crop=ih*9/16:ih,scale=1080:1920"


@dataclass(frozen=True)
class Clip:
    path: Path
    start: float
    end: float
    title: str
    reason: str


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len] or "clip"


def cut_clip(
    recorder: StreamRecorder,
    start: float,
    end: float,
    out_dir: Path,
    title: str,
    vertical: bool = False,
) -> Path:
    """Write the [start, end] wall-clock range to an .mp4 and return its path."""
    segs = [s for s in recorder.segments() if s.end > start and s.start < end]
    if not segs:
        raise ValueError("That time range is no longer (or not yet) in the buffer.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(start).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_{slugify(title)}{'_vertical' if vertical else ''}.mp4"

    paths = [s.path for s in segs]
    recorder.pin(paths)  # keep the pruner from deleting them mid-cut
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
            for p in paths:
                lst.write(f"file '{p.resolve()}'\n")
            list_path = lst.name

        offset = max(start - segs[0].start, 0)
        duration = min(end, segs[-1].end) - max(start, segs[0].start)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-ss", f"{offset:.2f}", "-t", f"{duration:.2f}",
        ]
        if vertical:
            cmd += ["-vf", VERTICAL_FILTER]
        # Re-encode so the cut lands exactly where the agent asked, not on a keyframe.
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        Path(list_path).unlink(missing_ok=True)
    finally:
        recorder.unpin(paths)
    return out_path
