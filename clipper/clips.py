"""Cuts raw clips out of the segment buffer with ffmpeg."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .recorder import StreamRecorder

VERTICAL_FILTER = "crop=ih*9/16:ih,scale=1080:1920"


@dataclass
class Clip:
    id: str
    start: float  # unix timestamps
    end: float
    title: str
    reason: str
    raw_path: Path
    path: Path | None = None  # the finished clip, once rendered
    edits: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len] or "clip"


def clip_basename(start: float, title: str, vertical: bool) -> str:
    stamp = datetime.fromtimestamp(start).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{slugify(title)}{'_vertical' if vertical else ''}"


def cut_clip(recorder: StreamRecorder, start: float, end: float, out_path: Path, vertical: bool = False) -> Path:
    """Write the [start, end] wall-clock range to an .mp4 and return its path."""
    segs = [s for s in recorder.segments() if s.end > start and s.start < end]
    if not segs:
        raise ValueError("That time range is no longer (or not yet) in the buffer.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [s.path for s in segs]
    recorder.pin(paths)  # keep the pruner from deleting them mid-cut
    list_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
            for p in paths:
                lst.write(f"file '{p.resolve()}'\n")
            list_path = Path(lst.name)

        offset = max(start - segs[0].start, 0)
        duration = min(end, segs[-1].end) - max(start, segs[0].start)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-ss", f"{offset:.2f}", "-t", f"{duration:.2f}",
        ]
        if vertical:
            cmd += ["-vf", VERTICAL_FILTER]
        # Re-encode so the cut lands exactly where the agent asked, not on a keyframe.
        # High quality because this is an intermediate that gets edited again.
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        recorder.unpin(paths)
        if list_path:
            list_path.unlink(missing_ok=True)
    return out_path
