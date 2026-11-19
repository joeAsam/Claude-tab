"""Treats a recorded video file (e.g. a screen recording of a stream) as a source.

The file is cropped to the stream player and split into the same 10-second
segments the live recorder produces, placed on a timeline that starts at
`virtual_start`. That way the signal detector, frame grabber, clip cutter and
editor all work on recordings exactly as they do on live streams.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .recorder import Segment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Crop:
    w: int
    h: int
    x: int
    y: int

    @classmethod
    def parse(cls, text: str) -> "Crop":
        """Parse ffmpeg-style W:H:X:Y."""
        try:
            w, h, x, y = (int(v) for v in text.split(":"))
        except ValueError:
            raise ValueError(f"crop must look like W:H:X:Y (e.g. 1280:720:0:80), got {text!r}") from None
        return cls(w, h, x, y)

    @property
    def filter(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"

    def __str__(self) -> str:
        return f"{self.w}:{self.h}:{self.x}:{self.y}"


class FileSource:
    """Same interface the live StreamRecorder offers to the rest of the clipper."""

    def __init__(
        self,
        path: Path,
        work_dir: Path,
        duration: float,
        player: Crop | None = None,
        chat: Crop | None = None,
        segment_seconds: int = 10,
        virtual_start: float = 1_000_000_000.0,
    ) -> None:
        self.path = path
        self.work_dir = work_dir
        self.duration = duration
        self.player = player
        self.chat = chat
        self.segment_seconds = segment_seconds
        self.virtual_start = virtual_start
        self.buffer_seconds = 10**9  # nothing is ever pruned
        self._segments: list[Segment] = []

    def prepare(self) -> None:
        """Crop to the player and split into segments (re-encodes once)."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for old in self.work_dir.glob("seg_*.ts"):
            old.unlink()
        log.info("Preparing %s (%.0f min); this takes a while for long recordings...",
                 self.path.name, self.duration / 60)
        vf = [self.player.filter] if self.player else []
        vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(self.path),
                "-vf", ",".join(vf),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                # Regular keyframes so the recording splits close to every segment_seconds.
                "-force_key_frames", "expr:gte(t,n_forced*2)",
                "-c:a", "aac", "-b:a", "192k",
                "-f", "segment", "-segment_time", str(self.segment_seconds), "-reset_timestamps", "1",
                str(self.work_dir / "seg_%05d.ts"),
            ],
            check=True,
        )
        # ffmpeg can only split on keyframes, so pieces aren't always exactly segment_seconds
        # long. Measure each one so the timeline matches the original recording.
        self._segments = []
        t = self.virtual_start
        for f in sorted(self.work_dir.glob("seg_*.ts")):
            length = _duration(f)
            self._segments.append(Segment(f, t, t + length))
            t += length
        log.info("Prepared %d segments (%.1f of %.1f seconds)",
                 len(self._segments), t - self.virtual_start, self.duration)

    # ---- StreamRecorder-compatible interface -------------------------------

    def segments(self, include_open: bool = False) -> list[Segment]:
        return list(self._segments)

    def available_range(self) -> tuple[float, float] | None:
        if not self._segments:
            return None
        return self._segments[0].start, self._segments[-1].end

    def pin(self, paths) -> None:
        pass

    def unpin(self, paths) -> None:
        pass

    # ---- extras for recordings -------------------------------------------

    def grab_chat_frame(self, t: float, max_height: int = 1000) -> bytes | None:
        """JPEG of the chat panel at timeline time t, at a resolution good enough to read."""
        if not self.chat:
            return None
        offset = t - self.virtual_start
        if not 0 <= offset <= self.duration:
            return None
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.2f}", "-i", str(self.path),
             "-frames:v", "1", "-vf", f"{self.chat.filter},scale=-2:'min({max_height},ih)'", "-q:v", "3",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=30,
        )
        return result.stdout or None


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return {"width": int(info["width"]), "height": int(info["height"]), "duration": float(info["duration"])}
