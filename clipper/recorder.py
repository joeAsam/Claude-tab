"""Records a live stream into a rolling buffer of short video segments.

streamlink pulls the stream and pipes it into ffmpeg, which writes fixed-length
.ts segments named by their wall-clock start time. Old segments are pruned so
disk use stays bounded, and clips are cut from whatever is still in the buffer.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

SEGMENT_NAME_FORMAT = "seg_%Y%m%d_%H%M%S.ts"


@dataclass(frozen=True)
class Segment:
    path: Path
    start: float  # unix timestamp
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class StreamRecorder:
    def __init__(
        self,
        stream_url: str,
        buffer_dir: Path,
        segment_seconds: int = 10,
        buffer_minutes: int = 15,
        quality: str = "best",
    ) -> None:
        self.stream_url = stream_url
        self.buffer_dir = buffer_dir
        self.segment_seconds = segment_seconds
        self.buffer_seconds = buffer_minutes * 60
        self.quality = quality
        self._streamlink: subprocess.Popen | None = None
        self._ffmpeg: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._pinned: set[Path] = set()
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._supervise, daemon=True, name="recorder").start()
        threading.Thread(target=self._prune_loop, daemon=True, name="pruner").start()

    def stop(self) -> None:
        self._stop.set()
        self._kill()

    def _launch(self) -> None:
        self._streamlink = subprocess.Popen(
            ["streamlink", "--stdout", "--twitch-disable-ads", self.stream_url, self.quality],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-c", "copy",
                "-f", "segment",
                "-segment_time", str(self.segment_seconds),
                "-reset_timestamps", "1",
                "-strftime", "1",
                str(self.buffer_dir / SEGMENT_NAME_FORMAT),
            ],
            stdin=self._streamlink.stdout,
        )
        # Let ffmpeg own the pipe so streamlink gets SIGPIPE if ffmpeg dies.
        assert self._streamlink.stdout is not None
        self._streamlink.stdout.close()

    def _kill(self) -> None:
        for proc in (self._ffmpeg, self._streamlink):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _supervise(self) -> None:
        """Keep the recording running; restart if the stream drops."""
        backoff = 5
        while not self._stop.is_set():
            log.info("Connecting to %s", self.stream_url)
            started = time.time()
            self._launch()
            assert self._ffmpeg is not None
            self._ffmpeg.wait()
            self._kill()
            if self._stop.is_set():
                break
            # A run that lasted a while was a real session: reset the backoff.
            backoff = 5 if time.time() - started > 60 else min(backoff * 2, 120)
            log.warning("Stream ended or went offline; retrying in %ss", backoff)
            self._stop.wait(backoff)

    # ---- buffer access ---------------------------------------------------

    def segments(self, include_open: bool = False) -> list[Segment]:
        """Segments in the buffer, oldest first.

        The newest file is still being written, so it is excluded unless
        include_open is True.
        """
        files = sorted(self.buffer_dir.glob("seg_*.ts"))
        starts = []
        for f in files:
            try:
                starts.append(datetime.strptime(f.name, SEGMENT_NAME_FORMAT).timestamp())
            except ValueError:
                continue
        result: list[Segment] = []
        for i, (f, start) in enumerate(zip(files, starts)):
            if i + 1 < len(starts):
                end = starts[i + 1]
                # A gap (stream dropped) means this segment ended early.
                end = min(end, start + self.segment_seconds * 1.5)
            elif include_open:
                end = time.time()
            else:
                continue
            result.append(Segment(f, start, end))
        return result

    def available_range(self) -> tuple[float, float] | None:
        segs = self.segments()
        if not segs:
            return None
        return segs[0].start, segs[-1].end

    def pin(self, paths: list[Path]) -> None:
        with self._lock:
            self._pinned.update(paths)

    def unpin(self, paths: list[Path]) -> None:
        with self._lock:
            self._pinned.difference_update(paths)

    def _prune_loop(self) -> None:
        while not self._stop.wait(self.segment_seconds):
            cutoff = time.time() - self.buffer_seconds
            for seg in self.segments():
                with self._lock:
                    if seg.end < cutoff and seg.path not in self._pinned:
                        seg.path.unlink(missing_ok=True)
