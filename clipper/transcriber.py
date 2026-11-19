"""Optional speech-to-text for buffered segments, using faster-whisper.

If faster-whisper isn't installed, the agent works from chat activity alone.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .recorder import Segment, StreamRecorder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptLine:
    start: float
    end: float
    text: str


class Transcriber:
    def __init__(self, recorder: StreamRecorder, model_size: str = "base") -> None:
        from faster_whisper import WhisperModel  # imported lazily: optional dependency

        self.recorder = recorder
        self.model = WhisperModel(model_size, device="auto", compute_type="int8")
        self._lines: list[TranscriptLine] = []
        self._done: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @classmethod
    def try_create(cls, recorder: StreamRecorder, model_size: str) -> "Transcriber | None":
        try:
            return cls(recorder, model_size)
        except ImportError:
            log.info("faster-whisper not installed; running without transcripts")
            return None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="transcriber").start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(2):
            for seg in self.recorder.segments():
                if seg.path.name in self._done or self._stop.is_set():
                    continue
                self._transcribe(seg)
                self._done.add(seg.path.name)

    def _transcribe(self, seg: Segment) -> None:
        try:
            parts, _ = self.model.transcribe(str(seg.path), vad_filter=True)
            lines = [TranscriptLine(seg.start + p.start, seg.start + p.end, p.text.strip()) for p in parts]
        except Exception as exc:  # a corrupt or deleted segment shouldn't kill the thread
            log.debug("Transcription failed for %s: %s", seg.path.name, exc)
            return
        with self._lock:
            self._lines.extend(lines)
            cutoff = seg.start - self.recorder.buffer_seconds
            self._lines = [ln for ln in self._lines if ln.end >= cutoff]

    def lines_between(self, start: float, end: float) -> list[TranscriptLine]:
        with self._lock:
            return [ln for ln in self._lines if ln.end > start and ln.start < end]
