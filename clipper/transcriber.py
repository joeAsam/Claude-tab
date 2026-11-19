"""Optional live speech-to-text for buffered segments, using faster-whisper.

This lets the agent read what the streamer said when deciding what to clip.
(Captions are made separately, from each finished clip.)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from . import whisper
from .recorder import Segment, StreamRecorder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptLine:
    start: float
    end: float
    text: str


class Transcriber:
    def __init__(self, recorder: StreamRecorder, model_size: str = "base") -> None:
        self.recorder = recorder
        self.model_size = model_size
        self._lines: list[TranscriptLine] = []
        self._done: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @classmethod
    def try_create(cls, recorder: StreamRecorder, model_size: str) -> "Transcriber | None":
        if not whisper.available():
            log.warning("faster-whisper not installed; running without live transcripts")
            return None
        return cls(recorder, model_size)

    def load_words(self, words: list[whisper.Word], timeline_start: float) -> None:
        """Use a transcript made up front (recordings) instead of transcribing live."""
        with self._lock:
            self._lines = [TranscriptLine(timeline_start + w.start, timeline_start + w.end, w.text) for w in words]

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
            words = whisper.transcribe_words(seg.path, self.model_size)
        except Exception as exc:  # a corrupt or deleted segment shouldn't kill the thread
            log.debug("Transcription failed for %s: %s", seg.path.name, exc)
            return
        lines = [TranscriptLine(seg.start + w.start, seg.start + w.end, w.text) for w in words]
        with self._lock:
            self._lines.extend(lines)
            cutoff = seg.start - self.recorder.buffer_seconds
            self._lines = [ln for ln in self._lines if ln.end >= cutoff]

    def lines_between(self, start: float, end: float) -> list[TranscriptLine]:
        with self._lock:
            return [ln for ln in self._lines if ln.end > start and ln.start < end]
