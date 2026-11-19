"""Shared faster-whisper model and word-level transcription."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_models: dict[str, object] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def get_model(size: str = "base"):
    """Load a whisper model once and share it between threads."""
    from faster_whisper import WhisperModel

    with _lock:
        if size not in _models:
            log.info("Loading whisper model '%s' (first run downloads it)", size)
            _models[size] = WhisperModel(size, device="auto", compute_type="int8")
        return _models[size]


def transcribe_words(media: Path, size: str = "base") -> list[Word]:
    """Speech in a media file as individual words with start/end times in seconds."""
    model = get_model(size)
    with _lock:  # one transcription at a time; the model isn't built for concurrent calls
        segments, _ = model.transcribe(str(media), vad_filter=True, word_timestamps=True)
        words = [
            Word(w.start, w.end, w.word.strip())
            for seg in segments
            for w in (seg.words or [])
            if w.word.strip()
        ]
    return words
