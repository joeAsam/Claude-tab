"""Spots potentially clip-worthy moments from several live signals at once.

Every finished video segment is analyzed for:
  - audio loudness   (screams, rage, hype, jump scares, laughing fits)
  - visual change    (explosions, deaths, jump scares, chaotic action, scene cuts)
and combined with:
  - chat speed and what KIND of reaction chat is having (laughing, hype, shock,
    rage/"L", sad, people asking to clip it)
  - what the streamer is saying (exclamations, swearing, "no way", "let's go")

Each signal is compared with the stream's own recent normal, so a quiet
chess stream and a screaming horror stream are both judged fairly. The result
is a per-window "interest score" with human-readable reasons, used to decide
when to wake the Claude agent and shown to it as a timeline.
"""

from __future__ import annotations

import logging
import re
import statistics
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .chat import TwitchChat
from .recorder import Segment, StreamRecorder
from .transcriber import Transcriber

log = logging.getLogger(__name__)

# ---- chat reaction lexicon ---------------------------------------------

REACTIONS: dict[str, set[str]] = {
    "laughing": {"lul", "lol", "lmao", "lmfao", "kekw", "omegalul", "icant", "haha", "hahaha", "xd", "kek",
                 "lulw", "pepelaugh", "😂", "🤣", "dead", "im dead", "crying"},
    "hype": {"pog", "pogchamp", "poggers", "pogu", "w", "lets go", "letsgo", "let's go", "🔥", "gg", "clutch",
             "insane", "goated", "ezclap", "peepoclap", "sheesh", "huge", "cracked"},
    "shock": {"???", "what", "wtf", "no way", "noway", "omg", "monkas", "holy", "jesus", "😱", "😳", "huh",
              "wait", "bro what", "what just happened", "nani", "o7"},
    "rage/fail": {"l", "rip", "💀", "malding", "tilted", "copium", "ff", "unlucky", "throw", "threw", "sold",
                  "skill issue", "washed", "bozo"},
    "sad/wholesome": {"sadge", "f", "biblethump", "pepehands", "😢", "🥺", "❤️", "<3", "wholesome", "aww"},
    "clip it": {"clip", "clip it", "clip that", "clipped", "clipper", "clip this"},
}

# What the streamer says when something big is happening.
SPEECH_CUES = re.compile(
    r"\b(oh my god|omg|no way|what the|let'?s go+|are you kidding|you'?re kidding|holy|no no no|"
    r"yes yes|i can'?t believe|bro+|dude|wait what|what\?|how\??|insane|clip (it|that)|"
    r"fuck\w*|shit\w*|damn|hell)\b",
    re.IGNORECASE,
)

WINDOW = 5  # seconds per scored window


def classify_message(text: str) -> set[str]:
    t = text.lower().strip()
    tokens = set(re.findall(r"[\w']+|[^\w\s]", t))
    found = set()
    for kind, words in REACTIONS.items():
        for w in words:
            if (" " in w and w in t) or w in tokens or (not w.isalnum() and w in t):
                found.add(kind)
                break
    if re.fullmatch(r"\?{2,}", t):
        found.add("shock")
    return found


@dataclass
class WindowScore:
    start: float
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)


class SignalMonitor:
    def __init__(self, recorder: StreamRecorder, chat: TwitchChat | None, transcriber: Transcriber | None) -> None:
        self.recorder = recorder
        self.chat = chat
        self.transcriber = transcriber
        self._loud: dict[float, float] = {}   # half-second timestamp -> RMS dB
        self._scene: dict[float, float] = {}  # half-second timestamp -> scene change score (0-1)
        self._chat_scroll: dict[float, float] = {}  # half-second timestamp -> on-screen chat movement (recordings)
        self.fixed_baseline: dict | None = None  # recordings: one "normal" for the whole file
        self._done: set[str] = set()
        self._analyzed_until = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ---- background analysis -------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="signals").start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def analyzed_until(self) -> float:
        return self._analyzed_until

    def _run(self) -> None:
        while not self._stop.wait(1):
            for seg in self.recorder.segments():
                if seg.path.name in self._done or self._stop.is_set():
                    continue
                try:
                    self._analyze(seg)
                except Exception as exc:  # deleted or corrupt segment: skip it
                    log.debug("Signal analysis failed for %s: %s", seg.path.name, exc)
                self._done.add(seg.path.name)
                self._analyzed_until = max(self._analyzed_until, seg.end)
            self._forget_old()

    def analyze_all(self) -> None:
        """Analyze every segment now (for recordings, where all footage already exists)."""
        segs = self.recorder.segments()
        for i, seg in enumerate(segs, 1):
            if seg.path.name not in self._done:
                try:
                    self._analyze(seg)
                except Exception as exc:
                    log.debug("Signal analysis failed for %s: %s", seg.path.name, exc)
                self._done.add(seg.path.name)
                self._analyzed_until = max(self._analyzed_until, seg.end)
            if i % max(len(segs) // 10, 1) == 0:
                log.info("Analyzing audio and video: %d%%", round(i / len(segs) * 100))

    def load_chat_scroll(self, video: Path, crop_filter: str, timeline_start: float) -> None:
        """Measure how much the on-screen chat panel moves (a stand-in for chat speed when
        chat is only visible in a screen recording, not readable as text)."""
        log.info("Measuring on-screen chat activity...")
        series = _metadata_series_path(
            ["-an", "-vf", f"{crop_filter},fps=2,scale=200:-2,select='gte(scene\\,0)',"
             "metadata=mode=print:key=lavfi.scene_score:file=-"],
            video, "lavfi.scene_score", timeout=None,
        )
        with self._lock:
            for t, v in series:
                self._chat_scroll[round((timeline_start + t) * 2) / 2] = v

    def _analyze(self, seg: Segment) -> None:
        loud = _metadata_series(
            ["-vn", "-af", "aresample=16000,asetnsamples=n=8000,astats=metadata=1:reset=1,"
             "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-"],
            seg, "lavfi.astats.Overall.RMS_level",
        )
        scene = _metadata_series(
            ["-an", "-vf", "fps=2,scale=160:-2,select='gte(scene\\,0)',"
             "metadata=mode=print:key=lavfi.scene_score:file=-"],
            seg, "lavfi.scene_score",
        )
        with self._lock:
            for t, v in loud:
                if v > -120:  # -inf means digital silence
                    self._loud[round((seg.start + t) * 2) / 2] = v
            for t, v in scene:
                self._scene[round((seg.start + t) * 2) / 2] = v

    def _forget_old(self) -> None:
        cutoff = time.time() - max(self.recorder.buffer_seconds, 900)
        with self._lock:
            for series in (self._loud, self._scene, self._chat_scroll):
                for k in [k for k in series if k < cutoff]:
                    del series[k]

    # ---- features --------------------------------------------------------

    def _values(self, series: dict[float, float], start: float, end: float) -> list[float]:
        with self._lock:
            return [v for k, v in series.items() if start <= k < end]

    def features(self, start: float, end: float) -> dict:
        """Raw signal values for one stretch of the stream."""
        f: dict = {}
        loud = self._values(self._loud, start, end)
        if loud:
            f["loudness_db"] = round(max(loud), 1)
        scene = self._values(self._scene, start, end)
        if scene:
            f["visual_change"] = round(max(scene), 2)
        scroll = self._values(self._chat_scroll, start, end)
        if scroll:
            f["chat_scroll"] = round(statistics.mean(scroll), 3)
        if self.chat:
            msgs = self.chat.messages_between(start, end)
            f["chat_msgs"] = len(msgs)
            f["chatters"] = len({m.user for m in msgs})
            kinds = Counter(k for m in msgs for k in classify_message(m.text))
            if kinds:
                f["chat_reactions"] = dict(kinds.most_common())
        if self.transcriber:
            text = " ".join(ln.text for ln in self.transcriber.lines_between(start, end))
            cues = [m.group(0).lower() for m in SPEECH_CUES.finditer(text)]
            if cues:
                f["speech_cues"] = cues
        return f

    def baseline(self, now: float, lookback: int = 600) -> dict:
        """What 'normal' looks like on this stream lately (excluding the last minute)."""
        if self.fixed_baseline is not None:
            return self.fixed_baseline
        return self.baseline_between(now - lookback, now - 60)

    def use_whole_range_as_baseline(self, start: float, end: float) -> None:
        self.fixed_baseline = self.baseline_between(start, end)

    def baseline_between(self, start: float, end: float) -> dict:
        b: dict = {}
        loud = self._values(self._loud, start, end)
        if len(loud) >= 20:
            b["loudness_db"] = statistics.median(loud)
            b["loudness_spread"] = max(statistics.pstdev(loud), 2.0)
        scene = self._values(self._scene, start, end)
        if len(scene) >= 20:
            b["visual_change"] = statistics.quantiles(scene, n=10)[-1]  # 90th percentile
        scroll = self._values(self._chat_scroll, start, end)
        if len(scroll) >= 20:
            b["chat_scroll"] = max(statistics.median(scroll), 0.003)
        if self.chat:
            # Only count time we were actually listening, or early chat looks artificially quiet.
            listen_start = max(start, self.chat.listening_since or end)
            span = end - listen_start
            if span >= 60:
                b["chat_per_window"] = len(self.chat.messages_between(listen_start, end)) / span * WINDOW
        return b

    def score_window(self, start: float, baseline: dict) -> WindowScore:
        f = self.features(start, start + WINDOW)
        ws = WindowScore(start, features=f)

        # Chat: speed relative to normal, and what kind of reaction it is.
        if "chat_msgs" in f and "chat_per_window" in baseline:
            normal = max(baseline["chat_per_window"], 0.5)
            ratio = f["chat_msgs"] / normal
            if ratio >= 1.8 and f["chat_msgs"] >= 4:
                ws.score += min((ratio - 1) * 0.8, 4.0)
                ws.reasons.append(f"chat {ratio:.1f}x busier than normal")
            reactions = f.get("chat_reactions", {})
            mood = [k for k, n in reactions.items() if k != "clip it" and n >= max(3, f["chat_msgs"] * 0.2)]
            if mood:
                ws.score += 0.5
                ws.reasons.append("chat is " + " + ".join(mood))
            if reactions.get("clip it", 0) >= 2:
                ws.score += 1.5
                ws.reasons.append(f"{reactions['clip it']} viewers asking to clip it")

        # On-screen chat (recordings): the chat panel scrolling much faster than usual.
        if "chat_scroll" in f and "chat_scroll" in baseline:
            ratio = f["chat_scroll"] / baseline["chat_scroll"]
            if ratio >= 2.0:
                ws.score += min((ratio - 1) * 0.7, 3.0)
                ws.reasons.append(f"on-screen chat scrolling {ratio:.1f}x faster than normal")

        # Audio: sudden loudness = yelling, screaming, laughing hard, explosions.
        if "loudness_db" in f and "loudness_db" in baseline:
            jump = f["loudness_db"] - baseline["loudness_db"]
            z = jump / baseline["loudness_spread"]
            if jump >= 6 and z >= 2:
                ws.score += min(1.0 + (jump - 6) / 3, 3.5)
                ws.reasons.append(f"audio {jump:.0f} dB louder than normal (yelling/screaming/explosion?)")

        # Video: big visual changes = action, deaths, jump scares, cuts.
        if "visual_change" in f and "visual_change" in baseline:
            # Relative to how much this stream's picture normally changes (a fast shooter vs a chess board).
            if f["visual_change"] >= max(0.08, baseline["visual_change"] * 3):
                ws.score += 1.0
                ws.reasons.append("sudden big change on screen")

        # Speech: the streamer reacting out loud.
        if f.get("speech_cues"):
            ws.score += min(0.75 * len(f["speech_cues"]), 2.0)
            ws.reasons.append("streamer said: " + ", ".join(f'"{c}"' for c in f["speech_cues"][:4]))

        ws.score = round(ws.score, 2)
        return ws

    def timeline(self, start: float, end: float, now: float | None = None) -> list[WindowScore]:
        base = self.baseline(now or time.time())
        out, t = [], start
        while t < end:
            out.append(self.score_window(t, base))
            t += WINDOW
        return out

    def hottest(self, start: float, end: float) -> WindowScore | None:
        """The highest-scoring window in a range, if any signal fired."""
        best = max(self.timeline(start, end), key=lambda w: w.score, default=None)
        return best if best and best.score > 0 else None


def _metadata_series(args: list[str], seg: Segment, key: str) -> list[tuple[float, float]]:
    return _metadata_series_path(args, seg.path, key)


def _metadata_series_path(
    args: list[str], path: Path, key: str, timeout: float | None = 60
) -> list[tuple[float, float]]:
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-loglevel", "error", "-i", str(path), *args, "-f", "null", "-"],
        capture_output=True, text=True, timeout=timeout,
    ).stdout
    series, t = [], None
    for line in out.splitlines():
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([\d.]+)", line)
            t = float(m.group(1)) if m else None
        elif line.startswith(key + "=") and t is not None:
            try:
                series.append((t, float(line.split("=", 1)[1])))
            except ValueError:
                pass
    return series
