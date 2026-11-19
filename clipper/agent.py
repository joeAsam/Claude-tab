"""The Claude agent that decides what to clip and how to edit it.

Each review is a short, fresh conversation: Claude gets a summary of what just
happened, digs in with tools (chat, transcript, buffer info), cuts clips, then
edits each one: synced captions, sound effects at exact moments, and music.

Times are given to Claude as "seconds ago" relative to a fixed moment captured
when the review starts, so they don't drift while it thinks. Once a clip is
cut, edit times are in seconds from the start of that clip.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import anthropic
from anthropic import beta_tool
from typing_extensions import NotRequired

from . import whisper
from .captions import write_ass
from .chat import TwitchChat
from .clips import Clip, clip_basename, cut_clip
from .editor import SoundCue, probe, render
from .recorder import StreamRecorder
from .sounds import list_sounds
from .transcriber import Transcriber
from .whisper import Word

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLIPPER_MODEL", "claude-opus-5-5")
MAX_SOUND_EFFECTS = 15

SYSTEM_PROMPT = """\
You are a stream clipper and short-form video editor. You watch a live stream \
through tools, cut clips of the moments viewers would want to rewatch or share \
(big plays, funny reactions, fails, surprising reveals, rage, wholesome moments, \
hype, great stories), and edit them into finished Shorts/TikTok-style videos.

Finding and cutting clips:
- Start from the trigger you're given, then check chat activity and (if available) \
the transcript to understand what happened and how long the moment really is.
- Chat reacts a few seconds AFTER the moment. Start the clip early enough to \
include the setup, and end it once the reaction has played out.
- Every clip must be between {min_clip} and {max_clip} seconds. Pick the length \
the moment deserves: a quick reaction needs about {min_clip}s with its setup; a \
story, a full fight, or an escalating bit can use more, up to {max_clip}s. Don't \
pad a clip with dead air to make it longer.
- Be selective. A small bump in chat, routine gameplay, or a moment you already \
clipped is not worth a clip. Making no clip is a perfectly good outcome.
- Don't clip a range that overlaps one listed under "Recent clips".
- Titles should be short and punchy, like a Shorts/TikTok title, with no hashtags.

Editing (after create_clip, call finish_clip for every clip you cut):
- create_clip returns what was said and how chat reacted, with times in seconds \
from the start of the clip. Use those times to place sound effects exactly.
- Sound effects punctuate a moment: put them right on the punchline, the fail, \
the big play or the awkward pause, not on top of someone mid-sentence unless it's \
a deliberate comedic interruption. Use them sparingly: about one per 15-30 \
seconds at most, and none if the moment is serious or emotional. Pick by name: \
e.g. a fail suits sad_trombone, a joke suits rimshot_ba_dum_tss, a sudden stop \
suits record_scratch, an awkward silence suits crickets_awkward_silence, a reveal \
suits dramatic_dun_dun_dun, a huge play suits air_horn.
- Background music sets the mood: choose a track whose name fits the clip, or \
"none" when music would undercut it. It is automatically turned down while \
people talk.
- Leave captions on unless there's no speech worth reading.
{edit_note}
Finish with one sentence saying what you clipped and how you edited it, or why \
you clipped nothing.
"""


class SoundEffectCue(TypedDict):
    sound: str
    at_seconds: float
    volume: NotRequired[float]


@dataclass
class ClipAgentConfig:
    out_dir: Path
    assets_dir: Path
    vertical: bool = False
    min_clip: int = 30
    max_clip: int = 240
    captions: bool = True
    whisper_model: str = "base"
    sound_effects: bool = True
    music: bool = True


class ClipAgent:
    def __init__(
        self,
        recorder: StreamRecorder,
        chat: TwitchChat | None,
        transcriber: Transcriber | None,
        config: ClipAgentConfig,
    ) -> None:
        self.recorder = recorder
        self.chat = chat
        self.transcriber = transcriber
        self.cfg = config
        if config.captions and not whisper.available():
            log.warning("faster-whisper is not installed, so captions are off. pip install faster-whisper")
            config.captions = False
        self.clips: list[Clip] = []
        self._words: dict[str, list[Word]] = {}
        self.client = anthropic.Anthropic()
        self.system_prompt = SYSTEM_PROMPT.format(
            min_clip=config.min_clip,
            max_clip=config.max_clip,
            edit_note=self._edit_note(),
        )

    def _edit_note(self) -> str:
        notes = []
        if not self.cfg.sound_effects:
            notes.append("Sound effects are turned off for this run; pass an empty list.")
        if not self.cfg.music:
            notes.append('Music is turned off for this run; pass "none".')
        if not self.cfg.captions:
            notes.append("Captions are unavailable for this run.")
        return "".join(f"- {n}\n" for n in notes)

    # ---- one review ------------------------------------------------------

    def review(self, trigger: str) -> None:
        now = time.time()

        def ts(seconds_ago: float) -> float:
            return now - seconds_ago

        def ago(t: float) -> int:
            return round(now - t)

        @beta_tool
        def get_buffer_info() -> str:
            """Return how far back the recorded video goes and which inputs are available."""
            rng = self.recorder.available_range()
            if not rng:
                return "The buffer is empty; nothing has been recorded yet."
            return json.dumps({
                "oldest_seconds_ago": ago(rng[0]),
                "newest_seconds_ago": ago(rng[1]),
                "transcript_available": self.transcriber is not None,
                "chat_available": self.chat is not None,
            })

        @beta_tool
        def get_chat_activity(from_seconds_ago: int, to_seconds_ago: int = 0, bucket_seconds: int = 10) -> str:
            """Chat message counts per time bucket, with the most repeated messages in each.

            Args:
                from_seconds_ago: Start of the window, in seconds before now.
                to_seconds_ago: End of the window, in seconds before now (0 = now).
                bucket_seconds: Width of each bucket in seconds.
            """
            if not self.chat:
                return "No chat is connected for this stream."
            rows = self.chat.buckets(ts(from_seconds_ago), ts(to_seconds_ago), max(bucket_seconds, 1))
            for r in rows:
                r["seconds_ago"] = ago(r.pop("start"))
            return json.dumps(rows)

        @beta_tool
        def get_chat_messages(from_seconds_ago: int, to_seconds_ago: int = 0, limit: int = 80) -> str:
            """Raw chat messages in a time window (sampled evenly if there are more than `limit`).

            Args:
                from_seconds_ago: Start of the window, in seconds before now.
                to_seconds_ago: End of the window, in seconds before now (0 = now).
                limit: Maximum number of messages to return.
            """
            if not self.chat:
                return "No chat is connected for this stream."
            msgs = self.chat.messages_between(ts(from_seconds_ago), ts(to_seconds_ago))
            if len(msgs) > limit > 0:
                step = len(msgs) / limit
                msgs = [msgs[int(i * step)] for i in range(limit)]
            return "\n".join(f"[-{ago(m.ts)}s] {m.user}: {m.text}" for m in msgs) or "(no messages)"

        @beta_tool
        def get_transcript(from_seconds_ago: int, to_seconds_ago: int = 0) -> str:
            """What the streamer said in a time window (live speech-to-text).

            Args:
                from_seconds_ago: Start of the window, in seconds before now.
                to_seconds_ago: End of the window, in seconds before now (0 = now).
            """
            if not self.transcriber:
                return "Live transcription is not enabled."
            lines = self.transcriber.lines_between(ts(from_seconds_ago), ts(to_seconds_ago))
            words = [Word(ln.start, ln.end, ln.text) for ln in lines]
            grouped = _group_words(words)
            return "\n".join(
                f"[-{ago(g[0].start)}s] {' '.join(w.text for w in g)}" for g in grouped
            ) or "(no speech found)"

        @beta_tool
        def create_clip(start_seconds_ago: int, end_seconds_ago: int, title: str, reason: str) -> str:
            """Cut a clip from the recorded buffer. Returns the clip's id, what was said and how
            chat reacted (times in seconds from the start of the clip), plus the sound effects and
            music you can use. Call finish_clip next to edit and save it.

            Args:
                start_seconds_ago: Where the clip starts, in seconds before now.
                end_seconds_ago: Where the clip ends, in seconds before now. Must be smaller than start.
                title: Short, punchy title for the clip.
                reason: One sentence on why this moment is worth clipping.
            """
            start, end = ts(start_seconds_ago), ts(end_seconds_ago)
            length = end - start
            if length < self.cfg.min_clip or length > self.cfg.max_clip:
                return (f"Error: the clip is {length:.0f}s; it must be between "
                        f"{self.cfg.min_clip} and {self.cfg.max_clip} seconds.")
            for c in self.clips:
                if c.start < end and start < c.end:
                    return f"Error: overlaps an existing clip ('{c.title}')."
            rng = self.recorder.available_range()
            if not rng or start < rng[0] or end > rng[1] + 1:
                return "Error: that range isn't in the buffer. Call get_buffer_info for the available range."

            clip_id = f"c{len(self.clips) + 1}"
            name = clip_basename(start, title, self.cfg.vertical)
            raw = self.cfg.out_dir / ".work" / f"{name}.raw.mp4"
            try:
                cut_clip(self.recorder, start, end, raw, self.cfg.vertical)
            except Exception as exc:
                return f"Error cutting clip: {exc}"
            clip = Clip(clip_id, start, end, title, reason, raw)
            self.clips.append(clip)

            words: list[Word] = []
            if self.cfg.captions:
                try:
                    words = whisper.transcribe_words(raw, self.cfg.whisper_model)
                except Exception as exc:
                    log.warning("Transcribing %s failed: %s", raw.name, exc)
            self._words[clip_id] = words
            log.info("Cut %s (%.0fs): %s", clip_id, length, title)

            return json.dumps({
                "clip_id": clip_id,
                "duration_seconds": round(length, 1),
                "speech": _speech_lines(words) if self.cfg.captions else "captions unavailable",
                "chat_reactions": self._chat_in_clip(start, end),
                "sound_effects_available": list(list_sounds(self.cfg.assets_dir / "sfx"))
                if self.cfg.sound_effects else [],
                "music_available": list(list_sounds(self.cfg.assets_dir / "music"))
                if self.cfg.music else [],
                "next_step": "Call finish_clip with this clip_id to add captions, sound effects and music.",
            })

        @beta_tool
        def finish_clip(
            clip_id: str,
            sound_effects: list[SoundEffectCue] = [],
            music: str = "none",
            music_volume: float = 0.15,
            captions: bool = True,
        ) -> str:
            """Edit a clip made by create_clip and save the finished video.

            Args:
                clip_id: The id returned by create_clip.
                sound_effects: Sound effects to add. Each has `sound` (a name from sound_effects_available), `at_seconds` (seconds from the start of the clip) and optional `volume` (0.1-1.5, default 0.8).
                music: A track name from music_available, or "none".
                music_volume: Music loudness from 0.05 (subtle) to 0.4 (loud); 0.15 suits most clips.
                captions: Whether to burn in captions synced to the speech.
            """
            clip = next((c for c in self.clips if c.id == clip_id), None)
            if not clip:
                return f"Error: no clip with id {clip_id}."
            if clip.path:
                return f"Error: {clip_id} is already finished ({clip.path.name})."
            try:
                path = self._finish(clip, sound_effects, music, music_volume, captions)
            except ValueError as exc:
                return f"Error: {exc}"
            except Exception as exc:
                log.exception("Rendering %s failed", clip_id)
                return f"Error rendering the clip: {exc}"
            return f"Saved {path.name} ({clip.duration:.0f}s) with {json.dumps(clip.edits)}."

        prompt = self._briefing(trigger, now, ago)
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            tools=[get_buffer_info, get_chat_activity, get_chat_messages, get_transcript, create_clip, finish_clip],
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium"},
            # If a request is declined by a safety classifier, retry it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            max_iterations=20,
        )
        try:
            final = runner.until_done()
        finally:
            self._finish_leftovers()
        if final.stop_reason == "refusal":
            log.warning("Review declined by the model.")
            return
        summary = " ".join(b.text for b in final.content if b.type == "text").strip()
        log.info("Review done: %s", summary or "(no summary)")

    # ---- editing ---------------------------------------------------------

    def _finish(self, clip: Clip, cues: list[SoundEffectCue], music: str, music_volume: float, captions: bool) -> Path:
        sfx_lib = list_sounds(self.cfg.assets_dir / "sfx") if self.cfg.sound_effects else {}
        music_lib = list_sounds(self.cfg.assets_dir / "music") if self.cfg.music else {}

        if len(cues) > MAX_SOUND_EFFECTS:
            raise ValueError(f"at most {MAX_SOUND_EFFECTS} sound effects per clip.")
        sound_cues = []
        for cue in cues:
            if cue["sound"] not in sfx_lib:
                raise ValueError(f"unknown sound effect '{cue['sound']}'. Available: {', '.join(sfx_lib) or 'none'}")
            if not 0 <= cue["at_seconds"] < clip.duration:
                raise ValueError(f"at_seconds {cue['at_seconds']} is outside the clip (0-{clip.duration:.0f}s).")
            volume = min(max(cue.get("volume", 0.8), 0.05), 1.5)
            sound_cues.append(SoundCue(sfx_lib[cue["sound"]], cue["at_seconds"], volume))

        music_path = None
        if music and music != "none":
            if music not in music_lib:
                raise ValueError(f"unknown music '{music}'. Available: {', '.join(music_lib) or 'none'}")
            music_path = music_lib[music]

        captions_file = None
        words = self._words.get(clip.id, [])
        if captions and self.cfg.captions and words:
            info = probe(clip.raw_path)
            captions_file = write_ass(
                words, info["width"], info["height"], info["duration"],
                clip.raw_path.with_suffix(".ass"),
            )

        out = self.cfg.out_dir / clip.raw_path.name.replace(".raw.mp4", ".mp4")
        render(clip.raw_path, out, captions_file, sound_cues, music_path, min(max(music_volume, 0.02), 0.6))

        clip.path = out
        clip.edits = {
            "captions": captions_file is not None,
            "sound_effects": [f"{c['sound']}@{c['at_seconds']:g}s" for c in cues],
            "music": music if music_path else "none",
        }
        self._log_clip(clip)
        clip.raw_path.unlink(missing_ok=True)
        if captions_file:
            captions_file.unlink(missing_ok=True)
        log.info("Finished %s -> %s %s", clip.id, out, clip.edits)
        return out

    def _finish_leftovers(self) -> None:
        """Clips Claude cut but never finished still get saved, with captions only."""
        for clip in self.clips:
            if clip.path is None and clip.raw_path.exists():
                log.info("Finishing %s with default edits", clip.id)
                try:
                    self._finish(clip, [], "none", 0.15, True)
                except Exception:
                    log.exception("Could not finish %s; raw cut kept at %s", clip.id, clip.raw_path)

    # ---- helpers ---------------------------------------------------------

    def _chat_in_clip(self, start: float, end: float) -> list[dict]:
        if not self.chat:
            return []
        rows = self.chat.buckets(start, end, 5)
        return [
            {"at_seconds": round(r["start"] - start), "messages": r["count"], "top": r["top"]}
            for r in rows if r["count"]
        ]

    def _briefing(self, trigger: str, now: float, ago) -> str:
        parts = [f"Trigger: {trigger}"]
        if self.chat:
            recent = self.chat.buckets(now - 300, now, 15)
            parts.append(
                "Chat messages per 15s over the last 5 minutes (oldest first): "
                + ", ".join(str(r["count"]) for r in recent)
            )
        recent_clips = [c for c in self.clips if now - c.end < 1800]
        if recent_clips:
            parts.append("Recent clips:\n" + "\n".join(
                f"- '{c.title}' from -{ago(c.start)}s to -{ago(c.end)}s" for c in recent_clips
            ))
        else:
            parts.append("Recent clips: none")
        parts.append("Decide whether anything here is worth clipping. If so, cut it and edit it.")
        return "\n\n".join(parts)

    def _log_clip(self, clip: Clip) -> None:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        row = asdict(clip) | {"path": str(clip.path), "raw_path": None}
        with open(self.cfg.out_dir / "clips.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def _group_words(words: list[Word], pause: float = 0.7, max_words: int = 14) -> list[list[Word]]:
    """Group words into short phrases, breaking on pauses."""
    groups: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current and (w.start - current[-1].end > pause or len(current) >= max_words):
            groups.append(current)
            current = []
        current.append(w)
    if current:
        groups.append(current)
    return groups


def _speech_lines(words: list[Word]) -> list[str]:
    """Timed lines (seconds from clip start) Claude can read and place effects against."""
    return [f"[{g[0].start:.1f}s-{g[-1].end:.1f}s] {' '.join(w.text for w in g)}" for g in _group_words(words)]
