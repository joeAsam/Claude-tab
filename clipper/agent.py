"""The Claude agent that finds, judges, cuts and edits clips.

Each review is a short, fresh conversation: Claude gets a briefing of what the
signal detector noticed, investigates with tools (signal timeline, chat,
transcript, and actual frames from the video), decides whether there is a
genuinely clip-worthy moment, cuts it with clean boundaries, and edits it
(hook title, synced captions, sound effects, music).

Times are given to Claude as "seconds ago" relative to a fixed moment captured
when the review starts, so they don't drift while it thinks. Once a clip is
cut, edit times are in seconds from the start of that clip.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

import anthropic
from anthropic import beta_tool
from typing_extensions import NotRequired

from . import whisper
from .captions import write_ass
from .chat import TwitchChat
from .clips import Clip, cut_clip, slugify
from .editor import SoundCue, probe, render
from .filesource import FileSource
from .frames import grab_frame, image_block
from .recorder import StreamRecorder
from .signals import SignalMonitor
from .sounds import list_sounds
from .transcriber import Transcriber
from .whisper import Word

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLIPPER_MODEL", "claude-opus-5-5")
MAX_SOUND_EFFECTS = 15
MAX_FRAMES = 8
SNAP_PAD = 6.0  # seconds of extra footage examined around a cut to find clean boundaries

Category = Literal[
    "funny", "rage", "hype/clutch", "fail", "shock/jump scare", "wholesome",
    "story/hot take", "intriguing/mystery", "unique/weird", "skill/impressive", "drama/conflict",
]

SYSTEM_PROMPT = """\
You are an expert stream clipper and short-form video editor. You watch a live \
stream through tools and find the moments that would make people stop scrolling, \
watch to the end and share: funny, rage, hype, fails, shocking, wholesome, \
intriguing, unique, impressive. Then you cut and edit them into finished \
Shorts/TikTok-style videos.

{source_note}

## What makes a moment worth clipping
Rate every candidate from 1 to 10 on these, then give an overall virality score:
- Hook: would the first 2-3 seconds make someone stop scrolling?
- Standalone: does it make sense to someone who has never seen this streamer?
- Payoff: is there a clear peak: a punchline, a reveal, an explosion of emotion, \
an unbelievable play?
- Emotion: does it make you laugh, gasp, cringe, feel hyped or feel something?
- Shareability: would someone send it to a friend or comment on it?
Only clip moments with an overall score of {min_score} or higher. Routine \
gameplay, small chat bumps, inside jokes that need context, and things you have \
already clipped don't qualify. Making no clip is a perfectly good outcome.

## Moment types: what they look like and how to cut them
- funny: chat laughing (KEKW, LMAO, 😂), streamer laughing or a sudden loud \
laugh. Include the setup, the punchline and the laugh; end shortly after the laugh.
- rage: the streamer gets much louder, swears, slams; chat says L, 💀, malding. \
Include what caused it; end at the peak of the outburst or the aftermath.
- hype/clutch: loud yelling, "let's go", chat spamming POG/W/🔥. Include the \
tension before the play so the payoff lands; end on the celebration.
- fail: something goes badly wrong (death, misplay, embarrassing moment); chat \
says L, RIP, 💀, "???". Start before the confident lead-in, end on the reaction.
- shock/jump scare: a sudden loud spike and a big change on screen; chat says \
"???", WTF, monkaS. Start 5-10 seconds before so the calm is felt.
- wholesome: kind moments, donations that move the streamer, sweet interactions; \
chat ❤️, <3, aww. Usually needs more context; keep the whole exchange.
- story/hot take: the streamer tells a gripping story or says something bold or \
controversial. Often quiet on audio and video: the transcript is the signal. Start \
at the first sentence of the story or take and end right after its punchline or \
point. These can run long.
- intriguing/mystery, unique/weird, skill/impressive, drama/conflict: something \
viewers have never seen, a mystery that makes you want the answer, a rare skill \
display, a heated argument.

## How to investigate
1. Read the briefing: it lists the moments the signal detector flagged and why.
2. Use get_signals around a flagged moment to see how it built up and faded.
3. Read chat (get_chat_messages) and the transcript (get_transcript) to \
understand WHAT happened.
4. Look at the video with view_frames, especially to confirm visual moments \
(deaths, jump scares, wins, reactions) and to judge whether it is \
understandable without context.
5. Decide. If it qualifies, choose the boundaries carefully.

## Cutting clips properly
- Every clip must be between {min_clip} and {max_clip} seconds. Use the length \
the moment deserves, never pad with dead air.
- Open as close to the action as possible while still including the setup a \
newcomer needs. The first seconds must hook.
- Chat reacts a few seconds AFTER the moment itself; the screen and audio show \
when it really happened.
- End once the reaction has played out, not mid-laugh or mid-sentence. Cuts are \
nudged automatically to the nearest pause in speech.
- Don't clip a range that overlaps one listed under "Recent clips".

## Titles and hooks
- title: short file title.
- hook: the text shown at the top of the video for the whole clip, 3-8 words. It \
must make people want to watch without spoiling the payoff: tease, ask, or \
provoke (e.g. "He did NOT expect this", "Worst timing ever", "Wait for the \
ending"). No emoji, no hashtags.

## Editing (call finish_clip for every clip you cut)
- create_clip returns what was said and how chat reacted, with times in seconds \
from the start of the clip. Use those times to place sound effects exactly.
- Sound effects punctuate a moment: put them right on the punchline, the fail, \
the big play or the awkward pause, not over someone mid-sentence unless it's a \
deliberate comedic interruption. Use about one per 15-30 seconds at most, and \
none if the moment is serious, emotional or tense. Pick by name: a fail suits \
sad_trombone, a joke suits rimshot_ba_dum_tss, a sudden stop suits \
record_scratch, an awkward silence suits crickets_awkward_silence, a reveal suits \
dramatic_dun_dun_dun, a huge play suits air_horn.
- Background music sets the mood: choose a track whose name fits the clip, or \
"none" when music would undercut it. It is automatically turned down under speech.
- Leave captions on unless there's no speech worth reading.
{edit_note}
Finish with one sentence: what you clipped (type, score) and how you edited it, \
or why nothing qualified.
"""


SOURCE_NOTES = {
    "live": (
        "You are watching a LIVE stream. All times you give and receive are in seconds "
        "before now: 30 means 30 seconds ago."
    ),
    "recording": (
        "You are working on a RECORDING of a stream (often a screen recording, already cropped "
        "to the stream player). All times you give and receive are in seconds from the start of "
        "the recording. Chat isn't available as text: the \"on-screen chat scrolling\" signal shows "
        "how fast chat was moving, and view_chat shows you the chat panel itself so you can read "
        "how viewers reacted (if the recording shows chat)."
    ),
}


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
    min_score: int = 7
    captions: bool = True
    whisper_model: str = "base"
    sound_effects: bool = True
    music: bool = True
    hook: bool = True
    fixed_hook: str | None = None  # same hook text on every clip, instead of Claude writing one


class Timeline:
    """Converts between the times Claude uses and real (timeline) timestamps."""

    def __init__(self, recording: FileSource | None, now: float) -> None:
        self.recording = recording
        self.now = now
        if recording:
            self.origin = recording.virtual_start
            self.unit = "seconds from the start of the recording"
            self.default_to = "omit for the end of the recording"
        else:
            self.origin = now
            self.unit = "seconds before now (30 = 30 seconds ago)"
            self.default_to = "omit for now"

    def to_abs(self, t: float | None) -> float:
        if t is None:
            return self.recording.available_range()[1] if self.recording else self.now
        return self.origin + t if self.recording else self.now - t

    def rel(self, t: float) -> int:
        return round(t - self.origin) if self.recording else round(self.now - t)

    def label(self, t: float) -> str:
        if self.recording:
            s = self.rel(t)
            return f"{s // 60}:{s % 60:02d} ({s}s)"
        return f"-{self.rel(t)}s"


def _tool(fn, tl: Timeline):
    """Fill the time-unit wording into a tool's docstring, then register it as a tool."""
    fn.__doc__ = fn.__doc__.format(unit=tl.unit, default_to=tl.default_to)
    return beta_tool(fn)


class ClipAgent:
    def __init__(
        self,
        recorder: StreamRecorder,
        chat: TwitchChat | None,
        transcriber: Transcriber | None,
        signals: SignalMonitor | None,
        config: ClipAgentConfig,
        recording: FileSource | None = None,
    ) -> None:
        self.recording = recording
        self.recorder = recorder
        self.chat = chat
        self.transcriber = transcriber
        self.signals = signals
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
            min_score=config.min_score,
            source_note=SOURCE_NOTES["recording" if recording else "live"],
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
        if self.cfg.fixed_hook:
            notes.append(f'Every clip uses the fixed hook "{self.cfg.fixed_hook}"; your hook text is ignored.')
        elif not self.cfg.hook:
            notes.append("Hook titles are turned off for this run; still fill in the field.")
        return "".join(f"- {n}\n" for n in notes)

    # ---- one review ------------------------------------------------------

    def review(self, trigger: str, focus: tuple[float, float] | None = None) -> None:
        """One investigation. `focus` is the timeline range the briefing should describe
        (default: the last 5 minutes of a live stream)."""
        tl = Timeline(self.recording, time.time())
        T, L = tl.to_abs, tl.label

        def get_buffer_info() -> str:
            """Return the time range of the footage you can clip from and which inputs are available."""
            rng = self.recorder.available_range()
            if not rng:
                return "No footage is available yet."
            return json.dumps({
                "footage_from": L(rng[0]),
                "footage_to": L(rng[1]),
                "times_are_in": tl.unit,
                "chat_text_available": self.chat is not None,
                "chat_panel_images_available": bool(self.recording and self.recording.chat),
                "transcript_available": self.transcriber is not None,
            })

        def get_signals(from_t: float, to_t: float | None = None) -> str:
            """Timeline of everything the signal detector measured, in 5-second windows: chat speed and
            reaction types (laughing, hype, shock, rage/fail, sad/wholesome, clip it), on-screen chat
            scrolling, audio loudness, visual change on screen, and what the streamer exclaimed. Each
            window has an interest score and reasons. Use it to see how a moment built up, peaked and faded.

            Args:
                from_t: Start of the range, in {unit}.
                to_t: End of the range, in {unit}; {default_to}.
            """
            if not self.signals:
                return "The signal detector is not running."
            a, b = sorted((T(from_t), T(to_t)))
            rows = [{"at": L(w.start), "score": w.score, "reasons": w.reasons, **w.features}
                    for w in self.signals.timeline(a, b, tl.now)]
            base = self.signals.baseline(tl.now)
            return json.dumps({"normal_for_this_stream": {k: round(v, 3) for k, v in base.items()},
                               "windows": rows})

        def get_chat_activity(from_t: float, to_t: float | None = None, bucket_seconds: int = 10) -> str:
            """Chat message counts per time bucket, with the most repeated messages in each.

            Args:
                from_t: Start of the range, in {unit}.
                to_t: End of the range, in {unit}; {default_to}.
                bucket_seconds: Width of each bucket in seconds.
            """
            if not self.chat:
                return "Chat isn't available as text for this stream."
            a, b = sorted((T(from_t), T(to_t)))
            rows = self.chat.buckets(a, b, max(bucket_seconds, 1))
            for r in rows:
                r["at"] = L(r.pop("start"))
            return json.dumps(rows)

        def get_chat_messages(from_t: float, to_t: float | None = None, limit: int = 80) -> str:
            """Raw chat messages in a time range (sampled evenly if there are more than `limit`).

            Args:
                from_t: Start of the range, in {unit}.
                to_t: End of the range, in {unit}; {default_to}.
                limit: Maximum number of messages to return.
            """
            if not self.chat:
                return "Chat isn't available as text for this stream."
            a, b = sorted((T(from_t), T(to_t)))
            msgs = self.chat.messages_between(a, b)
            if len(msgs) > limit > 0:
                step = len(msgs) / limit
                msgs = [msgs[int(i * step)] for i in range(limit)]
            return "\n".join(f"[{L(m.ts)}] {m.user}: {m.text}" for m in msgs) or "(no messages)"

        def get_transcript(from_t: float, to_t: float | None = None) -> str:
            """What the streamer said in a time range (speech-to-text).

            Args:
                from_t: Start of the range, in {unit}.
                to_t: End of the range, in {unit}; {default_to}.
            """
            if not self.transcriber:
                return "Transcription is not enabled."
            a, b = sorted((T(from_t), T(to_t)))
            words = [Word(ln.start, ln.end, ln.text) for ln in self.transcriber.lines_between(a, b)]
            return "\n".join(
                f"[{L(g[0].start)}] {' '.join(w.text for w in g)}" for g in _group_words(words)
            ) or "(no speech found)"

        def view_frames(from_t: float, to_t: float, count: int = 4):
            """Look at the video: returns `count` evenly spaced still frames (max 8) between two times.
            Use it to see what actually happened on screen: deaths, wins, jump scares, the streamer's
            face, whether a moment is understandable without context.

            Args:
                from_t: First frame, in {unit}.
                to_t: Last frame, in {unit}.
                count: How many frames to return (1-8).
            """
            count = min(max(count, 1), MAX_FRAMES)
            a, b = sorted((T(from_t), T(to_t)))
            times = [a] if count == 1 else [a + (b - a) * i / (count - 1) for i in range(count)]
            blocks: list[dict] = []
            for t in times:
                jpeg = grab_frame(self.recorder, t)
                if jpeg:
                    blocks += [{"type": "text", "text": f"Frame at {L(t)}:"}, image_block(jpeg)]
            return blocks or "No frames available for that range."

        def view_chat(at_t: float, count: int = 2, spacing_seconds: float = 4.0):
            """Look at the chat panel from the screen recording to read what viewers were saying.
            Returns `count` images (max 4) starting at `at_t`, `spacing_seconds` apart. Chat reacts a
            few seconds after a moment, so look slightly after it.

            Args:
                at_t: Time of the first image, in {unit}.
                count: How many images (1-4).
                spacing_seconds: Seconds between images.
            """
            if not (self.recording and self.recording.chat):
                return "No chat panel was found in this recording."
            blocks: list[dict] = []
            for i in range(min(max(count, 1), 4)):
                t = T(at_t) + i * spacing_seconds
                jpeg = self.recording.grab_chat_frame(t)
                if jpeg:
                    blocks += [{"type": "text", "text": f"Chat at {L(t)}:"}, image_block(jpeg)]
            return blocks or "No chat images available for that time."

        def create_clip(
            start_t: float,
            end_t: float,
            category: Category,
            virality_score: int,
            title: str,
            hook: str,
            reason: str,
        ) -> str:
            """Cut a clip. Boundaries are nudged to the nearest pause in speech so the clip doesn't
            start or end mid-sentence. Returns the clip's id and real length, what was said and how
            chat reacted (times in seconds from the start of the CLIP), and the sound effects and
            music you can use. Call finish_clip next to edit and save it.

            Args:
                start_t: Where the clip starts, in {unit}.
                end_t: Where the clip ends, in {unit}.
                category: What kind of moment this is.
                virality_score: Your overall 1-10 rating of how well this would do as a short.
                title: Short title for the file.
                hook: 3-8 word hook shown at the top of the video for the whole clip.
                reason: One or two sentences on what happens and why it's worth clipping.
            """
            if virality_score < self.cfg.min_score:
                return (f"Not clipped: a score of {virality_score} is below the minimum of "
                        f"{self.cfg.min_score}. Only clip moments that clearly qualify.")
            start, end = sorted((T(start_t), T(end_t)))
            length = end - start
            if length < self.cfg.min_clip or length > self.cfg.max_clip:
                return (f"Error: the clip is {length:.0f}s; it must be between "
                        f"{self.cfg.min_clip} and {self.cfg.max_clip} seconds.")
            for c in self.clips:
                if c.start < end and start < c.end:
                    return f"Error: overlaps an existing clip ('{c.title}')."
            rng = self.recorder.available_range()
            if not rng or start < rng[0] - 1 or end > rng[1] + 1:
                return "Error: that range isn't in the footage. Call get_buffer_info for the available range."
            start, end = max(start, rng[0]), min(end, rng[1])

            clip_id = f"c{len(self.clips) + 1}"
            try:
                start, end, words, raw = self._cut_with_clean_boundaries(clip_id, start, end, title, rng)
            except Exception as exc:
                log.exception("Cutting %s failed", clip_id)
                return f"Error cutting clip: {exc}"
            clip = Clip(clip_id, start, end, title, reason, raw)
            clip.edits = {"category": category, "virality_score": virality_score, "hook": hook}
            self.clips.append(clip)
            self._words[clip_id] = words
            log.info("Cut %s (%.0fs at %s, %s, %d/10): %s",
                     clip_id, clip.duration, L(start), category, virality_score, title)

            return json.dumps({
                "clip_id": clip_id,
                "duration_seconds": round(clip.duration, 1),
                "actual_start": L(start),
                "actual_end": L(end),
                "speech": _speech_lines(words) if self.cfg.captions else "captions unavailable",
                "chat_reactions": self._chat_in_clip(start, end),
                "sound_effects_available": list(list_sounds(self.cfg.assets_dir / "sfx"))
                if self.cfg.sound_effects else [],
                "music_available": list(list_sounds(self.cfg.assets_dir / "music"))
                if self.cfg.music else [],
                "next_step": "Call finish_clip with this clip_id to add captions, sound effects and music.",
            })

        def finish_clip(
            clip_id: str,
            sound_effects: list[SoundEffectCue] = [],
            music: str = "none",
            music_volume: float = 0.15,
            captions: bool = True,
            hook: str = "",
        ) -> str:
            """Edit a clip made by create_clip and save the finished video.

            Args:
                clip_id: The id returned by create_clip.
                sound_effects: Sound effects to add. Each has `sound` (a name from sound_effects_available), `at_seconds` (seconds from the start of the clip) and optional `volume` (0.1-1.5, default 0.8).
                music: A track name from music_available, or "none".
                music_volume: Music loudness from 0.05 (subtle) to 0.4 (loud); 0.15 suits most clips.
                captions: Whether to burn in captions synced to the speech.
                hook: Optional replacement for the hook given to create_clip, e.g. after reading the speech.
            """
            clip = next((c for c in self.clips if c.id == clip_id), None)
            if not clip:
                return f"Error: no clip with id {clip_id}."
            if clip.path:
                return f"Error: {clip_id} is already finished ({clip.path.name})."
            if hook.strip():
                clip.edits["hook"] = hook.strip()
            try:
                path = self._finish(clip, sound_effects, music, music_volume, captions)
            except ValueError as exc:
                return f"Error: {exc}"
            except Exception as exc:
                log.exception("Rendering %s failed", clip_id)
                return f"Error rendering the clip: {exc}"
            return f"Saved {path.name} ({clip.duration:.0f}s) with {json.dumps(clip.edits)}."

        tools = [get_buffer_info, get_signals, get_chat_activity, get_chat_messages, get_transcript,
                 view_frames, create_clip, finish_clip]
        if self.recording and self.recording.chat:
            tools.insert(6, view_chat)
        if self.recording:  # no text chat in recordings: don't offer tools that can only say so
            tools = [t for t in tools if t not in (get_chat_activity, get_chat_messages)]
        tools = [_tool(t, tl) for t in tools]

        prompt = self._briefing(trigger, tl, focus)
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=[{"role": "user", "content": prompt}],
            # Judging what's clip-worthy is the hard part; give it room to think.
            output_config={"effort": "high"},
            # If a request is declined by a safety classifier, retry it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            max_iterations=25,
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

    # ---- cutting ---------------------------------------------------------

    def _cut_with_clean_boundaries(
        self, clip_id: str, start: float, end: float, title: str, rng: tuple[float, float]
    ) -> tuple[float, float, list[Word], Path]:
        """Cut [start, end], nudging both ends to pauses in speech. Returns the final range,
        words (clip-relative) and the raw clip path."""
        work = self.cfg.out_dir / ".work"
        if self.cfg.captions:
            # Look at a few extra seconds either side to find where sentences start and end.
            p_start, p_end = max(start - SNAP_PAD, rng[0]), min(end + SNAP_PAD, rng[1])
            probe_path = work / f"{clip_id}.probe.mp4"
            cut_clip(self.recorder, p_start, p_end, probe_path, self.cfg.vertical)
            try:
                padded_words = whisper.transcribe_words(probe_path, self.cfg.whisper_model)
            except Exception as exc:
                log.warning("Transcribing %s failed: %s", clip_id, exc)
                padded_words = []
            finally:
                probe_path.unlink(missing_ok=True)
            abs_words = [Word(w.start + p_start, w.end + p_start, w.text) for w in padded_words]
            new_start = snap_start(abs_words, start, lo=p_start)
            new_end = snap_end(abs_words, end, hi=p_end)
            if self.cfg.min_clip <= new_end - new_start <= self.cfg.max_clip + 2:
                start, end = new_start, new_end
        else:
            abs_words = []

        name = self._clip_name(start, title)
        raw = work / f"{name}.raw.mp4"
        cut_clip(self.recorder, start, end, raw, self.cfg.vertical)
        words = [Word(w.start - start, w.end - start, w.text) for w in abs_words if w.start >= start and w.end <= end]
        return start, end, words, raw

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

        hook = self.cfg.fixed_hook or (clip.edits.get("hook") if self.cfg.hook else None)
        words = self._words.get(clip.id, []) if captions and self.cfg.captions else []
        overlay = None
        if words or hook:
            info = probe(clip.raw_path)
            overlay = write_ass(
                words, info["width"], info["height"], info["duration"],
                clip.raw_path.with_suffix(".ass"), hook=hook,
            )

        out = self.cfg.out_dir / clip.raw_path.name.replace(".raw.mp4", ".mp4")
        render(clip.raw_path, out, overlay, sound_cues, music_path, min(max(music_volume, 0.02), 0.6))

        clip.path = out
        clip.edits |= {
            "hook": hook,
            "captions": bool(words),
            "sound_effects": [f"{c['sound']}@{c['at_seconds']:g}s" for c in cues],
            "music": music if music_path else "none",
        }
        self._log_clip(clip)
        clip.raw_path.unlink(missing_ok=True)
        if overlay:
            overlay.unlink(missing_ok=True)
        log.info("Finished %s -> %s %s", clip.id, out, clip.edits)
        return out

    def _finish_leftovers(self) -> None:
        """Clips Claude cut but never finished still get saved, with captions and hook only."""
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

    def _briefing(self, trigger: str, tl: Timeline, focus: tuple[float, float] | None) -> str:
        if focus is None:
            focus = (tl.now - 300, tl.now)
        parts = [f"Trigger: {trigger}", f"All times are in {tl.unit}."]
        if self.signals:
            windows = self.signals.timeline(focus[0], focus[1], tl.now)
            hot = sorted((w for w in windows if w.score > 0), key=lambda w: w.score, reverse=True)[:6]
            span = f"{tl.label(focus[0])} to {tl.label(focus[1])}"
            if hot:
                parts.append(f"Most notable windows from {span} (signal detector):\n" + "\n".join(
                    f"- at {tl.label(w.start)} (score {w.score}): {'; '.join(w.reasons)}"
                    for w in sorted(hot, key=lambda w: w.start)
                ))
            else:
                parts.append(f"The signal detector found nothing unusual from {span}.")
        if self.chat:
            recent = self.chat.buckets(focus[0], focus[1], 15)
            parts.append("Chat messages per 15s over that range (oldest first): "
                         + ", ".join(str(r["count"]) for r in recent))
        if self.transcriber:
            parts.append("A transcript is available (get_transcript): check it for stories and hot takes too.")
        relevant = self.clips if self.recording else [c for c in self.clips if tl.now - c.end < 1800]
        if relevant:
            parts.append("Recent clips (don't overlap these):\n" + "\n".join(
                f"- '{c.title}' from {tl.label(c.start)} to {tl.label(c.end)}" for c in relevant
            ))
        else:
            parts.append("Recent clips: none")
        parts.append("Investigate, decide whether anything here truly qualifies, and if so cut and edit it.")
        return "\n\n".join(parts)

    def _clip_name(self, start: float, title: str) -> str:
        suffix = "_vertical" if self.cfg.vertical else ""
        if self.recording:
            s = round(start - self.recording.virtual_start)
            return f"{slugify(self.recording.path.stem, 30)}_{s // 60:02d}m{s % 60:02d}s_{slugify(title)}{suffix}"
        stamp = datetime.fromtimestamp(start).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{slugify(title)}{suffix}"

    def _log_clip(self, clip: Clip) -> None:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        row = asdict(clip) | {"path": str(clip.path), "raw_path": None}
        with open(self.cfg.out_dir / "clips.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


# ---- boundary snapping ----------------------------------------------------

def _inside_speech(words: list[Word], t: float, margin: float = 0.12) -> bool:
    return any(w.start - margin <= t <= w.end + margin for w in words)


def snap_start(words: list[Word], target: float, lo: float, back: float = 5.0, forward: float = 2.0) -> float:
    """Move a start time that lands mid-speech to just before the nearest phrase start,
    preferring the start of a sentence, so the clip doesn't open mid-word."""
    if not words or not _inside_speech(words, target):
        return target
    candidates = []
    for i, w in enumerate(words):
        gap_before = w.start - words[i - 1].end if i else w.start - lo
        if gap_before >= 0.25 and target - back <= w.start <= target + forward:
            point = max(w.start - min(0.2, gap_before / 2), lo)
            sentence = i == 0 or words[i - 1].text[-1:] in ".!?"
            candidates.append((abs(point - target) - (1.0 if sentence else 0), point))
    return min(candidates)[1] if candidates else target


def snap_end(words: list[Word], target: float, hi: float, back: float = 2.0, forward: float = 4.0) -> float:
    """Move an end time that lands mid-speech to just after the nearest phrase end,
    preferring the end of a sentence, so the clip doesn't cut someone off."""
    if not words or not _inside_speech(words, target):
        return target
    candidates = []
    for i, w in enumerate(words):
        gap_after = words[i + 1].start - w.end if i + 1 < len(words) else hi - w.end
        if gap_after >= 0.25 and target - back <= w.end <= target + forward:
            point = min(w.end + min(0.35, gap_after / 2), hi)
            sentence = w.text[-1:] in ".!?"
            candidates.append((abs(point - target) - (1.0 if sentence else 0), point))
    return min(candidates)[1] if candidates else target


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
