"""The Claude agent that decides what to clip.

Each review is a short, fresh conversation: Claude gets a summary of what just
happened, digs in with tools (chat, transcript, buffer info), and cuts zero or
more clips. Times are given to Claude as "seconds ago" relative to a fixed
moment captured when the review starts, so they don't drift while it thinks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import anthropic
from anthropic import beta_tool

from .chat import TwitchChat
from .clips import Clip, cut_clip
from .recorder import StreamRecorder
from .transcriber import Transcriber

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLIPPER_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """\
You are a stream clipper. You watch a live stream through tools and cut short \
clips of the moments viewers would actually want to rewatch or share: big plays, \
funny reactions, fails, surprising reveals, rage, wholesome moments, hype.

How to work:
- Start from the trigger you're given, then check chat activity and (if available) \
the transcript around it to understand what happened.
- Chat reacts a few seconds AFTER the moment itself. Start a clip 5-15 seconds \
before the chat spike begins so the setup is included, and end it once the \
reaction has played out.
- Keep clips between 15 and 60 seconds unless the moment clearly needs more \
(never over 90).
- Be selective. A small bump in chat, routine gameplay, or a moment you already \
clipped is not worth a clip. Making no clip is a perfectly good outcome.
- Don't clip a range that overlaps one listed under "Recent clips".
- Titles should be short and punchy, like a Shorts/TikTok title, with no hashtags.
- Finish with one sentence saying what you clipped, or why you clipped nothing.
"""


class ClipAgent:
    def __init__(
        self,
        recorder: StreamRecorder,
        chat: TwitchChat | None,
        transcriber: Transcriber | None,
        out_dir: Path,
        vertical: bool = False,
    ) -> None:
        self.recorder = recorder
        self.chat = chat
        self.transcriber = transcriber
        self.out_dir = out_dir
        self.vertical = vertical
        self.clips: list[Clip] = []
        self.client = anthropic.Anthropic()

    # ---- one review ------------------------------------------------------

    def review(self, trigger: str) -> None:
        now = time.time()

        def ts(seconds_ago: float) -> float:
            return now - seconds_ago

        def ago(t: float) -> int:
            return round(now - t)

        @beta_tool
        def get_buffer_info() -> str:
            """Return how far back the recorded video goes and what's already been clipped."""
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
            """What the streamer said in a time window (speech-to-text).

            Args:
                from_seconds_ago: Start of the window, in seconds before now.
                to_seconds_ago: End of the window, in seconds before now (0 = now).
            """
            if not self.transcriber:
                return "Transcription is not enabled."
            lines = self.transcriber.lines_between(ts(from_seconds_ago), ts(to_seconds_ago))
            return "\n".join(f"[-{ago(ln.start)}s] {ln.text}" for ln in lines) or "(no speech found)"

        @beta_tool
        def create_clip(start_seconds_ago: int, end_seconds_ago: int, title: str, reason: str) -> str:
            """Cut a clip from the recorded buffer and save it as an .mp4.

            Args:
                start_seconds_ago: Where the clip starts, in seconds before now.
                end_seconds_ago: Where the clip ends, in seconds before now. Must be smaller than start.
                title: Short, punchy title for the clip.
                reason: One sentence on why this moment is worth clipping.
            """
            start, end = ts(start_seconds_ago), ts(end_seconds_ago)
            if end <= start:
                return "Error: end_seconds_ago must be smaller than start_seconds_ago."
            if end - start > 90:
                return "Error: clips can be at most 90 seconds long."
            for c in self.clips:
                if c.start < end and start < c.end:
                    return f"Error: overlaps an existing clip ('{c.title}')."
            rng = self.recorder.available_range()
            if not rng or start < rng[0] or end > rng[1] + 1:
                return f"Error: that range isn't in the buffer. Call get_buffer_info for the available range."
            try:
                path = cut_clip(self.recorder, start, end, self.out_dir, title, self.vertical)
            except Exception as exc:
                return f"Error cutting clip: {exc}"
            clip = Clip(path, start, end, title, reason)
            self.clips.append(clip)
            self._log_clip(clip)
            log.info("Clipped %.0fs: %s -> %s", end - start, title, path)
            return f"Saved {path.name} ({end - start:.0f}s)."

        prompt = self._briefing(trigger, now, ago)
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=[get_buffer_info, get_chat_activity, get_chat_messages, get_transcript, create_clip],
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium"},
            # If a request is declined by a safety classifier, retry it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            max_iterations=12,
        )
        final = runner.until_done()
        if final.stop_reason == "refusal":
            log.warning("Review declined by the model.")
            return
        summary = " ".join(b.text for b in final.content if b.type == "text").strip()
        log.info("Review done: %s", summary or "(no summary)")

    # ---- helpers ---------------------------------------------------------

    def _briefing(self, trigger: str, now: float, ago) -> str:
        parts = [f"Trigger: {trigger}"]
        if self.chat:
            recent = self.chat.buckets(now - 120, now, 10)
            parts.append(
                "Chat messages per 10s over the last 2 minutes (oldest first): "
                + ", ".join(str(r["count"]) for r in recent)
            )
        recent_clips = [c for c in self.clips if now - c.end < 1800]
        if recent_clips:
            parts.append("Recent clips:\n" + "\n".join(
                f"- '{c.title}' from -{ago(c.start)}s to -{ago(c.end)}s" for c in recent_clips
            ))
        else:
            parts.append("Recent clips: none")
        parts.append("Decide whether anything here is worth clipping, and clip it if so.")
        return "\n\n".join(parts)

    def _log_clip(self, clip: Clip) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "clips.jsonl", "a") as f:
            row = asdict(clip) | {"path": str(clip.path)}
            f.write(json.dumps(row) + "\n")
