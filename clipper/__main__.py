"""Command-line entry point: python -m clipper <stream-url>"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import anthropic

from . import whisper
from .agent import ClipAgent, ClipAgentConfig
from .clips import slugify
from .filesource import Crop, FileSource, probe_video
from .layout import Layout, detect_layout, save_preview
from .chat import TwitchChat
from .recorder import StreamRecorder
from .signals import SignalMonitor
from .sounds import ensure_default_assets
from .transcriber import Transcriber

log = logging.getLogger("clipper")


def twitch_channel(url: str) -> str | None:
    m = re.match(r"^(?:https?://)?(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", url)
    return m.group(1) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Let Claude find, clip and edit the best moments of a live stream or a recording.")
    p.add_argument("source", help="A live stream URL (Twitch, YouTube, Kick, ...) OR a video file, "
                                  "such as a screen recording of a stream")

    g = p.add_argument_group("clips")
    g.add_argument("--out", type=Path, default=Path("clips"), help="Where to save clips")
    g.add_argument("--min-clip", type=int, default=30, help="Shortest clip in seconds (default 30)")
    g.add_argument("--max-clip", type=int, default=240, help="Longest clip in seconds (default 240 = 4 min)")
    g.add_argument("--vertical", action="store_true", help="Crop clips to 9:16 for Shorts/TikTok/Reels")
    g.add_argument("--min-score", type=int, default=7,
                   help="Only clip moments Claude rates at least this viral, 1-10 (default 7)")

    g = p.add_argument_group("editing")
    g.add_argument("--no-captions", action="store_true", help="Don't burn in captions")
    g.add_argument("--no-sfx", action="store_true", help="Don't add sound effects")
    g.add_argument("--no-music", action="store_true", help="Don't add background music")
    g.add_argument("--hook", metavar="TEXT", help="Use this fixed hook title on every clip instead of Claude's")
    g.add_argument("--no-hook", action="store_true", help="Don't put a hook title on the video")
    g.add_argument("--assets", type=Path, default=Path("assets"),
                   help="Folder with sfx/ and music/ subfolders (starter sounds are created here)")
    g.add_argument("--whisper-model", default="base",
                   help="faster-whisper model for captions: tiny, base, small, medium, large-v3")

    g = p.add_argument_group("video files / screen recordings")
    g.add_argument("--max-clips", type=int, default=10, help="Stop after this many clips (default 10)")
    g.add_argument("--max-reviews", type=int, default=15,
                   help="Most candidate moments Claude reviews (each review costs API usage; default 15)")
    g.add_argument("--crop", type=Crop.parse, metavar="W:H:X:Y",
                   help="Where the stream player is in the recording, in pixels (default: found automatically)")
    g.add_argument("--chat-crop", type=Crop.parse, metavar="W:H:X:Y",
                   help="Where the chat panel is in the recording (default: found automatically)")
    g.add_argument("--no-crop", action="store_true", help="The file is already just the stream: don't crop")

    g = p.add_argument_group("live recording")
    g.add_argument("--buffer-dir", type=Path, default=Path("buffer"), help="Where to keep the rolling recording")
    g.add_argument("--buffer-minutes", type=int, default=15, help="How much recent video to keep")
    g.add_argument("--quality", default="best", help="streamlink quality, e.g. best, 720p60, 480p")

    g = p.add_argument_group("finding moments")
    g.add_argument("--no-transcribe", action="store_true",
                   help="Don't transcribe the stream live (saves CPU, but stories and hot takes get missed)")
    g.add_argument("--sensitivity", type=float, default=3.0,
                   help="Interest score that wakes the agent; lower = more reviews (default 3.0)")
    g.add_argument("--review-minutes", type=float, default=0,
                   help="Also review every N minutes even if nothing triggers (0 = off)")
    g.add_argument("--cooldown", type=int, default=45, help="Minimum seconds between reviews")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if not 1 <= args.min_clip <= args.max_clip:
        p.error("--min-clip must be at least 1 and no more than --max-clip")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if Path(args.source).is_file():
        return run_file(args)
    return run_live(args)


def setup(args) -> bool:
    for tool in ("ffmpeg", "ffprobe") + (() if Path(args.source).is_file() else ("streamlink",)):
        if not shutil.which(tool):
            log.error("%s is not installed or not on PATH.", tool)
            return False
    ensure_default_assets(args.assets)
    return True


def agent_config(args) -> ClipAgentConfig:
    return ClipAgentConfig(
        out_dir=args.out,
        assets_dir=args.assets,
        vertical=args.vertical,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
        min_score=args.min_score,
        captions=not args.no_captions,
        whisper_model=args.whisper_model,
        sound_effects=not args.no_sfx,
        music=not args.no_music,
        hook=not args.no_hook,
        fixed_hook=args.hook,
    )


# ---- video files / screen recordings ----------------------------------------

def run_file(args) -> int:
    if not setup(args):
        return 1
    path = Path(args.source)
    info = probe_video(path)
    log.info("Recording: %s, %dx%d, %.1f minutes", path.name, info["width"], info["height"], info["duration"] / 60)
    if info["duration"] < args.min_clip:
        log.error("The recording is shorter than --min-clip (%ds).", args.min_clip)
        return 1

    # 1. Find the stream player and the chat panel in the recording.
    work = args.buffer_dir / slugify(path.stem, 40)
    work.mkdir(parents=True, exist_ok=True)
    if args.no_crop:
        layout = Layout(None, args.chat_crop)
    elif args.crop:
        layout = Layout(args.crop, args.chat_crop)
    else:
        log.info("Finding the stream player and chat in the recording...")
        layout = detect_layout(anthropic.Anthropic(), path, info["width"], info["height"], info["duration"])
        if args.chat_crop:
            layout.chat = args.chat_crop
    preview = save_preview(path, info["duration"] * 0.3, layout, work / "layout_preview.jpg")
    log.info("Player crop: %s | chat panel: %s | check: %s",
             layout.player or "full frame", layout.chat or "not found", preview)
    if layout.notes:
        log.info("Layout notes: %s", layout.notes)

    # 2. Crop and split the recording, transcribe it, and measure every signal.
    source = FileSource(path, work, info["duration"], layout.player, layout.chat)
    source.prepare()
    transcriber = None
    if not args.no_transcribe and whisper.available():
        log.info("Transcribing the whole recording (the first run downloads the speech model)...")
        transcriber = Transcriber(source, args.whisper_model)
        transcriber.load_words(whisper.transcribe_words(path, args.whisper_model), source.virtual_start)
    signals = SignalMonitor(source, None, transcriber)
    if layout.chat:
        signals.load_chat_scroll(path, layout.chat.filter, source.virtual_start)
    signals.analyze_all()
    start, end = source.available_range()
    signals.use_whole_range_as_baseline(start, end)

    # 3. Rank the moments, then let Claude review the best ones.
    agent = ClipAgent(source, None, transcriber, signals, agent_config(args), recording=source)
    candidates = pick_candidates(signals, start, end, args.sensitivity, spacing=args.min_clip)
    log.info("Found %d candidate moments above sensitivity %.1f", len(candidates), args.sensitivity)

    def finished() -> int:
        return sum(1 for c in agent.clips if c.path)

    reviews = 0
    for w in candidates:
        if finished() >= args.max_clips or reviews >= args.max_reviews:
            break
        if any(c.start - 10 <= w.start <= c.end + 10 for c in agent.clips):
            continue  # already covered by a clip
        label = _mmss(w.start - start)
        log.info("Reviewing candidate at %s (score %.1f): %s", label, w.score, "; ".join(w.reasons))
        reviews += 1
        try:
            agent.review(
                f"candidate moment at {label} = {round(w.start - start)}s (score {w.score}): {'; '.join(w.reasons)}",
                focus=(max(start, w.start - 120), min(end, w.start + 90)),
            )
        except Exception:
            log.exception("Review failed; moving on")

    # 4. A last pass over the transcript for stories and hot takes the signals can't hear.
    if transcriber and finished() < args.max_clips and reviews < args.max_reviews:
        log.info("Final pass: reading the whole transcript for stories, hot takes and quieter gems...")
        try:
            agent.review(
                "final pass over the whole recording. Read the full transcript (get_transcript from 0) "
                "and look for great moments the loudness/chat signals may have missed: gripping "
                "stories, hot takes, funny exchanges, intriguing or unique moments. Clip any that "
                f"qualify (up to {args.max_clips - finished()} more), without overlapping existing clips.",
                focus=(start, end),
            )
        except Exception:
            log.exception("Final pass failed")

    log.info("Done: %d clip(s) in %s/", finished(), args.out)
    return 0


def pick_candidates(signals: SignalMonitor, start: float, end: float, threshold: float, spacing: float):
    """Highest-scoring windows, at least `spacing` seconds apart."""
    windows = sorted((w for w in signals.timeline(start, end) if w.score >= threshold),
                     key=lambda w: w.score, reverse=True)
    picked = []
    for w in windows:
        if all(abs(w.start - p.start) >= spacing for p in picked):
            picked.append(w)
    return picked


def _mmss(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 60}:{s % 60:02d}"


# ---- live streams ------------------------------------------------------------

def run_live(args) -> int:
    if not setup(args):
        return 1
    if args.buffer_minutes * 60 < args.max_clip + 120:
        args.buffer_minutes = (args.max_clip + 120 + 59) // 60
        log.info("Raised --buffer-minutes to %d to fit %ds clips.", args.buffer_minutes, args.max_clip)

    recorder = StreamRecorder(args.source, args.buffer_dir, buffer_minutes=args.buffer_minutes, quality=args.quality)
    channel = twitch_channel(args.source)
    chat = TwitchChat(channel) if channel else None
    transcriber = None if args.no_transcribe else Transcriber.try_create(recorder, args.whisper_model)
    signals = SignalMonitor(recorder, chat, transcriber)

    if not chat and not args.review_minutes:
        log.info("No chat reader for this platform: detecting moments from audio, video and speech, "
                 "plus a review every 5 minutes for anything subtler.")
        args.review_minutes = 5

    agent = ClipAgent(recorder, chat, transcriber, signals, agent_config(args))

    recorder.start()
    signals.start()
    if chat:
        chat.start()
    if transcriber:
        transcriber.start()

    log.info("Watching %s. Clips (%d-%ds) go to %s/. Press Ctrl+C to stop.",
             args.source, args.min_clip, args.max_clip, args.out)
    last_review = time.time()
    try:
        while True:
            time.sleep(5)
            now = time.time()
            rng = recorder.available_range()
            # Wait until there's at least one minimum-length clip's worth of footage.
            if now - last_review < args.cooldown or not rng or rng[1] - rng[0] < args.min_clip:
                continue
            trigger = None
            # Score windows that ended at least 10s before the newest analyzed footage,
            # so the recording already covers the reaction when Claude goes to clip it.
            done = signals.analyzed_until
            hot = signals.hottest(done - 40, done - 10) if done else None
            if hot and hot.score >= args.sensitivity:
                trigger = (f"interest score {hot.score} around {now - hot.start:.0f}s ago: "
                           + "; ".join(hot.reasons))
            if not trigger and args.review_minutes and now - last_review >= args.review_minutes * 60:
                trigger = f"scheduled check of the last {args.review_minutes:g} minutes"
            if trigger:
                log.info("Reviewing (%s)", trigger)
                last_review = now
                try:
                    agent.review(trigger)
                except Exception:
                    log.exception("Review failed; will keep watching")
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        recorder.stop()
        signals.stop()
        if chat:
            chat.stop()
        if transcriber:
            transcriber.stop()
    log.info("Made %d clip(s).", sum(1 for c in agent.clips if c.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
