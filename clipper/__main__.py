"""Command-line entry point: python -m clipper <stream-url>"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path

from .agent import ClipAgent, ClipAgentConfig
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
    p = argparse.ArgumentParser(description="Watch a live stream and let Claude clip and edit the best moments.")
    p.add_argument("url", help="Stream URL (Twitch, YouTube, Kick, ... anything streamlink supports)")

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

    g = p.add_argument_group("recording")
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
    if args.buffer_minutes * 60 < args.max_clip + 120:
        args.buffer_minutes = (args.max_clip + 120 + 59) // 60
        print(f"Raised --buffer-minutes to {args.buffer_minutes} to fit {args.max_clip}s clips.")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for tool in ("ffmpeg", "ffprobe", "streamlink"):
        if not shutil.which(tool):
            log.error("%s is not installed or not on PATH.", tool)
            return 1

    ensure_default_assets(args.assets)

    recorder = StreamRecorder(args.url, args.buffer_dir, buffer_minutes=args.buffer_minutes, quality=args.quality)
    channel = twitch_channel(args.url)
    chat = TwitchChat(channel) if channel else None
    transcriber = None if args.no_transcribe else Transcriber.try_create(recorder, args.whisper_model)
    signals = SignalMonitor(recorder, chat, transcriber)

    if not chat and not args.review_minutes:
        log.info("No chat reader for this platform: detecting moments from audio, video and speech, "
                 "plus a review every 5 minutes for anything subtler.")
        args.review_minutes = 5

    agent = ClipAgent(recorder, chat, transcriber, signals, ClipAgentConfig(
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
    ))

    recorder.start()
    signals.start()
    if chat:
        chat.start()
    if transcriber:
        transcriber.start()

    log.info("Watching %s. Clips (%d-%ds) go to %s/. Press Ctrl+C to stop.",
             args.url, args.min_clip, args.max_clip, args.out)
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
