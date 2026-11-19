"""Command-line entry point: python -m clipper <stream-url>"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path

from .agent import ClipAgent
from .chat import TwitchChat
from .recorder import StreamRecorder
from .transcriber import Transcriber

log = logging.getLogger("clipper")


def twitch_channel(url: str) -> str | None:
    m = re.match(r"^(?:https?://)?(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", url)
    return m.group(1) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a live stream and let Claude clip the best moments.")
    p.add_argument("url", help="Stream URL (Twitch, YouTube, Kick, ... anything streamlink supports)")
    p.add_argument("--out", type=Path, default=Path("clips"), help="Where to save clips")
    p.add_argument("--buffer-dir", type=Path, default=Path("buffer"), help="Where to keep the rolling recording")
    p.add_argument("--buffer-minutes", type=int, default=15, help="How much recent video to keep")
    p.add_argument("--quality", default="best", help="streamlink quality, e.g. best, 720p60, 480p")
    p.add_argument("--vertical", action="store_true", help="Crop clips to 9:16 for Shorts/TikTok/Reels")
    p.add_argument("--transcribe", metavar="MODEL", nargs="?", const="base",
                   help="Transcribe speech with faster-whisper (model size, default 'base')")
    p.add_argument("--spike-ratio", type=float, default=2.5,
                   help="Chat must be this many times busier than usual to trigger a review")
    p.add_argument("--min-rate", type=float, default=0.5,
                   help="...and at least this many messages per second")
    p.add_argument("--review-minutes", type=float, default=0,
                   help="Also review every N minutes even without a chat spike (0 = off)")
    p.add_argument("--cooldown", type=int, default=45, help="Minimum seconds between reviews")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for tool in ("ffmpeg", "streamlink"):
        if not shutil.which(tool):
            log.error("%s is not installed or not on PATH.", tool)
            return 1

    recorder = StreamRecorder(args.url, args.buffer_dir, buffer_minutes=args.buffer_minutes, quality=args.quality)
    channel = twitch_channel(args.url)
    chat = TwitchChat(channel) if channel else None
    transcriber = Transcriber.try_create(recorder, args.transcribe) if args.transcribe else None

    if not chat and not transcriber and not args.review_minutes:
        log.warning("No chat for this platform: falling back to a review every 3 minutes. "
                    "Add --transcribe so the agent can hear what's happening.")
        args.review_minutes = 3

    agent = ClipAgent(recorder, chat, transcriber, args.out, vertical=args.vertical)

    recorder.start()
    if chat:
        chat.start()
    if transcriber:
        transcriber.start()

    log.info("Watching %s. Clips go to %s/. Press Ctrl+C to stop.", args.url, args.out)
    last_review = time.time()
    try:
        while True:
            time.sleep(5)
            now = time.time()
            if now - last_review < args.cooldown or not recorder.available_range():
                continue
            trigger = None
            if chat:
                # Look at a window that ended 15s ago, so the recording already
                # covers the reaction by the time Claude goes to clip it.
                recent = chat.rate(now - 35, now - 15)
                baseline = chat.rate(now - 600, now - 60)
                if recent >= args.min_rate and recent >= args.spike_ratio * max(baseline, 0.05):
                    trigger = (f"chat spike: {recent:.1f} msg/s between 35s and 15s ago, "
                               f"vs a usual {baseline:.1f} msg/s")
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
        if chat:
            chat.stop()
        if transcriber:
            transcriber.stop()
    log.info("Made %d clip(s).", len(agent.clips))
    return 0


if __name__ == "__main__":
    sys.exit(main())
