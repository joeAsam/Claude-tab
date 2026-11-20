"""Auto-post SupoClip clips to TikTok and YouTube through Buffer.

Runs on the same computer as SupoClip. Every few minutes it asks SupoClip's API
for finished clips, and for each new one:
  1. downloads it as a 1080x1920 TikTok/Shorts export,
  2. has Claude write a caption and a YouTube title from SupoClip's hook title,
     transcript and reasoning (or falls back to the hook title),
  3. uploads it to Cloudinary and adds it to your Buffer queues.
Buffer then posts one clip per daily slot, so a video that produced 8 clips
keeps both channels posting for 8 days. Clips are queued best-first (SupoClip's
virality score), and every clip is posted at most once.

Setup, on the computer running SupoClip:
  SUPOCLIP_API_KEY   SupoClip app -> Agents & API (/settings/api-keys), starts with sk_
  SUPOCLIP_API_URL   SupoClip's backend (default http://localhost:8000)
  BUFFER_API_KEY     Buffer -> Settings -> API
  CLOUDINARY_URL     Cloudinary dashboard -> "API environment variable"
  ANTHROPIC_API_KEY  optional: Claude writes the captions (otherwise the hook title is used)

  python -m clipper.supoclip              # keep watching (Ctrl+C to stop)
  python -m clipper.supoclip --once       # check once, e.g. from cron / Task Scheduler
  python -m clipper.supoclip --dry-run    # show what would be posted, change nothing

On the first run, clips that already exist are only remembered, not posted, so an
old backlog doesn't flood your queues. Pass --backlog to post them too.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .clips import slugify
from .publish import PostPlan, Publisher, PublishError, youtube_title_from

log = logging.getLogger("clipper.supoclip")

MODEL = os.environ.get("CLIPPER_MODEL", "claude-opus-5-5")


class SupoClipError(RuntimeError):
    pass


class SupoClipClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, path: str, timeout: float = 60) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "stream-clipper"},
        )

    def get_json(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self._request(path), timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SupoClipError("SupoClip rejected the API key. Check SUPOCLIP_API_KEY.") from None
            raise SupoClipError(f"SupoClip API error {exc.code} on {path}: "
                                f"{exc.read().decode(errors='replace')[:200]}") from None
        except urllib.error.URLError as exc:
            raise SupoClipError(f"Can't reach SupoClip at {self.base_url} ({exc.reason}). Is it running?") from None

    def completed_tasks(self) -> list[dict]:
        tasks = self.get_json("/tasks/?limit=50").get("tasks", [])
        return [t for t in tasks if t.get("status") == "completed"]

    def clips(self, task_id: str) -> list[dict]:
        return self.get_json(f"/tasks/{urllib.parse.quote(task_id)}/clips").get("clips", [])

    def download(self, task_id: str, clip_id: str, dest: Path, preset: str = "tiktok") -> Path:
        """Export the clip with a 1080x1920 platform preset (waits for the render) and save it."""
        path = (f"/tasks/{urllib.parse.quote(task_id)}/clips/{urllib.parse.quote(clip_id)}"
                f"/export?preset={preset}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(self._request(path), timeout=900) as resp, open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            raise SupoClipError(f"Export of clip {clip_id} failed ({exc.code}): "
                                f"{exc.read().decode(errors='replace')[:200]}") from None
        tmp.replace(dest)
        return dest


# ---- captions ----------------------------------------------------------------

def fallback_caption(clip: dict, task: dict) -> str:
    title = (clip.get("hook_title") or "").strip() or (clip.get("text") or "").strip()[:80] or "Wait for it"
    return f"{title} #fyp #viral #shorts"


def write_caption(clip: dict, task: dict) -> tuple[str, str]:
    """(caption, youtube_title). Uses Claude when ANTHROPIC_API_KEY is set, else SupoClip's hook title."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        caption = fallback_caption(clip, task)
        return caption, youtube_title_from(caption)
    try:
        import anthropic
        from pydantic import BaseModel, Field

        class Caption(BaseModel):
            caption: str = Field(description="TikTok/Shorts caption: one punchy sentence plus 4-8 hashtags")
            youtube_title: str = Field(description="YouTube Shorts title, at most 90 characters, no hashtags")

        transcript = re.sub(r"\s+", " ", clip.get("text") or "")[:1500]
        prompt = f"""Write the posting caption for a short vertical video clip.

Source video: {task.get("source_title") or "unknown"}
On-screen hook title: {clip.get("hook_title") or "none"}
Why it was picked: {clip.get("reasoning") or "n/a"}
What is said in the clip: {transcript or "n/a"}

Caption: one punchy sentence that makes people want to watch, naming the person or \
streamer if the source makes it clear who it is, plus one or two emoji at most, then \
4-8 hashtags (the person, the topic or game, and broad ones like #fyp). YouTube title: \
the same hook without hashtags, at most 90 characters. Only use facts from the \
information above; never invent names or events."""
        response = anthropic.Anthropic().beta.messages.parse(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            output_format=Caption,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.parsed_output is None:
            raise RuntimeError(f"no caption (stop reason {response.stop_reason})")
        out = response.parsed_output
        return out.caption.strip(), youtube_title_from(out.youtube_title)
    except Exception as exc:
        log.warning("Claude couldn't write the caption (%s); using SupoClip's hook title", exc)
        caption = fallback_caption(clip, task)
        return caption, youtube_title_from(caption)


# ---- state ---------------------------------------------------------------------

class State:
    """Remembers which SupoClip clips were already handled."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.existed = path.exists()
        self.data = json.loads(path.read_text()) if self.existed else {"handled": {}}

    def handled(self, clip_id: str) -> bool:
        return clip_id in self.data["handled"]

    def mark(self, clip_id: str, how: str) -> None:
        self.data["handled"][clip_id] = {"how": how, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)


# ---- one pass ------------------------------------------------------------------

def run_once(supo: SupoClipClient, publisher: Publisher, state: State, work_dir: Path,
             min_virality: int, backlog: bool, dry_run: bool) -> int:
    """Handle every new finished clip. Returns how many clips were queued."""
    first_run = not state.existed and not state.data["handled"]
    candidates = []
    for task in supo.completed_tasks():
        for clip in supo.clips(task["id"]):
            if not state.handled(clip["id"]):
                candidates.append((task, clip))

    if first_run and not backlog:
        for _, clip in candidates:
            if not dry_run:
                state.mark(clip["id"], "existing before first run")
        log.info("First run: remembered %d existing clip(s) without posting them. New clips from now on "
                 "will be posted (use --backlog to post existing ones).", len(candidates))
        return 0

    candidates.sort(key=lambda tc: tc[1].get("virality_score") or 0, reverse=True)
    queued = 0
    for task, clip in candidates:
        score = clip.get("virality_score") or 0
        label = clip.get("hook_title") or clip.get("filename") or clip["id"]
        if score < min_virality:
            log.info("Skipping '%s' (virality %d < %d)", label, score, min_virality)
            if not dry_run:
                state.mark(clip["id"], f"skipped: virality {score}")
            continue
        caption, yt_title = write_caption(clip, task)
        if dry_run:
            log.info("[dry run] would queue '%s' (virality %d)\n    caption: %s\n    YouTube title: %s",
                     label, score, caption, yt_title)
            continue
        name = f"{task['id'][:8]}_{clip['id'][:8]}_{slugify(clip.get('hook_title') or 'clip', 40)}.mp4"
        path = work_dir / name
        if not path.exists():
            log.info("Downloading '%s' from SupoClip...", label)
            supo.download(task["id"], clip["id"], path)
        publisher.post(PostPlan(path, caption, yt_title))
        state.mark(clip["id"], "queued")
        queued += 1
    return queued


def main() -> int:
    p = argparse.ArgumentParser(description="Post new SupoClip clips to TikTok and YouTube through Buffer.")
    p.add_argument("--api-url", default=os.environ.get("SUPOCLIP_API_URL", "http://localhost:8000"),
                   help="SupoClip backend URL (default http://localhost:8000)")
    p.add_argument("--channels", default="tiktok,youtube", help="Buffer channels to post to (default tiktok,youtube)")
    p.add_argument("--youtube-category", default="20", help="YouTube category ID (20 = Gaming, 24 = Entertainment)")
    p.add_argument("--min-virality", type=int, default=0, help="Only post clips SupoClip scored at least this (0-100)")
    p.add_argument("--every", type=float, default=2.0, help="Minutes between checks (default 2)")
    p.add_argument("--once", action="store_true", help="Check once and exit")
    p.add_argument("--backlog", action="store_true", help="On the first run, also post clips that already exist")
    p.add_argument("--dry-run", action="store_true", help="Show what would be posted without posting anything")
    p.add_argument("--work-dir", type=Path, default=Path("downloads/supoclip"),
                   help="Where downloaded clips and the posted-clip records are kept")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    api_key = os.environ.get("SUPOCLIP_API_KEY")
    if not api_key:
        log.error("SUPOCLIP_API_KEY is not set. Create one in SupoClip: Agents & API (/settings/api-keys).")
        return 1
    try:
        supo = SupoClipClient(args.api_url, api_key)
        publisher = Publisher([c for c in args.channels.split(",") if c.strip()], args.work_dir / "posted.jsonl",
                              args.youtube_category, dry_run=args.dry_run)
    except PublishError as exc:
        log.error("%s", exc)
        return 1
    state = State(args.work_dir / "supoclip_state.json")
    log.info("Watching SupoClip at %s; posting to %s", args.api_url,
             ", ".join(f"{c['name']} ({c['service']})" for c in publisher.channels))

    while True:
        try:
            n = run_once(supo, publisher, state, args.work_dir, args.min_virality, args.backlog, args.dry_run)
            if n:
                log.info("Queued %d new clip(s) on Buffer", n)
        except (SupoClipError, PublishError) as exc:
            log.error("%s", exc)
            if args.once:
                return 1
        if args.once:
            return 0
        try:
            time.sleep(max(args.every, 0.5) * 60)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
