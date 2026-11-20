"""Posting finished clips through Buffer.

Buffer has no upload endpoint: it fetches each video from a public URL when the
post goes out. So a clip is first uploaded to Cloudinary (a free media host),
then added to your Buffer queue for each chosen channel. With one posting slot
a day in Buffer's posting schedule, queued clips go out once a day.

Setup (on the machine that runs the clipper):
  BUFFER_API_KEY   from Buffer: Settings -> API (publish.buffer.com/settings/api)
  CLOUDINARY_URL   from the Cloudinary dashboard ("API environment variable"),
                   e.g. cloudinary://<key>:<secret>@<cloud_name>
     or, instead of CLOUDINARY_URL, an unsigned upload preset:
  CLOUDINARY_CLOUD_NAME and CLOUDINARY_UPLOAD_PRESET

Command line, for clips already made:
  python -m clipper.publish                    # every clip in clips/ not posted yet
  python -m clipper.publish clips/x.mp4 --channels tiktok,youtube --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

BUFFER_API = "https://api.buffer.com"
YOUTUBE_TITLE_MAX = 100
YOUTUBE_SHORTS_MAX_SECONDS = 180


class PublishError(RuntimeError):
    pass


# ---- Buffer ------------------------------------------------------------------

class BufferClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("BUFFER_API_KEY")
        if not self.api_key:
            raise PublishError("BUFFER_API_KEY is not set. Create a key in Buffer: Settings -> API.")
        self._channels: list[dict] | None = None

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        body = json.dumps({"query": query, "variables": variables or {}}).encode()
        req = urllib.request.Request(BUFFER_API, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "stream-clipper",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 401:
                raise PublishError("Buffer rejected the API key (401). Check BUFFER_API_KEY.") from None
            if exc.code == 429:
                raise PublishError(f"Buffer rate limit reached; try again later. {detail}") from None
            raise PublishError(f"Buffer API error {exc.code}: {detail}") from None
        if data.get("errors"):
            raise PublishError("Buffer API: " + "; ".join(e.get("message", "?") for e in data["errors"]))
        return data["data"]

    def channels(self) -> list[dict]:
        if self._channels is None:
            orgs = self.graphql("query { account { organizations { id name } } }")["account"]["organizations"]
            found = []
            for org in orgs:
                found += self.graphql(
                    "query($org: OrganizationId!) { channels(input: { organizationId: $org }) "
                    "{ id name service isDisconnected isQueuePaused } }",
                    {"org": org["id"]},
                )["channels"]
            self._channels = found
        return self._channels

    def find_channel(self, wanted: str) -> dict:
        """Find a channel by service ("tiktok", "youtube") or by its name/ID."""
        w = wanted.lower().strip()
        matches = [c for c in self.channels()
                   if w in (c["id"].lower(), c["name"].lower(), str(c["service"]).lower())]
        if not matches:
            have = ", ".join(f"{c['name']} ({c['service']})" for c in self.channels()) or "none"
            raise PublishError(f"No Buffer channel matches '{wanted}'. Connected channels: {have}")
        if len(matches) > 1:
            names = ", ".join(f"{c['name']} ({c['id']})" for c in matches)
            raise PublishError(f"'{wanted}' matches several channels ({names}); use the channel name or ID.")
        ch = matches[0]
        if ch.get("isDisconnected"):
            raise PublishError(f"Buffer channel {ch['name']} is disconnected; reconnect it in Buffer.")
        return ch

    def queue_video(
        self,
        channel: dict,
        video_url: str,
        text: str,
        youtube_title: str | None = None,
        youtube_category: str = "20",
        due_at: str | None = None,
    ) -> dict:
        """Add a video post to the channel's queue (next free slot), or schedule it at due_at (ISO UTC)."""
        post_input: dict = {
            "channelId": channel["id"],
            "text": text,
            "schedulingType": "automatic",
            "mode": "customScheduled" if due_at else "addToQueue",
            "needsApproval": False,
            "assets": [{"video": {"url": video_url}}],
        }
        if due_at:
            post_input["dueAt"] = due_at
        if channel["service"] == "youtube":
            if "#shorts" not in text.lower():
                post_input["text"] = f"{text} #Shorts"
            post_input["metadata"] = {"youtube": {
                "title": youtube_title_from(youtube_title or text),
                "categoryId": youtube_category,
                "privacy": "public",
                "madeForKids": False,
                "notifySubscribers": False,
            }}
        data = self.graphql(
            """mutation($input: CreatePostInput!) {
              createPost(input: $input) {
                ... on PostActionSuccess { post { id dueAt status } }
                ... on MutationError { message }
              }
            }""",
            {"input": post_input},
        )["createPost"]
        if "post" not in data:
            raise PublishError(f"Buffer refused the post for {channel['name']}: {data.get('message', data)}")
        return data["post"]


# ---- media hosting --------------------------------------------------------------

def upload_video(path: Path) -> str:
    """Upload a video to Cloudinary and return its permanent public https URL."""
    try:
        import cloudinary
        import cloudinary.uploader
    except ImportError:
        raise PublishError("The cloudinary package is missing: pip install cloudinary") from None

    options = {"resource_type": "video", "folder": "clipper", "use_filename": True, "unique_filename": True}
    if os.environ.get("CLOUDINARY_URL"):
        cloudinary.config(secure=True)  # reads CLOUDINARY_URL
        result = cloudinary.uploader.upload_large(str(path), **options)
    elif os.environ.get("CLOUDINARY_CLOUD_NAME") and os.environ.get("CLOUDINARY_UPLOAD_PRESET"):
        cloudinary.config(cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"], secure=True)
        result = cloudinary.uploader.unsigned_upload(
            str(path), os.environ["CLOUDINARY_UPLOAD_PRESET"], **options)
    else:
        raise PublishError(
            "No media host configured. Set CLOUDINARY_URL (from the Cloudinary dashboard), or "
            "CLOUDINARY_CLOUD_NAME and CLOUDINARY_UPLOAD_PRESET for an unsigned upload preset.")
    url = result.get("secure_url")
    if not url:
        raise PublishError(f"Cloudinary upload returned no URL: {result}")
    return url


# ---- posting clips -----------------------------------------------------------------

@dataclass
class PostPlan:
    path: Path
    caption: str
    title: str


def youtube_title_from(caption: str) -> str:
    """A YouTube title from a caption: first line, hashtags removed, at most 100 characters."""
    line = next((ln for ln in caption.splitlines() if ln.strip()), caption)
    title = re.sub(r"\s*#[\w-]+", "", line).strip() or line.strip()
    if len(title) > YOUTUBE_TITLE_MAX:
        title = title[: YOUTUBE_TITLE_MAX - 1].rsplit(" ", 1)[0] + "…"
    return title


def _clip_shape(path: Path) -> tuple[float, bool] | None:
    """(duration in seconds, is vertical), or None if ffprobe can't read it."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        info = json.loads(out)
        stream = info["streams"][0]
        return float(info["format"]["duration"]), stream["height"] > stream["width"]
    except Exception:
        return None


def default_caption(title: str, hook: str | None) -> str:
    return (hook or title).strip()


class Publisher:
    """Uploads clips once and queues them on each chosen Buffer channel, remembering what was posted."""

    def __init__(self, channels: list[str], log_path: Path, youtube_category: str = "20",
                 dry_run: bool = False, client: BufferClient | None = None) -> None:
        self.channel_names = channels
        self.log_path = log_path
        self.youtube_category = youtube_category
        self.dry_run = dry_run
        self.client = client or BufferClient()
        self.channels = [self.client.find_channel(c) for c in channels]  # fail fast on typos

    def already_posted(self) -> set[tuple[str, str]]:
        if not self.log_path.exists():
            return set()
        done = set()
        for line in self.log_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add((row["clip"], row["channel_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
        return done

    def post(self, plan: PostPlan) -> list[dict]:
        done = self.already_posted()
        todo = [c for c in self.channels if (plan.path.name, c["id"]) not in done]
        if not todo:
            log.info("%s is already queued on every chosen channel", plan.path.name)
            return []
        if self.dry_run:
            for c in todo:
                log.info("[dry run] would queue %s on %s (%s): %r", plan.path.name, c["name"], c["service"],
                         plan.caption[:80])
            return []
        if any(c["service"] == "youtube" for c in todo):
            shape = _clip_shape(plan.path)
            if shape and (shape[0] > YOUTUBE_SHORTS_MAX_SECONDS or not shape[1]):
                log.warning("%s is %s and %.0fs long, so YouTube will post it as a regular video, not a Short "
                            "(Shorts must be vertical and at most %ds).", plan.path.name,
                            "vertical" if shape[1] else "horizontal", shape[0], YOUTUBE_SHORTS_MAX_SECONDS)
        log.info("Uploading %s to Cloudinary...", plan.path.name)
        url = upload_video(plan.path)
        results = []
        for c in todo:
            post = self.client.queue_video(c, url, plan.caption, youtube_title=youtube_title_from(plan.caption),
                                           youtube_category=self.youtube_category)
            log.info("Queued %s on %s (%s) for %s", plan.path.name, c["name"], c["service"], post.get("dueAt"))
            row = {"clip": plan.path.name, "channel_id": c["id"], "channel": c["name"],
                   "service": c["service"], "post_id": post["id"], "due_at": post.get("dueAt"), "video_url": url}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            results.append(row)
        return results


def plans_from_clip_log(clips_dir: Path) -> list[PostPlan]:
    """Captions and titles for finished clips, from the clipper's clips.jsonl."""
    plans = []
    log_file = clips_dir / "clips.jsonl"
    if not log_file.exists():
        return plans
    for line in log_file.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = Path(row.get("path") or "")
        if not path.name or not (clips_dir / path.name).exists():
            continue
        edits = row.get("edits") or {}
        caption = edits.get("post_caption") or default_caption(row.get("title", path.stem), edits.get("hook"))
        plans.append(PostPlan(clips_dir / path.name, caption, row.get("title") or path.stem))
    return plans


def main() -> int:
    p = argparse.ArgumentParser(description="Queue finished clips on your Buffer channels.")
    p.add_argument("clips", nargs="*", type=Path, help="Clip files (default: every unposted clip in --clips-dir)")
    p.add_argument("--clips-dir", type=Path, default=Path("clips"))
    p.add_argument("--channels", default="tiktok,youtube",
                   help="Comma-separated Buffer channels: service (tiktok, youtube), channel name or ID")
    p.add_argument("--caption", help="Caption to use (default: the one Claude wrote for each clip)")
    p.add_argument("--youtube-category", default="20", help="YouTube category ID (20 = Gaming, 24 = Entertainment)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be queued without uploading or posting")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    try:
        pub = Publisher([c for c in args.channels.split(",") if c.strip()], args.clips_dir / "posted.jsonl",
                        args.youtube_category, args.dry_run)
        known = {pl.path.name: pl for pl in plans_from_clip_log(args.clips_dir)}
        if args.clips:
            plans = [known.get(c.name) or PostPlan(c, c.stem.replace("-", " "), c.stem) for c in args.clips]
        else:
            plans = list(known.values())
        if not plans:
            log.info("No clips to post.")
            return 0
        for plan in plans:
            if args.caption:
                plan.caption = args.caption
            pub.post(plan)
    except PublishError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
