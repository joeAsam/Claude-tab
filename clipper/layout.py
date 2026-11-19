"""Finds the stream player and the chat panel inside a screen recording.

Claude looks at a frame (with a labelled grid drawn on it to help it measure)
and returns both rectangles. The player box is then tidied up to a clean 16:9
shape, and a preview image with both boxes drawn on is saved so you can check.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from .filesource import Crop

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLIPPER_MODEL", "claude-opus-5-5")


class Box(BaseModel):
    left: float = Field(description="Left edge as a fraction of the image width (0-1)")
    top: float = Field(description="Top edge as a fraction of the image height (0-1)")
    right: float = Field(description="Right edge as a fraction of the image width (0-1)")
    bottom: float = Field(description="Bottom edge as a fraction of the image height (0-1)")


class LayoutAnswer(BaseModel):
    player_found: bool
    player: Box = Field(description="The stream video itself, excluding browser UI, titles and chat")
    chat_found: bool
    chat: Box = Field(description="The live chat message list (the scrolling messages only)")
    notes: str


@dataclass
class Layout:
    player: Crop | None
    chat: Crop | None
    notes: str = ""


PROMPT = """\
This is a frame from a screen recording of someone watching a livestream (Twitch, \
YouTube, Kick...). A grid is drawn on it: thin lines every 10% of the width and \
height, so the lines are at 0.1, 0.2, ... 0.9.

Find:
1. The stream player: the rectangle of the actual stream video only. Exclude the \
browser's tabs and address bar, the site's header and sidebars, the stream title \
and info below the player, and the chat. If the stream video fills the whole \
image, return the whole image.
2. The live chat panel: the area where chat messages scroll. Exclude the chat \
header and the message input box.

Give each edge as a fraction of the image size, as precisely as you can using \
the grid lines."""


def _frame(video: Path, at: float, extra_vf: str = "") -> bytes:
    vf = "scale='min(1568,iw)':-2" + (f",{extra_vf}" if extra_vf else "")
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video),
         "-frames:v", "1", "-vf", vf, "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True, check=True,
    ).stdout


def _to_crop(box: Box, width: int, height: int) -> Crop | None:
    x0, x1 = sorted((min(max(box.left, 0), 1), min(max(box.right, 0), 1)))
    y0, y1 = sorted((min(max(box.top, 0), 1), min(max(box.bottom, 0), 1)))
    w, h = (x1 - x0) * width, (y1 - y0) * height
    if w < width * 0.05 or h < height * 0.05:
        return None
    return Crop(int(w) // 2 * 2, int(h) // 2 * 2, int(x0 * width), int(y0 * height))


def tidy_player(crop: Crop, width: int, height: int) -> Crop:
    """Stream players are 16:9. If the box is close to that, snap it to exactly 16:9
    around its centre so small measuring errors don't leave slivers of browser UI."""
    ratio = crop.w / crop.h
    if abs(ratio - 16 / 9) / (16 / 9) > 0.12:
        return crop
    cx, cy = crop.x + crop.w / 2, crop.y + crop.h / 2
    w = min(crop.w, crop.h * 16 / 9)  # shrink rather than grow, to stay inside the player
    h = w * 9 / 16
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    x = int(min(max(cx - w / 2, 0), width - w))
    y = int(min(max(cy - h / 2, 0), height - h))
    return Crop(w, h, x, y)


def detect_layout(client: anthropic.Anthropic, video: Path, width: int, height: int, duration: float) -> Layout:
    jpeg = _frame(video, duration * 0.3, "drawgrid=w=iw/10:h=ih/10:t=1:c=red@0.6")
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpeg).decode()}},
            {"type": "text", "text": PROMPT},
        ]}],
        output_format=LayoutAnswer,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    answer = response.parsed_output
    if answer is None:
        log.warning("Couldn't read the layout (stop reason %s); using the full frame.", response.stop_reason)
        return Layout(None, None)
    player = _to_crop(answer.player, width, height) if answer.player_found else None
    if player:
        player = tidy_player(player, width, height)
        if player.w >= width * 0.97 and player.h >= height * 0.97:
            player = None  # the recording is already just the stream
    chat = _to_crop(answer.chat, width, height) if answer.chat_found else None
    return Layout(player, chat, answer.notes)


def save_preview(video: Path, at: float, layout: Layout, out: Path) -> Path:
    """Frame with the player (green) and chat (blue) boxes drawn on, for checking the crop."""
    boxes = []
    if layout.player:
        p = layout.player
        boxes.append(f"drawbox=x={p.x}:y={p.y}:w={p.w}:h={p.h}:color=lime:t=6")
    if layout.chat:
        c = layout.chat
        boxes.append(f"drawbox=x={c.x}:y={c.y}:w={c.w}:h={c.h}:color=cyan:t=6")
    vf = ",".join(boxes + ["scale='min(1280,iw)':-2"])
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video),
         "-frames:v", "1", "-vf", vf, str(out)],
        check=True,
    )
    return out
