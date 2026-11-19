"""Grabs still frames from the buffer so Claude can see what's on screen."""

from __future__ import annotations

import base64
import subprocess

from .recorder import StreamRecorder


def grab_frame(recorder: StreamRecorder, t: float, width: int = 640) -> bytes | None:
    """JPEG of the frame at wall-clock time t, or None if it isn't in the buffer."""
    seg = next((s for s in recorder.segments() if s.start <= t < s.end), None)
    if not seg:
        return None
    # Seek after -i (decode up to the frame): exact, and works whatever the keyframe spacing.
    # Segments are short, so this stays fast.
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(seg.path), "-ss", f"{max(t - seg.start, 0):.2f}",
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "5",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True, timeout=30,
    )
    return result.stdout or None


def image_block(jpeg: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpeg).decode()},
    }
