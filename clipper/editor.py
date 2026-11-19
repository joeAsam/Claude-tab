"""Final render: burn in captions and mix in sound effects and background music."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

AUDIO_FMT = "aformat=sample_rates=48000:channel_layouts=stereo"


@dataclass(frozen=True)
class SoundCue:
    path: Path
    at: float  # seconds from the start of the clip
    volume: float = 1.0


def probe(path: Path) -> dict:
    """Duration, video size and whether the file has audio."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    return {
        "duration": float(info["format"]["duration"]),
        "width": video.get("width", 0),
        "height": video.get("height", 0),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def render(
    raw: Path,
    out: Path,
    captions: Path | None = None,
    sound_effects: list[SoundCue] = (),
    music: Path | None = None,
    music_volume: float = 0.15,
) -> Path:
    info = probe(raw)
    duration = info["duration"]

    inputs = ["-i", str(raw.resolve())]
    filters: list[str] = []
    mix: list[str] = []

    # Voice (the stream's own audio). Make silence if the stream had none.
    if info["has_audio"]:
        voice_src = "[0:a]"
    else:
        inputs += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        voice_src = "[1:a]"
    next_input = sum(1 for a in inputs if a == "-i")

    if music:
        filters.append(f"{voice_src}{AUDIO_FMT},asplit=2[voice][duckkey]")
        inputs += ["-stream_loop", "-1", "-i", str(music.resolve())]
        fade_out = max(duration - 2.5, 0)
        filters.append(
            f"[{next_input}:a]{AUDIO_FMT},atrim=0:{duration:.3f},volume={music_volume:.3f},"
            f"afade=t=in:d=1.5,afade=t=out:st={fade_out:.3f}:d=2.5[musicraw]"
        )
        # Duck the music whenever someone is talking.
        filters.append("[musicraw][duckkey]sidechaincompress=threshold=0.02:ratio=8:attack=40:release=500[music]")
        mix += ["[voice]", "[music]"]
        next_input += 1
    else:
        filters.append(f"{voice_src}{AUDIO_FMT}[voice]")
        mix.append("[voice]")

    for i, cue in enumerate(sound_effects):
        inputs += ["-i", str(cue.path.resolve())]
        delay_ms = int(max(cue.at, 0) * 1000)
        filters.append(f"[{next_input}:a]{AUDIO_FMT},volume={cue.volume:.3f},adelay={delay_ms}:all=1[sfx{i}]")
        mix.append(f"[sfx{i}]")
        next_input += 1

    filters.append(
        f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0,alimiter=limit=0.95[aout]"
    )

    if captions:
        # Run from the captions folder so the filter path needs no escaping.
        filters.append(f"[0:v]ass={captions.name}[vout]")
        video_map, video_codec = "[vout]", ["-c:v", "libx264", "-preset", "medium", "-crf", "19"]
    else:
        video_map, video_codec = "0:v", ["-c:v", "copy"]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", video_map, "-map", "[aout]",
        *video_codec,
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(out.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=captions.parent if captions else None)
    return out
