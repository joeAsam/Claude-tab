"""Builds TikTok-style burned-in captions (.ass) from word timestamps, plus a hook title.

Words are shown a few at a time, and the word being spoken is highlighted,
so the captions stay locked to the speaker's voice.
"""

from __future__ import annotations

from pathlib import Path

from .whisper import Word

MAX_WORDS_PER_CHUNK = 3
MAX_CHUNK_SECONDS = 1.6
PAUSE_BREAK_SECONDS = 0.5
LINGER_SECONDS = 0.4  # keep the last words on screen briefly after speech stops

WHITE = "&H00FFFFFF"
HIGHLIGHT = "&H0000E5FF"  # ASS colours are &HAABBGGRR: this is a warm yellow
OUTLINE = "&H00000000"


def chunk_words(words: list[Word]) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current:
            too_many = len(current) >= MAX_WORDS_PER_CHUNK
            too_long = w.end - current[0].start > MAX_CHUNK_SECONDS
            paused = w.start - current[-1].end > PAUSE_BREAK_SECONDS
            sentence_end = current[-1].text[-1:] in ".!?"
            if too_many or too_long or paused or sentence_end:
                chunks.append(current)
                current = []
        current.append(w)
    if current:
        chunks.append(current)
    return chunks


def _ts(seconds: float) -> str:
    seconds = max(seconds, 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(
    words: list[Word],
    width: int,
    height: int,
    clip_duration: float,
    hook: str | None = None,
    uppercase: bool = True,
) -> str:
    vertical = height > width
    font_size = round(height * (0.045 if vertical else 0.075))
    margin_v = round(height * (0.27 if vertical else 0.08))  # vertical: clear of the app's UI
    outline = max(3, round(font_size * 0.09))
    hook_size = round(height * (0.038 if vertical else 0.062))
    hook_margin = round(height * (0.11 if vertical else 0.045))
    side = round(width * 0.06)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,{font_size},{WHITE},{WHITE},{OUTLINE},&H80000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,{side},{side},{margin_v},1
Style: Hook,Arial,{hook_size},&H00000000,&H00000000,{WHITE},&H00000000,-1,0,0,0,100,100,0,0,3,{round(hook_size * 0.35)},0,8,{side},{side},{hook_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if hook:
        # A fixed title on a white box at the top, on screen for the whole clip.
        events.append(f"Dialogue: 1,{_ts(0)},{_ts(clip_duration)},Hook,,0,0,0,,{_escape(hook)}")
    chunks = chunk_words(words)
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else clip_duration
        chunk_end = min(chunk[-1].end + LINGER_SECONDS, next_start, clip_duration)
        for wi, word in enumerate(chunk):
            start = word.start
            end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
            if end <= start:
                continue
            parts = []
            for k, w in enumerate(chunk):
                text = _escape(w.text.upper() if uppercase else w.text)
                parts.append(f"{{\\c{HIGHLIGHT}&}}{text}{{\\c{WHITE}&}}" if k == wi else text)
            pop = "{\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}" if wi == 0 else ""
            events.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Caption,,0,0,0,,{pop}{' '.join(parts)}")
    return header + "\n".join(events) + "\n"


def write_ass(
    words: list[Word], width: int, height: int, clip_duration: float, out: Path, hook: str | None = None
) -> Path:
    out.write_text(build_ass(words, width, height, clip_duration, hook), encoding="utf-8")
    return out
