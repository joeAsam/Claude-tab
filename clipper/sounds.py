"""Sound-effect and music library.

On first run, a starter set of royalty-free sound effects and music loops is
synthesized into assets/sfx and assets/music. Drop your own .mp3/.wav/.ogg/.m4a
files into those folders and the agent will see and use them too. Name files
descriptively (e.g. "vine_boom.mp3", "hype_trap_beat.mp3"): the agent picks
sounds by name.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SR = 44100
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac"}


# ---- library -------------------------------------------------------------

def list_sounds(folder: Path) -> dict[str, Path]:
    """Map of sound name (file stem) -> path."""
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in AUDIO_EXTS}


def ensure_default_assets(assets_dir: Path) -> None:
    """Synthesize the starter sounds and music that aren't on disk yet."""
    sfx_dir, music_dir = assets_dir / "sfx", assets_dir / "music"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for folder, makers in ((sfx_dir, SFX), (music_dir, MUSIC)):
        for name, make in makers.items():
            path = folder / f"{name}.wav"
            if not path.exists():
                _write_wav(path, make())
                made += 1
    if made:
        log.info("Created %d starter sounds in %s/", made, assets_dir)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    peak = np.max(np.abs(samples)) or 1.0
    data = (samples / peak * 0.9 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


# ---- synthesis helpers ---------------------------------------------------

def _t(seconds: float) -> np.ndarray:
    return np.arange(int(SR * seconds)) / SR


def _tone(freq, seconds: float, harmonics=(1.0,)) -> np.ndarray:
    """Tone with a constant or time-varying frequency (array), phase-continuous."""
    t = _t(seconds)
    f = np.broadcast_to(np.asarray(freq, dtype=float), t.shape)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return sum(a * np.sin((i + 1) * phase) for i, a in enumerate(harmonics))


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, int(SR * seconds))


def _lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """One-pole low-pass. Smaller alpha = darker."""
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def _at(total: np.ndarray, start: float, sound: np.ndarray) -> None:
    i = int(start * SR)
    end = min(i + len(sound), len(total))
    total[i:end] += sound[: end - i]


def _env(seconds: float, attack: float = 0.005, decay: float = 5.0) -> np.ndarray:
    t = _t(seconds)
    return np.minimum(t / attack, 1.0) * np.exp(-decay * t)


# ---- sound effects -------------------------------------------------------

def boing() -> np.ndarray:
    d = 0.9
    t = _t(d)
    freq = 170 + 90 * np.sin(2 * np.pi * 11 * t) * np.exp(-2 * t) + 120 * t
    return _tone(freq, d, (1, 0.3)) * _env(d, decay=3.5)


def pop() -> np.ndarray:
    d = 0.12
    t = _t(d)
    return _tone(1400 * np.exp(-25 * t) + 300, d) * _env(d, 0.001, 35)


def air_horn() -> np.ndarray:
    out = np.zeros(int(SR * 1.6))
    for start, length in ((0, 0.22), (0.3, 0.22), (0.6, 0.9)):
        blast = sum(_tone(f, length, (1, 0.8, 0.6, 0.45, 0.3, 0.2)) for f in (466, 587, 698))
        blast *= np.minimum(_t(length) / 0.02, 1) * np.minimum((length - _t(length)) / 0.05, 1)
        _at(out, start, blast)
    return out


def record_scratch() -> np.ndarray:
    d = 0.55
    t = _t(d)
    rate = np.abs(np.sin(2 * np.pi * 2.5 * t))
    return (_lowpass(_noise(d, 3), 0.25) * 0.6 + _tone(250 + 900 * rate, d) * 0.4) * rate * _env(d, 0.01, 2.5)


def ding() -> np.ndarray:
    d = 1.4
    return _tone(1318, d, (1, 0.0, 0.25, 0.0, 0.1)) * _env(d, 0.002, 3.0)


def whoosh() -> np.ndarray:
    d = 0.8
    t = _t(d)
    swell = np.sin(np.pi * t / d) ** 2
    return _lowpass(_noise(d, 7), 0.08) * swell


def rimshot() -> np.ndarray:
    """Ba-dum-tss."""
    out = np.zeros(int(SR * 1.4))
    for start, f in ((0.0, 190), (0.2, 140)):
        _at(out, start, _tone(f * (1 + np.exp(-30 * _t(0.3))), 0.3) * _env(0.3, 0.002, 12))
    hiss = _noise(1.0, 9) - _lowpass(_noise(1.0, 9), 0.3)  # crude high-pass = cymbal
    _at(out, 0.42, hiss * _env(1.0, 0.002, 4) * 0.8)
    return out


def sad_trombone() -> np.ndarray:
    notes = ((293.7, 0.45), (277.2, 0.45), (261.6, 0.45), (246.9, 1.3))
    parts = []
    for i, (f, d) in enumerate(notes):
        t = _t(d)
        wobble = 1 + (0.025 * np.sin(2 * np.pi * 6 * t) if i == 3 else 0)
        env = np.minimum(t / 0.04, 1) * np.minimum((d - t) / 0.08, 1)
        parts.append(_tone(f * wobble, d, (1, 0.7, 0.5, 0.35, 0.2)) * env)
    return np.concatenate(parts)


def crickets() -> np.ndarray:
    """Awkward silence."""
    d = 2.5
    t = _t(d)
    chirp_on = (np.sin(2 * np.pi * 28 * t) > 0) & (np.sin(2 * np.pi * 1.6 * t) > 0.4)
    return _tone(4400, d) * chirp_on * 0.6


def dramatic() -> np.ndarray:
    """Dun dun DUNNN."""
    out = np.zeros(int(SR * 2.6))
    for start, d in ((0, 0.35), (0.45, 0.35), (0.9, 1.6)):
        chord = sum(_tone(f, d, (1, 0.5, 0.3, 0.2)) for f in (110, 130.8, 164.8))
        _at(out, start, chord * _env(d, 0.01, 1.5 if d > 1 else 4))
    return out


SFX = {
    "boing": boing,
    "pop": pop,
    "air_horn": air_horn,
    "record_scratch": record_scratch,
    "ding": ding,
    "whoosh": whoosh,
    "rimshot_ba_dum_tss": rimshot,
    "sad_trombone": sad_trombone,
    "crickets_awkward_silence": crickets,
    "dramatic_dun_dun_dun": dramatic,
}


# ---- music loops ---------------------------------------------------------

def _kick() -> np.ndarray:
    d = 0.35
    t = _t(d)
    return _tone(48 + 110 * np.exp(-28 * t), d) * _env(d, 0.001, 9)


def _hat(seed: int) -> np.ndarray:
    d = 0.06
    n = _noise(d, seed)
    return (n - _lowpass(n, 0.4)) * _env(d, 0.001, 70) * 0.35


def _pluck(freq: float, d: float = 0.3, bright: float = 0.5) -> np.ndarray:
    return _tone(freq, d, (1, bright, bright / 2, bright / 4)) * _env(d, 0.003, 9)


def _pad(freqs, d: float) -> np.ndarray:
    t = _t(d)
    env = np.minimum(t / 0.4, 1) * np.minimum((d - t) / 0.4, 1)
    return sum(_tone(f, d, (1, 0.15)) for f in freqs) * env


def hype_beat() -> np.ndarray:
    """140 BPM: punchy kick, off-beat hats, driving bass."""
    beat = 60 / 140
    bars = 8
    out = np.zeros(int(SR * beat * 4 * bars))
    roots = (55.0, 55.0, 65.4, 49.0)
    for b in range(bars * 4):
        t0 = b * beat
        _at(out, t0, _kick())
        _at(out, t0 + beat / 2, _hat(b))
        root = roots[(b // 4) % 4]
        for e in range(2):
            _at(out, t0 + e * beat / 2, _tone(root, beat / 2, (1, 0.6, 0.3)) * _env(beat / 2, 0.003, 6) * 0.5)
    return out


def chill_lofi() -> np.ndarray:
    """80 BPM: soft jazzy chords, gentle kick, tape hiss."""
    beat = 60 / 80
    chords = ((220, 261.6, 329.6, 392), (174.6, 220, 261.6, 329.6),
              (261.6, 329.6, 392, 493.9), (196, 246.9, 293.7, 349.2))
    bars = 8
    out = np.zeros(int(SR * beat * 4 * bars))
    for bar in range(bars):
        t0 = bar * beat * 4
        _at(out, t0, _pad(chords[bar % 4], beat * 4) * 0.35)
        for b in (0, 2.5):
            _at(out, t0 + b * beat, _kick() * 0.6)
        for b in range(8):
            _at(out, t0 + b * beat / 2, _hat(bar * 8 + b) * 0.5)
    out += _lowpass(_noise(len(out) / SR, 11), 0.2) * 0.02
    return out


def goofy_bounce() -> np.ndarray:
    """120 BPM oom-pah with a bouncy plucked melody."""
    beat = 0.5
    melody = (523.3, 659.3, 784.0, 659.3, 587.3, 523.3, 440.0, 523.3,
              587.3, 698.5, 880.0, 698.5, 659.3, 587.3, 523.3, 392.0)
    bass = (130.8, 196.0, 174.6, 196.0)
    bars = 8
    out = np.zeros(int(SR * beat * 4 * bars))
    for b in range(bars * 4):
        t0 = b * beat
        _at(out, t0, _tone(bass[(b // 4) % 4] / (1 if b % 2 else 2), 0.25, (1, 0.5, 0.25)) * _env(0.25, 0.005, 10) * 0.6)
        for e in range(2):
            step = (b * 2 + e) % len(melody)
            if step % 4 != 3:  # leave little gaps so it bounces
                _at(out, t0 + e * beat / 2, _pluck(melody[step], 0.22, 0.6) * 0.45)
    return out


def suspense_drone() -> np.ndarray:
    """Tense low drone with a slow heartbeat: for build-ups and clutch moments."""
    d = 16.0
    t = _t(d)
    drone = _tone(55, d, (1, 0.5, 0.2)) + 0.5 * _tone(58.3, d, (1, 0.3))  # dissonant minor second
    drone *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t) ** 2
    out = drone * 0.4
    for i in range(int(d / 0.9)):
        _at(out, i * 0.9, _kick() * 0.7)
        _at(out, i * 0.9 + 0.22, _kick() * 0.45)
    return out


MUSIC = {
    "hype_beat": hype_beat,
    "chill_lofi": chill_lofi,
    "goofy_bounce": goofy_bounce,
    "suspense_drone": suspense_drone,
}
