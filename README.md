# Claude Stream Clipper

An AI agent that watches a **live** stream, clips the best moments as they happen, and edits them into
finished Shorts/TikTok-style videos with **synced captions, sound effects and background music**.

```
                         ┌─ audio loudness  (screams, rage, laughing fits, explosions)
streamlink ─▶ ffmpeg ─▶  ├─ visual change   (deaths, jump scares, chaos)        ┐
  rolling buffer         └─ live transcript (exclamations, stories, hot takes)  ├─▶ interest score ─▶ Claude agent
Twitch chat ──────────────── speed + reaction type (laughing/hype/shock/L/clip it) ┘                      │
                                                                                                         ▼
           investigates (signal timeline, chat, transcript, LOOKS at video frames) ─▶ rates it 1-10 ─▶ cut + edit
```

1. **Record**: the stream is recorded continuously into a rolling buffer of short segments, so moments can be clipped *after* they happen.
2. **Detect**: every few seconds the stream is scored on several signals at once, each compared with *this* stream's normal (so a calm chess stream and a screaming horror stream are both judged fairly):
   - **Chat** (Twitch): how much busier than usual, and *what kind* of reaction: laughing (KEKW, LMAO), hype (POG, W, 🔥), shock (???, WTF), rage/fail (L, 💀), wholesome (<3), and viewers typing "clip it".
   - **Audio**: sudden loudness, which usually means yelling, screaming, rage or an explosion.
   - **Video**: sudden big changes on screen, such as deaths, jump scares or chaotic action.
   - **Speech** (live transcript): the streamer exclaiming ("oh my god", "no way", "let's go", swearing). The transcript also lets Claude catch quiet but gripping moments like stories and hot takes.

   When the combined score passes `--sensitivity`, Claude is woken up with a briefing of what was flagged and why. This works on YouTube and Kick too: audio, video and speech don't need chat.
3. **Decide**: Claude investigates like an editor would. It reads the signal timeline to see how the moment built up and faded, reads chat and the transcript to understand what happened, and **looks at frames from the video**. It labels the moment (funny, rage, hype/clutch, fail, shock/jump scare, wholesome, story/hot take, intriguing, unique/weird, skill, drama) and rates it 1–10 on hook, standalone appeal, payoff, emotion and shareability. Anything below `--min-score` (default 7) is not clipped.
4. **Cut**: each moment type has its own cutting rules (a jump scare starts in the calm before, a story starts at its first sentence, a fail starts before the confident lead-in). Clips are **30 seconds to 4 minutes** (configurable), and both ends are nudged to the nearest pause in speech so nobody is cut off mid-sentence. They can be cropped to 9:16 for Shorts, TikTok or Reels.
5. **Edit**: Whisper transcribes the clip word by word. Claude reads what was said and when chat reacted (to the second), then:
   - puts a **hook title** at the top of the video for the whole clip (3–8 words that make people want to watch, without spoiling the ending), or your own fixed text with `--hook`;
   - burns in **captions synced to the speech**, a few words at a time, with the word being spoken highlighted;
   - drops **sound effects** right on the punchline, fail or big play (air horn, boing, sad trombone, rimshot, record scratch, crickets...);
   - adds **background music** that fits the mood, automatically turned down whenever someone talks.

   Each finished clip is logged to `clips/clips.jsonl` with its edits.

## Sound effects and music

On first run, a starter library of royalty-free sounds is synthesized into `assets/`:

| `assets/sfx/` | `assets/music/` |
|---|---|
| air_horn, boing, pop, ding, whoosh, record_scratch, rimshot_ba_dum_tss, sad_trombone, crickets_awkward_silence, dramatic_dun_dun_dun | hype_beat, chill_lofi, goofy_bounce, suspense_drone |

**Add your own** by dropping `.mp3`, `.wav`, `.ogg` or `.m4a` files into those folders. Claude chooses sounds **by file name**, so name them clearly (`vine_boom.mp3`, `emotional_piano.mp3`, `phonk_drift_beat.mp3`). Only use audio you have the rights to: platforms mute or take down clips with copyrighted music.

## Setup

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # from https://console.anthropic.com
```

The first clip downloads the Whisper speech model (about 150 MB for `base`).

## Run

```bash
# Twitch (chat spikes trigger reviews automatically)
python -m clipper https://twitch.tv/somechannel

# Vertical clips for Shorts/TikTok, 45s to 2 min, more accurate captions
python -m clipper https://twitch.tv/somechannel --vertical --min-clip 45 --max-clip 120 --whisper-model small

# Same fixed hook on every clip, and only the very best moments
python -m clipper https://twitch.tv/somechannel --vertical --hook "WAIT FOR IT" --min-score 8

# YouTube/Kick/etc.: moments are found from audio, video and speech
python -m clipper https://youtube.com/watch?v=LIVE_ID
```

Press `Ctrl+C` to stop. Clips are saved in `clips/`.

| Option | Default | What it does |
|---|---|---|
| `--vertical` | off | Crop clips to 1080×1920 |
| `--min-clip` / `--max-clip` | 30 / 240 | Shortest and longest clip, in seconds |
| `--min-score` | 7 | Only clip moments Claude rates at least this viral (1–10) |
| `--hook TEXT` / `--no-hook` | Claude writes one | Fixed hook title for every clip, or none |
| `--no-captions` / `--no-sfx` / `--no-music` | on | Turn off an editing feature |
| `--assets` | assets | Folder holding `sfx/` and `music/` |
| `--whisper-model` | base | Caption accuracy vs speed: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--sensitivity` | 3.0 | Interest score that wakes Claude; lower = more reviews (and more cost) |
| `--no-transcribe` | transcribing | Stop live transcription (saves CPU, but stories and hot takes get missed) |
| `--review-minutes` | 0 (5 without chat) | Also review on a timer, even if nothing triggers |
| `--cooldown` | 45 | Minimum seconds between reviews |
| `--buffer-minutes` | 15 | How much recent video to keep on disk (raised automatically to fit `--max-clip`) |
| `--quality` | best | streamlink quality (`720p60`, `480p`, ...) |

Set `CLIPPER_MODEL` to use a different Claude model (default `claude-opus-5-5`).

## Code layout

| File | Role |
|---|---|
| `clipper/recorder.py` | streamlink → ffmpeg segment recorder, rolling buffer, auto-reconnect |
| `clipper/chat.py` | Anonymous Twitch IRC reader + chat-rate stats |
| `clipper/signals.py` | Moment detector: audio, video, chat reactions and speech, scored against the stream's normal |
| `clipper/frames.py` | Grabs still frames so Claude can see the video |
| `clipper/transcriber.py` | Live faster-whisper transcription of segments |
| `clipper/clips.py` | Cuts clips out of the buffer with ffmpeg |
| `clipper/whisper.py` | Shared Whisper model, word-level timestamps |
| `clipper/captions.py` | Builds the synced, word-highlighted captions (.ass) |
| `clipper/sounds.py` | Sound library + synthesizer for the starter sounds and music |
| `clipper/editor.py` | Final render: burns in captions, mixes voice + sound effects + ducked music |
| `clipper/agent.py` | The Claude agent: its tools and system prompt |
| `clipper/__main__.py` | CLI and the trigger loop |

## Notes

- Only clip streams you have the right to repost (many clipping campaigns grant this; check the streamer's terms).
- Each review is one Claude conversation (with a few video frames), so cost scales with how often something triggers. Raise `--sensitivity` or `--cooldown` to review less often.
- Live transcription and signal analysis run continuously on your CPU. On a slow machine use `--whisper-model tiny` or `--no-transcribe`.
