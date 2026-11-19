# Claude Stream Clipper

An AI agent that watches a **live** stream, clips the best moments as they happen, and edits them into
finished Shorts/TikTok-style videos with **synced captions, sound effects and background music**.

```
streamlink ─▶ ffmpeg ─▶ rolling buffer of 10s segments
                               │
Twitch chat ─▶ spike detector ─┤──▶ Claude agent ──▶ create_clip ──▶ finish_clip ──▶ clips/*.mp4
(optional) live transcript ────┘   (chat, transcript)  (cut + word-     (captions, sound
                                                         level speech)   effects, music)
```

1. **Record**: the stream is recorded continuously into a rolling buffer of short segments, so moments can be clipped *after* they happen.
2. **Detect**: chat is read anonymously. When it suddenly gets much busier than usual (people spamming "LMAO", "CLIP IT", emotes), a review starts.
3. **Decide**: Claude looks at chat activity, the actual messages and, optionally, a transcript of what the streamer said. It decides whether the moment is worth clipping, picks the start and end (including the setup before chat reacted), and writes a title.
4. **Cut**: ffmpeg cuts a frame-accurate clip between **30 seconds and 4 minutes** long (configurable), optionally cropped to 9:16 for Shorts, TikTok or Reels.
5. **Edit**: Whisper transcribes the clip word by word. Claude reads what was said and when chat reacted (to the second), then:
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

# YouTube/Kick/etc.: no chat reader yet, so review on a timer + transcript
python -m clipper https://youtube.com/watch?v=LIVE_ID --transcribe --review-minutes 2
```

Press `Ctrl+C` to stop. Clips are saved in `clips/`.

| Option | Default | What it does |
|---|---|---|
| `--vertical` | off | Crop clips to 1080×1920 |
| `--min-clip` / `--max-clip` | 30 / 240 | Shortest and longest clip, in seconds |
| `--no-captions` / `--no-sfx` / `--no-music` | on | Turn off an editing feature |
| `--assets` | assets | Folder holding `sfx/` and `music/` |
| `--whisper-model` | base | Caption accuracy vs speed: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--transcribe` | off | Also transcribe live, so the agent can read what's said while deciding what to clip |
| `--spike-ratio` | 2.5 | How many times busier than usual chat must be to trigger a review |
| `--min-rate` | 0.5 | Minimum chat messages per second to trigger (filters out tiny streams' noise) |
| `--review-minutes` | 0 | Also review on a timer, even without a spike |
| `--cooldown` | 45 | Minimum seconds between reviews |
| `--buffer-minutes` | 15 | How much recent video to keep on disk (raised automatically to fit `--max-clip`) |
| `--quality` | best | streamlink quality (`720p60`, `480p`, ...) |

Set `CLIPPER_MODEL` to use a different Claude model (default `claude-opus-5-5`).

## Code layout

| File | Role |
|---|---|
| `clipper/recorder.py` | streamlink → ffmpeg segment recorder, rolling buffer, auto-reconnect |
| `clipper/chat.py` | Anonymous Twitch IRC reader + chat-rate stats |
| `clipper/transcriber.py` | Optional faster-whisper transcription of segments |
| `clipper/clips.py` | Cuts clips out of the buffer with ffmpeg |
| `clipper/whisper.py` | Shared Whisper model, word-level timestamps |
| `clipper/captions.py` | Builds the synced, word-highlighted captions (.ass) |
| `clipper/sounds.py` | Sound library + synthesizer for the starter sounds and music |
| `clipper/editor.py` | Final render: burns in captions, mixes voice + sound effects + ducked music |
| `clipper/agent.py` | The Claude agent: its tools and system prompt |
| `clipper/__main__.py` | CLI and the trigger loop |

## Notes

- Only clip streams you have the right to repost (many clipping campaigns grant this; check the streamer's terms).
- Each review is one short Claude conversation, so cost scales with how often chat spikes. Raise `--spike-ratio` or `--cooldown` to review less often.
