# Claude Stream Clipper

An AI agent that watches a **live** stream and clips the best moments as they happen.

```
streamlink ─▶ ffmpeg ─▶ rolling buffer of 10s segments (last 15 min)
                               │
Twitch chat ─▶ spike detector ─┤──▶ Claude agent ──▶ clips/*.mp4
(optional) whisper transcript ─┘     (tools: chat, transcript, create_clip)
```

1. **Record**: the stream is recorded continuously into a rolling buffer of short segments, so moments can be clipped *after* they happen.
2. **Detect**: chat is read anonymously. When it suddenly gets much busier than usual (people spamming "LMAO", "CLIP IT", emotes), a review starts.
3. **Decide**: Claude looks at chat activity, the actual messages and, optionally, a transcript of what the streamer said. It decides whether the moment is worth clipping, picks the start and end (including the setup before chat reacted), and writes a title.
4. **Cut**: ffmpeg cuts a frame-accurate `.mp4`, optionally cropped to 9:16 for Shorts, TikTok or Reels. Each clip is also logged to `clips/clips.jsonl`.

## Setup

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # from https://console.anthropic.com
```

Optional: to let the agent *hear* the stream, install `pip install faster-whisper` and pass `--transcribe`.

## Run

```bash
# Twitch (chat spikes trigger reviews automatically)
python -m clipper https://twitch.tv/somechannel

# Vertical clips for Shorts/TikTok, with speech transcription
python -m clipper https://twitch.tv/somechannel --vertical --transcribe

# YouTube/Kick/etc.: no chat reader yet, so review on a timer + transcript
python -m clipper https://youtube.com/watch?v=LIVE_ID --transcribe --review-minutes 2
```

Press `Ctrl+C` to stop. Clips are saved in `clips/`.

| Option | Default | What it does |
|---|---|---|
| `--vertical` | off | Crop clips to 1080×1920 |
| `--transcribe [size]` | off | Speech-to-text with faster-whisper (`tiny`, `base`, `small`, ...) |
| `--spike-ratio` | 2.5 | How many times busier than usual chat must be to trigger a review |
| `--min-rate` | 0.5 | Minimum chat messages per second to trigger (filters out tiny streams' noise) |
| `--review-minutes` | 0 | Also review on a timer, even without a spike |
| `--cooldown` | 45 | Minimum seconds between reviews |
| `--buffer-minutes` | 15 | How much recent video to keep on disk |
| `--quality` | best | streamlink quality (`720p60`, `480p`, ...) |

Set `CLIPPER_MODEL` to use a different Claude model (default `claude-opus-5-5`).

## Code layout

| File | Role |
|---|---|
| `clipper/recorder.py` | streamlink → ffmpeg segment recorder, rolling buffer, auto-reconnect |
| `clipper/chat.py` | Anonymous Twitch IRC reader + chat-rate stats |
| `clipper/transcriber.py` | Optional faster-whisper transcription of segments |
| `clipper/clips.py` | Cuts clips out of the buffer with ffmpeg |
| `clipper/agent.py` | The Claude agent: its tools and system prompt |
| `clipper/__main__.py` | CLI and the trigger loop |

## Notes

- Only clip streams you have the right to repost (many clipping campaigns grant this; check the streamer's terms).
- Each review is one short Claude conversation, so cost scales with how often chat spikes. Raise `--spike-ratio` or `--cooldown` to review less often.
