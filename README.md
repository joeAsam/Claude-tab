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

## Clipping a past stream (VOD)

Download a past stream and clip it in one command:

```bash
# First hour of a YouTube stream (video + chat replay), vertical clips
python -m clipper "https://youtu.be/VIDEO_ID" --first 60 --vertical

# The whole VOD
python -m clipper "https://youtu.be/VIDEO_ID" --vod
```

The video goes to `downloads/`. For YouTube streams the **live chat replay** is downloaded too, so the clipper gets real chat reactions with timestamps (laughing, W/L, "clip it", channel emotes like `_kaiLUL` read as LUL), exactly as on a live stream. It then runs like recording mode below. A replay saved next to any video file as `<name>.live_chat.json` is picked up automatically, or pass one with `--chat-replay`.

yt-dlp needs a JavaScript runtime for YouTube (install [Deno](https://deno.com), or Node and add `--js-runtimes node` to a yt-dlp config file). YouTube sometimes blocks downloads from cloud servers ("Sign in to confirm you're not a bot"); from a home connection it normally works.

## Clipping a recording (e.g. a screen recording)

Pass a video file instead of a link:

```bash
python -m clipper my_screen_recording.mp4 --vertical
```

1. **Finds the stream on your screen**: Claude looks at a frame and finds the stream player and the chat panel, so clips contain only the stream, not your browser, desktop or taskbar. A check image with both boxes drawn on (`buffer/<name>/layout_preview.jpg`) is saved; if a box is off, set it yourself with `--crop W:H:X:Y` (and `--chat-crop`), or use `--no-crop` if the file is already just the stream.
2. **Analyzes the whole recording at once**: transcript, loudness, on-screen action, and how fast the on-screen chat is scrolling (a stand-in for chat speed, since chat in a video isn't text).
3. **Ranks the best moments** and has Claude review them one by one. Claude can **read the chat panel** from the recording to see how viewers reacted, and look at frames from the stream.
4. **Final pass**: Claude reads the full transcript for stories, hot takes and other quieter moments the signals can't hear.
5. Clips are cut and edited exactly as in live mode (hook, captions, sound effects, music) and named after the recording and the time they start, e.g. `my_screen_recording_12m34s_jump-scare.mp4`.

| Option | Default | What it does |
|---|---|---|
| `--max-clips` | 10 | Stop after this many clips |
| `--max-reviews` | 15 | Most candidate moments Claude reviews (each review costs API usage) |
| `--crop` / `--chat-crop` | found automatically | Player / chat position in pixels, `W:H:X:Y` |
| `--no-crop` | off | The file is already just the stream |

Tips: record at 1080p if you can (clips can only be as sharp as the recording), and turn off notifications while recording, since anything inside the stream player ends up in the clips.

## Posting daily through Buffer

Finished clips can go straight into your [Buffer](https://buffer.com) queue. Buffer then posts them at your posting-schedule times, so **one slot a day in each channel's Posting Schedule = one post a day**.

One-time setup:
1. In Buffer, connect your channels (Channels → Connect a channel) and give each one a posting time for **every** day of the week (channel → Settings → Posting Schedule).
2. Create a Buffer API key (Settings → API) and set it as `BUFFER_API_KEY`.
3. Buffer has no upload: it fetches each video from a public link when the post goes out. Create a free [Cloudinary](https://cloudinary.com) account and set `CLOUDINARY_URL` to the "API environment variable" shown on its dashboard.

```bash
# Clip a stream and queue every finished clip on TikTok and YouTube Shorts
python -m clipper https://twitch.tv/somechannel --vertical --post-to-buffer

# TikTok only
python -m clipper my_recording.mp4 --vertical --post-to-buffer --buffer-channels tiktok

# Queue clips you already made (skips ones already queued; --dry-run to preview)
python -m clipper.publish
```

Claude writes a posting caption for every clip (one punchy line plus hashtags, kept in `clips/clips.jsonl`). On YouTube the caption becomes the description (with `#Shorts` added) and its first line, without hashtags, becomes the title; category 20 (Gaming) by default, change it with `--youtube-category`. YouTube only counts **vertical clips of 3 minutes or less** as Shorts, so use `--vertical` and keep `--max-clip 180` when posting there; longer or horizontal clips are posted as regular videos (the clipper warns when that happens). Queued posts are logged in `clips/posted.jsonl` so nothing is posted twice. Channels can be named by service (`tiktok`, `youtube`), channel name or ID.

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
| `clipper/filesource.py` | Recording mode: crops a video file to the player and splits it like a live buffer |
| `clipper/layout.py` | Finds the stream player and chat panel in a screen recording |
| `clipper/download.py` | Downloads past streams (VODs) and their chat replays with yt-dlp |
| `clipper/publish.py` | Uploads clips to Cloudinary and queues them on Buffer (also a standalone command) |
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
