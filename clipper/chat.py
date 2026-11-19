"""Reads Twitch chat anonymously and tracks how fast it is moving.

Chat velocity is the strongest signal that something clip-worthy just happened:
viewers spam "LMAO", "CLIP IT", emotes, etc. within seconds of a big moment.
"""

from __future__ import annotations

import logging
import random
import re
import socket
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667
PRIVMSG = re.compile(r"^:(\w+)!\S+ PRIVMSG #\w+ :(.*)$")


@dataclass(frozen=True)
class ChatMessage:
    ts: float
    user: str
    text: str


class TwitchChat:
    def __init__(self, channel: str, keep_seconds: int = 1800) -> None:
        self.channel = channel.lower().lstrip("#")
        self.keep_seconds = keep_seconds
        self._messages: deque[ChatMessage] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.listening_since: float | None = None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="chat").start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_and_read()
            except OSError as exc:
                log.warning("Chat connection lost (%s); reconnecting", exc)
            self._stop.wait(5)

    def _connect_and_read(self) -> None:
        # "justinfanNNNN" is Twitch's anonymous read-only login: no token needed.
        nick = f"justinfan{random.randint(10000, 99999)}"
        with socket.create_connection((IRC_HOST, IRC_PORT), timeout=300) as sock:
            sock.sendall(f"NICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
            log.info("Joined Twitch chat #%s", self.channel)
            if self.listening_since is None:
                self.listening_since = time.time()
            buf = b""
            while not self._stop.is_set():
                data = sock.recv(4096)
                if not data:
                    return
                buf += data
                *lines, buf = buf.split(b"\r\n")
                for raw in lines:
                    line = raw.decode("utf-8", errors="replace")
                    if line.startswith("PING"):
                        sock.sendall(line.replace("PING", "PONG", 1).encode() + b"\r\n")
                        continue
                    m = PRIVMSG.match(line)
                    if m:
                        self._add(ChatMessage(time.time(), m.group(1), m.group(2)))

    def _add(self, msg: ChatMessage) -> None:
        if self.listening_since is None:
            self.listening_since = msg.ts
        with self._lock:
            self._messages.append(msg)
            cutoff = msg.ts - self.keep_seconds
            while self._messages and self._messages[0].ts < cutoff:
                self._messages.popleft()

    # ---- queries ---------------------------------------------------------

    def messages_between(self, start: float, end: float) -> list[ChatMessage]:
        with self._lock:
            return [m for m in self._messages if start <= m.ts < end]

    def rate(self, start: float, end: float) -> float:
        """Messages per second over [start, end)."""
        span = max(end - start, 1e-6)
        return len(self.messages_between(start, end)) / span

    def buckets(self, start: float, end: float, bucket_seconds: int = 10) -> list[dict]:
        """Per-bucket message counts and the most repeated phrases."""
        out = []
        t = start
        while t < end:
            msgs = self.messages_between(t, min(t + bucket_seconds, end))
            common = Counter(m.text.strip().lower() for m in msgs).most_common(3)
            out.append({"start": t, "count": len(msgs), "top": [f"{txt} x{n}" for txt, n in common]})
            t += bucket_seconds
        return out


class ReplayChat(TwitchChat):
    """Chat from a saved replay (e.g. a YouTube live chat replay) on the clip timeline.
    Offers the same queries as live Twitch chat, so the rest of the clipper can't tell
    the difference."""

    def __init__(self, messages: list[ChatMessage], timeline_start: float) -> None:
        super().__init__("replay", keep_seconds=10**9)
        self._messages = deque(sorted(messages, key=lambda m: m.ts))
        self.listening_since = timeline_start

    def start(self) -> None:  # nothing to connect to
        pass


def load_youtube_live_chat(path, timeline_start: float, max_seconds: float | None = None) -> ReplayChat:
    """Parse yt-dlp's <video>.live_chat.json (one JSON object per line)."""
    import json

    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            replay = entry.get("replayChatItemAction", {})
            offset_ms = replay.get("videoOffsetTimeMsec", entry.get("videoOffsetTimeMsec"))
            if offset_ms is None:
                continue
            offset = int(offset_ms) / 1000
            if offset < 0 or (max_seconds is not None and offset > max_seconds):
                continue
            for action in replay.get("actions", []):
                item = action.get("addChatItemAction", {}).get("item", {})
                renderer = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
                if not renderer:
                    continue
                text = "".join(_run_text(run) for run in renderer.get("message", {}).get("runs", []))
                if not text.strip():
                    continue
                user = renderer.get("authorName", {}).get("simpleText", "viewer").lstrip("@")
                messages.append(ChatMessage(timeline_start + offset, user, text.strip()))
    log.info("Loaded %d chat replay messages", len(messages))
    return ReplayChat(messages, timeline_start)


def _run_text(run: dict) -> str:
    if "text" in run:
        return run["text"]
    emoji = run.get("emoji", {})
    if not emoji:
        return ""
    if not emoji.get("isCustomEmoji") and emoji.get("emojiId"):
        return emoji["emojiId"]  # standard emoji: the id is the character itself, e.g. 😂
    shortcut = (emoji.get("shortcuts") or [""])[0].strip(":")
    # Channel emotes look like "_kaiLUL"; keep the readable part so "LUL", "W" etc. still register.
    return " " + re.sub(r"^_?[a-z0-9]+(?=[A-Z])", "", shortcut.lstrip("_")) + " "
