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
