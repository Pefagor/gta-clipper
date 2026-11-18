"""Чтение чата Twitch через анонимное IRC-подключение (одно на все каналы)."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

HOST, PORT = "irc.chat.twitch.tv", 6667


BOTS = {"nightbot", "moobot", "streamelements", "streamlabs", "fossabot", "wizebot", "sery_bot", "botrixoficial",
        "soundalerts", "kofistreambot", "commanderroot", "pokemoncommunitygame", "own3d", "blerp", "deepbot",
        "phantombot", "coebot", "vivbot", "stay_hydrated_bot", "streamholics", "creatisbot", "dixperbro"}


def is_noise(text: str, user: str = "") -> bool:
    """Сообщения, которые не реакция зрителей: команды ботам, ответы ботов, «+» в розыгрышах.
    Без этого чат-игры (!язомби, !elo…) давали ложные «всплески»: до четверти чата — команды."""
    t = text.strip()
    u = user.lower()
    if u in BOTS or (u.endswith("bot") and len(u) > 4):
        return True
    return t.startswith("!") or t in ("+", "++", "+++", "+1", "-", "1", "2")


def parse_privmsg(line: str) -> tuple[str, str, float, str] | None:
    """(канал, текст, время, ник автора) из строки IRC, либо None."""
    tags = {}
    if line.startswith("@"):
        raw_tags, _, line = line[1:].partition(" ")
        for kv in raw_tags.split(";"):
            k, _, v = kv.partition("=")
            tags[k] = v
    parts = line.split(" ", 3)
    if len(parts) < 4 or parts[1] != "PRIVMSG":
        return None
    channel = parts[2].lstrip("#").lower()
    text = parts[3][1:] if parts[3].startswith(":") else parts[3]
    try:
        ts = int(tags.get("tmi-sent-ts", "")) / 1000.0
    except ValueError:
        ts = time.time()
    user = parts[0].lstrip(":").split("!", 1)[0].lower()
    return channel, text, ts, user


class ChatHub:
    def __init__(self):
        self.messages: dict[str, deque[tuple[float, str]]] = {}
        self.log_files: dict[str, Path] = {}
        self.wanted: set[str] = set()
        self.noise: dict[str, int] = {}  # сколько отброшено команд/ботов (для статистики)
        self._joined: set[str] = set()
        self._writer: asyncio.StreamWriter | None = None
        self.connected = False

    def history(self, login: str) -> deque:
        return self.messages.setdefault(login, deque(maxlen=50000))

    def set_log(self, login: str, path: Path | None) -> None:
        if path is None:
            self.log_files.pop(login, None)
        else:
            self.log_files[login] = path

    async def join(self, login: str) -> None:
        login = login.lower()
        self.wanted.add(login)
        self.history(login)
        if self._writer and login not in self._joined:
            await self._send(f"JOIN #{login}")
            self._joined.add(login)

    async def part(self, login: str) -> None:
        login = login.lower()
        self.wanted.discard(login)
        if self._writer and login in self._joined:
            await self._send(f"PART #{login}")
        self._joined.discard(login)

    async def _send(self, line: str) -> None:
        if not self._writer:
            return
        try:
            self._writer.write((line + "\r\n").encode("utf-8"))
            await self._writer.drain()
        except (OSError, RuntimeError) as e:  # соединение умерло — run() переподключится и сделает JOIN заново
            log.warning("чат: ошибка отправки (%s)", e)
            self._writer.close()

    async def run(self) -> None:
        backoff = 2
        while True:
            try:
                reader, writer = await asyncio.open_connection(HOST, PORT)
                self._writer = writer
                await self._send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                await self._send("PASS SCHMOOPIIE")
                await self._send(f"NICK justinfan{random.randint(10000, 99999)}")
                self._joined.clear()
                for login in sorted(self.wanted):
                    await self._send(f"JOIN #{login}")
                    self._joined.add(login)
                self.connected = True
                backoff = 2
                log.info("💬 чат Twitch подключён")
                while True:
                    raw = await asyncio.wait_for(reader.readline(), timeout=360)
                    if not raw:
                        raise ConnectionError("соединение закрыто")
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("PING"):
                        await self._send("PONG" + line[4:])
                        continue
                    if " RECONNECT" in line[:80]:
                        raise ConnectionError("сервер попросил переподключиться")
                    msg = parse_privmsg(line)
                    if msg:
                        self._store(*msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — любая ошибка сети: переподключаемся
                log.warning("чат: %s, переподключение через %d с", e, backoff)
            finally:
                self.connected = False
                if self._writer:
                    self._writer.close()
                self._writer = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _store(self, channel: str, text: str, ts: float, user: str = "") -> None:
        if is_noise(text, user):
            self.noise[channel] = self.noise.get(channel, 0) + 1
            return
        self.history(channel).append((ts, text))
        path = self.log_files.get(channel)
        if path:
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": ts, "m": text}, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def window(self, login: str, start: float, end: float) -> list[tuple[float, str]]:
        return [(t, m) for t, m in self.history(login) if start <= t <= end]
