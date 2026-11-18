"""Чат YouTube-эфира через официальный YouTube Data API v3 (нужен бесплатный ключ YOUTUBE_API_KEY).

Квота API — 10 000 единиц в день на ключ, один запрос чата стоит 5 единиц. Поэтому чат забирается не чаще
раза в youtube.chat_poll_seconds (30 с по умолчанию ≈ 600 единиц в час эфира): у сообщений точное время,
детектор для YouTube-каналов просто смотрит с этим запаздыванием."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

import httpx

log = logging.getLogger(__name__)
API = "https://www.googleapis.com/youtube/v3"


def parse_rfc3339(s: str) -> float:
    s = s.replace("Z", "+00:00")
    if "." in s:  # 2026-11-18T12:00:00.123456789+00:00 — дробную часть обрезаем до микросекунд
        head, rest = s.split(".", 1)
        n = len(rest) - len(rest.lstrip("0123456789"))  # только ведущие цифры, без часового пояса
        frac, tz = rest[:n], rest[n:]
        s = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    return datetime.fromisoformat(s).timestamp()


class QuotaExceeded(Exception):
    pass


class YouTubeChat:
    def __init__(self, api_key: str, store, cfg: dict, on_alert=None):
        """store(login, text, ts) — куда складывать сообщения (общий ChatHub детектора)."""
        self.api_key = api_key
        self.store = store
        self.poll = float(cfg.get("chat_poll_seconds", 30))
        self.on_alert = on_alert
        self.http = httpx.AsyncClient(timeout=20, trust_env=False)
        self.tasks: dict[str, asyncio.Task] = {}
        self.quota_until = 0.0
        self.connected: dict[str, bool] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def lag(self, login: str) -> float:
        """На сколько секунд детектору смотреть назад, чтобы чат успел прийти."""
        return self.poll + 5 if self.connected.get(login) else 0.0

    def start(self, login: str, video_id: str) -> None:
        if not self.enabled or not video_id or login in self.tasks:
            return
        self.tasks[login] = asyncio.create_task(self._run(login, video_id), name=f"ytchat-{login}")

    async def stop(self, login: str) -> None:
        t = self.tasks.pop(login, None)
        self.connected.pop(login, None)
        if t:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)

    async def _get(self, path: str, params: dict) -> dict:
        r = await self.http.get(f"{API}/{path}", params={**params, "key": self.api_key})
        if r.status_code == 403 and "quota" in r.text.lower():
            raise QuotaExceeded(r.text[:200])
        if r.status_code >= 400:
            raise RuntimeError(f"YouTube API HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    async def live_chat_id(self, video_id: str) -> str | None:
        data = await self._get("videos", {"part": "liveStreamingDetails", "id": video_id})
        items = data.get("items") or []
        return (items[0].get("liveStreamingDetails") or {}).get("activeLiveChatId") if items else None

    async def _run(self, login: str, video_id: str) -> None:
        backoff = 10
        while True:
            if time.time() < self.quota_until:
                await asyncio.sleep(min(600, self.quota_until - time.time()))
                continue
            try:
                chat_id = await self.live_chat_id(video_id)
                if not chat_id:
                    log.info("💬 [%s] у эфира YouTube нет открытого чата — ищу моменты по звуку", login)
                    return
                log.info("💬 [%s] чат YouTube подключён (опрос раз в %.0f с)", login, self.poll)
                self.connected[login] = True
                token = None
                while True:
                    params = {"liveChatId": chat_id, "part": "snippet", "maxResults": 2000}
                    if token:
                        params["pageToken"] = token
                    data = await self._get("liveChat/messages", params)
                    for it in data.get("items") or []:
                        sn = it.get("snippet") or {}
                        text = sn.get("displayMessage")
                        if text and sn.get("publishedAt"):
                            self.store(login, text, parse_rfc3339(sn["publishedAt"]))
                    token = data.get("nextPageToken") or token
                    backoff = 10
                    await asyncio.sleep(max(self.poll, (data.get("pollingIntervalMillis") or 0) / 1000))
            except asyncio.CancelledError:
                raise
            except QuotaExceeded:
                self.connected[login] = False
                self.quota_until = time.time() + 3600
                log.warning("💬 квота YouTube API на сегодня исчерпана — чат YouTube-эфиров пауза на час, "
                            "моменты ищутся по звуку (увеличьте youtube.chat_poll_seconds)")
                if self.on_alert:
                    await self.on_alert("ytquota", "Квота YouTube API исчерпана: чат YouTube-эфиров на паузе, моменты "
                                                   "ищутся по звуку. Увеличьте youtube.chat_poll_seconds в config.")
            except Exception as e:  # noqa: BLE001
                self.connected[login] = False
                from .errors import explain
                log.warning("💬 [%s] чат YouTube: %s — повтор через %d с", login, explain(e), backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)
