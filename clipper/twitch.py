"""Twitch Helix API (статус эфира, пользователи, клипы) + запасная проверка эфира через streamlink."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import httpx

log = logging.getLogger(__name__)

HELIX = "https://api.twitch.tv/helix"


class TwitchAPI:
    def __init__(self, client_id: str, client_secret: str, streamlink: list[str] | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.streamlink = streamlink or ["streamlink"]
        self._token: str | None = None
        self._token_exp = 0.0
        self.http = httpx.AsyncClient(timeout=20)

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def _auth(self) -> str:
        if self._token and time.time() < self._token_exp - 300:
            return self._token
        r = await self.http.post("https://id.twitch.tv/oauth2/token", params={
            "client_id": self.client_id, "client_secret": self.client_secret,
            "grant_type": "client_credentials"})
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._token_exp = time.time() + data.get("expires_in", 3600)
        return self._token

    async def _get(self, path: str, params) -> list[dict]:
        for attempt in range(2):
            token = await self._auth()
            r = await self.http.get(f"{HELIX}/{path}", params=params,
                                    headers={"Client-Id": self.client_id, "Authorization": f"Bearer {token}"})
            if r.status_code == 401 and attempt == 0:
                self._token = None
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        return []

    async def user(self, login: str) -> dict | None:
        data = await self._get("users", {"login": login})
        return data[0] if data else None

    async def live_info(self, logins: list[str]) -> dict[str, dict]:
        """Каналы, которые сейчас в эфире: {логин: {"game": категория Twitch, "title": название стрима}}."""
        if not logins:
            return {}
        if not self.enabled:
            results = await asyncio.gather(*(self._streamlink_live(l) for l in logins))
            return {l: info for l, info in zip(logins, results) if info is not None}
        live: dict[str, dict] = {}
        for i in range(0, len(logins), 100):
            data = await self._get("streams", [("user_login", l) for l in logins[i:i + 100]])
            for s in data:
                if s.get("type") == "live":
                    live[s["user_login"].lower()] = {"game": s.get("game_name") or "", "title": s.get("title") or ""}
        return live

    async def live_logins(self, logins: list[str]) -> set[str]:
        return set(await self.live_info(logins))

    async def _streamlink_live(self, login: str) -> dict | None:
        """None — не в эфире; иначе метаданные стрима (streamlink отдаёт категорию без ключей Twitch)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.streamlink, "--json", f"https://twitch.tv/{login}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        except OSError:
            return None
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), 40)
            data = json.loads(out or b"{}")
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        except ValueError:
            return None
        if not data.get("streams"):
            return None
        meta = data.get("metadata") or {}
        return {"game": meta.get("category") or "", "title": meta.get("title") or ""}

    async def recent_clips(self, broadcaster_id: str, since: float) -> list[dict]:
        started = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return await self._get("clips", {"broadcaster_id": broadcaster_id, "started_at": started, "first": 50})


def parse_ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
