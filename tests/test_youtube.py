"""YouTube-эфиры: распознавание канала и чат через YouTube Data API (ответы Google подменены)."""
import asyncio

import httpx

from clipper.platforms import credit, game_from_category, live_url, parse_channel
from clipper.ytchat import YouTubeChat, parse_rfc3339


def test_parse_channel_and_urls():
    assert parse_channel("https://www.youtube.com/@LofiGirl")[:2] == ("youtube", "yt_lofigirl")
    assert parse_channel("stariy_bog") == ("twitch", "stariy_bog", "https://twitch.tv/stariy_bog")
    ch = {"login": "yt_lofigirl", "platform": "youtube", "url": "https://www.youtube.com/@LofiGirl",
          "display_name": "Lofi Girl"}
    assert live_url(ch) == "https://www.youtube.com/@LofiGirl/live"
    assert credit(ch, "yt_lofigirl") == "YouTube: Lofi Girl"
    assert game_from_category("Gaming", "youtube") == "" and game_from_category("Dota 2", "twitch") == "Dota 2"


def test_rfc3339():
    assert abs(parse_rfc3339("2026-11-18T12:00:00.123456789Z") - 1795003200.123456) < 1e-3
    assert parse_rfc3339("2026-11-18T12:00:00Z") == 1795003200.0


def test_chat_messages_reach_detector():
    got, calls = [], {"messages": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["key"] == "test-key"
        if req.url.path.endswith("/videos"):
            return httpx.Response(200, json={"items": [{"liveStreamingDetails": {"activeLiveChatId": "CHAT1"}}]})
        calls["messages"] += 1
        assert req.url.params["liveChatId"] == "CHAT1"
        if calls["messages"] == 1:
            return httpx.Response(200, json={"nextPageToken": "p2", "pollingIntervalMillis": 10, "items": [
                {"snippet": {"publishedAt": "2026-11-18T12:00:01Z", "displayMessage": "KEKW"}},
                {"snippet": {"publishedAt": "2026-11-18T12:00:02.5Z", "displayMessage": "ахаха"}}]})
        return httpx.Response(403, text='{"error": {"errors": [{"reason": "quotaExceeded"}]}}')

    async def go():
        alerts = []

        async def on_alert(key, text):
            alerts.append(key)
        chat = YouTubeChat("test-key", lambda login, text, ts: got.append((login, text, ts)),
                           {"chat_poll_seconds": 0.01}, on_alert=on_alert)
        chat.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        chat.start("yt_x", "VIDEO1")
        for _ in range(100):
            await asyncio.sleep(0.02)
            if alerts:
                break
        assert chat.lag("yt_x") == 0.0          # после исчерпания квоты детектор без запаздывания (чата нет)
        await chat.stop("yt_x")
        return alerts

    alerts = asyncio.run(go())
    assert [(t, round(ts - 1795003200, 1)) for _, t, ts in got] == [("KEKW", 1.0), ("ахаха", 2.5)]
    assert alerts == ["ytquota"]
