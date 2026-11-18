"""Площадки стримов: Twitch и YouTube. Как распознать канал из присланного текста, где смотреть эфир,
как подписывать источник."""
from __future__ import annotations

import re

YT_PREFIX = "yt_"  # логин YouTube-канала в базе: yt_<handle> (у Twitch — просто ник)

_YT_HANDLE = re.compile(r"(?:https?://)?(?:www\.|m\.)?youtube\.com/@([\w.\-]+)", re.I)
_YT_CHANNEL_ID = re.compile(r"(?:https?://)?(?:www\.|m\.)?youtube\.com/channel/(UC[\w\-]{20,})", re.I)
_YT_CUSTOM = re.compile(r"(?:https?://)?(?:www\.|m\.)?youtube\.com/(?:c|user)/([\w.\-]+)", re.I)


def parse_channel(text: str) -> tuple[str, str, str] | None:
    """(площадка, логин в базе, адрес канала) или None, если не похоже на канал.

    YouTube: «https://youtube.com/@handle», «youtube.com/channel/UC…», «yt:@handle», «@handle youtube».
    Twitch: ник или ссылка twitch.tv/ник."""
    t = text.strip()
    for rx, kind in ((_YT_HANDLE, "@"), (_YT_CHANNEL_ID, "channel/"), (_YT_CUSTOM, "c/")):
        m = rx.search(t)
        if m:
            ident = m.group(1)
            url = f"https://www.youtube.com/{'@' if kind == '@' else kind}{ident}"
            return "youtube", (YT_PREFIX + ident).lower()[:40], url
    low = t.lower()
    if low.startswith(("yt:", "youtube:")) or (t.startswith("@") and "youtube" in low):
        ident = t.split(":", 1)[-1].split()[0].lstrip("@")
        if ident and all(ch.isalnum() or ch in "._-" for ch in ident):
            return "youtube", (YT_PREFIX + ident).lower()[:40], f"https://www.youtube.com/@{ident}"
    login = low.removeprefix("@")
    for pref in ("https://", "http://", "www.", "m.twitch.tv/", "twitch.tv/"):
        login = login.removeprefix(pref)
    login = login.split("/")[0].split("?")[0]
    if login and all(ch.isalnum() or ch == "_" for ch in login):
        return "twitch", login, f"https://twitch.tv/{login}"
    return None


def platform_of(ch: dict | None, login: str = "") -> str:
    if ch and ch.get("platform"):
        return ch["platform"]
    return "youtube" if (login or (ch or {}).get("login", "")).startswith(YT_PREFIX) else "twitch"


def live_url(ch: dict) -> str:
    """Адрес, по которому streamlink находит текущий эфир канала."""
    if platform_of(ch) == "youtube":
        return (ch.get("url") or f"https://www.youtube.com/@{ch['login'][len(YT_PREFIX):]}").rstrip("/") + "/live"
    return f"https://twitch.tv/{ch['login']}"


def credit(ch: dict | None, login: str) -> str:
    """Подпись источника на видео: «twitch.tv/ник» или «YouTube: Название»."""
    if platform_of(ch, login) == "youtube":
        name = (ch or {}).get("display_name") or login[len(YT_PREFIX):]
        return f"YouTube: {name}"
    return f"twitch.tv/{login}"


def link(ch: dict | None, login: str) -> str:
    """Ссылка на канал для описания публикации."""
    if platform_of(ch, login) == "youtube":
        return (ch or {}).get("url") or f"https://www.youtube.com/@{login[len(YT_PREFIX):]}"
    return f"twitch.tv/{login}"


def game_from_category(category: str | None, platform: str) -> str:
    """У YouTube категория общая («Gaming», «Music») — это не название игры."""
    if not category:
        return ""
    if platform == "youtube" and category in ("Gaming", "People & Blogs", "Entertainment", "Music", "Education",
                                              "Howto & Style", "Science & Technology", "Sports", "Comedy"):
        return ""
    return category
