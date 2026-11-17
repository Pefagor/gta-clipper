"""Готовые видео по ссылке: YouTube, записи стримов Twitch (VOD) или локальный файл.

Два режима:
- «best»  — лучшие моменты: тепловая карта YouTube («самые пересматриваемые»), крики, чат записи стрима
            и Haiku по расшифровке всего видео → кандидаты в обычный конвейер (Sonnet, монтаж, модерация);
- «parts» — нарезка всего видео на части ~75 с по паузам в речи, «Часть N/M» на видео.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------- скачивание ----------------
def download(url: str, out_dir: Path, ffmpeg: str, max_height: int = 1080, progress=None) -> dict:
    """Синхронно (вызывать через asyncio.to_thread). Возвращает сведения о видео и путь к файлу."""
    out_dir.mkdir(parents=True, exist_ok=True)
    local = Path(url).expanduser()
    if local.exists():  # свой файл с диска
        return {"path": str(local.resolve()), "title": local.stem, "uploader": "", "uploader_id": "",
                "extractor": "file", "duration": None, "heatmap": [], "webpage_url": "", "chat": None}
    import yt_dlp

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(f"скачиваю… {d.get('downloaded_bytes', 0) * 100 // total}%")

    opts = {
        "quiet": True, "no_warnings": True, "noprogress": True, "logger": _YtdlpLog(),
        "format": (f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/bv*[height<={max_height}]+ba/"
                   f"b[height<={max_height}]/b"),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "ffmpeg_location": str(Path(ffmpeg).parent) if Path(ffmpeg).is_absolute() else None,
        "retries": 5, "fragment_retries": 10, "concurrent_fragment_downloads": 4,
        "progress_hooks": [hook],
        # записи стримов Twitch: чат приходит как «субтитры» rechat
        "writesubtitles": True, "subtitleslangs": ["rechat"],
    }
    with yt_dlp.YoutubeDL({k: v for k, v in opts.items() if v is not None}) as ydl:
        info = ydl.extract_info(url, download=True)
    files = sorted(out_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    video = next((p for p in files if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")), None)
    if not video:
        raise RuntimeError("видео не скачалось")
    chat = next((p for p in out_dir.glob("source*.rechat.json")), None)
    return {"path": str(video), "title": info.get("title") or "", "uploader": info.get("uploader") or "",
            "uploader_id": (info.get("uploader_id") or "").lower(), "extractor": info.get("extractor_key") or "",
            "duration": info.get("duration"), "heatmap": info.get("heatmap") or [],
            "webpage_url": info.get("webpage_url") or url, "chat": str(chat) if chat else None,
            "game": info.get("game") or None}


class _YtdlpLog:
    """Сообщения yt-dlp — в подробный лог, а не в консоль: он пишет «ERROR» и на ошибки, после которых
    сам переключается на запасной способ и успешно скачивает. Настоящая ошибка придёт исключением."""

    def debug(self, msg):
        log.debug("yt-dlp: %s", msg)

    def info(self, msg):
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg):
        log.debug("yt-dlp: %s", msg)

    def error(self, msg):
        log.debug("yt-dlp: %s", msg)


def parse_rechat(path: str | Path | None) -> list[tuple[float, str]]:
    """Чат записи стрима Twitch (yt-dlp rechat) → [(секунда видео, текст)]."""
    if not path or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    items = data.get("comments", data) if isinstance(data, dict) else data
    out = []
    for c in items if isinstance(items, list) else []:
        t = c.get("content_offset_seconds") if isinstance(c, dict) else None
        msg = c.get("message") or {} if isinstance(c, dict) else {}
        body = msg.get("body") if isinstance(msg, dict) else None
        if body is None and isinstance(msg, dict):
            body = "".join(f.get("text", "") for f in msg.get("fragments", []) if isinstance(f, dict))
        if t is not None and body:
            out.append((float(t), str(body)))
    out.sort()
    return out


# ---------------- сигналы ----------------
def heat_peaks(heatmap: list[dict], duration: float, min_value: float = 0.45) -> list[tuple[float, float]]:
    """Пики тепловой карты YouTube: [(секунда, сила 0..1)]. Самое начало видео пропускаем — оно всегда «горит»."""
    if not heatmap:
        return []
    vals = [float(h.get("value", 0)) for h in heatmap]
    peaks = []
    for i, h in enumerate(heatmap):
        center = (float(h["start_time"]) + float(h["end_time"])) / 2
        if i == 0 or center < max(15.0, 0.03 * (duration or 0)):  # начало видео смотрят все — не пик
            continue
        v = vals[i]
        if v >= min_value and v >= max(vals[max(0, i - 2):i + 3]):
            peaks.append((center, v))
    return peaks


def chat_peaks(chat: list[tuple[float, str]], duration: float, bucket: float = 10.0,
               min_z: float = 3.0) -> list[tuple[float, float]]:
    """Всплески чата записи стрима: [(секунда, z-оценка)]."""
    if len(chat) < 50:
        return []
    import statistics
    n = int(duration // bucket) + 1
    counts = [0] * n
    for t, _ in chat:
        if 0 <= t < duration:
            counts[int(t // bucket)] += 1
    med = statistics.median(counts)
    sd = max(statistics.pstdev(counts), 1.0)
    out = []
    for i, c in enumerate(counts):
        z = (c - med) / sd
        if z >= min_z and c == max(counts[max(0, i - 2):i + 3]):
            out.append((i * bucket + bucket / 2, round(z, 1)))
    return out


def merge_windows(items: list[dict], max_len: float = 100.0) -> list[dict]:
    """Склеивает пересекающиеся окна кандидатов. item: {start, end, score, ...}."""
    items = sorted(items, key=lambda x: x["start"])
    out: list[dict] = []
    for it in items:
        if out and it["start"] < out[-1]["end"] - 5 and max(out[-1]["end"], it["end"]) - out[-1]["start"] <= max_len:
            last = out[-1]
            last["end"] = max(last["end"], it["end"])
            last["score"] = max(last["score"], it["score"]) + 0.3 * min(last["score"], it["score"])
            last["sources"] = sorted(set(last.get("sources", [])) | set(it.get("sources", [])))
            for k in ("title", "why"):
                if it.get(k) and not last.get(k):
                    last[k] = it[k]
        else:
            out.append(dict(it))
    return out


# ---------------- нарезка на части ----------------
def split_parts(segments: list[dict], duration: float, target: float = 75, min_len: float = 45,
                max_len: float = 120) -> list[tuple[float, float]]:
    """Части видео ~target секунд, разрез — в паузе между фразами (чем длиннее пауза, тем лучше), не посреди слова."""
    gaps = []  # (секунда разреза, длина паузы)
    for a, b in zip(segments, segments[1:]):
        gap = b["start"] - a["end"]
        if gap >= 0.15:
            gaps.append(((a["end"] + b["start"]) / 2, gap))
    parts, cur = [], 0.0
    while duration - cur > max_len:
        lo, hi = cur + min_len, cur + max_len
        options = [(t, g) for t, g in gaps if lo <= t <= hi]
        if options:  # ближе к целевой длине и в паузе подлиннее
            cut = min(options, key=lambda o: abs(o[0] - cur - target) / target - min(o[1], 2.0) * 0.15)[0]
        else:
            cut = cur + target
        parts.append((round(cur, 2), round(cut, 2)))
        cur = cut
    rest = duration - cur
    if parts and rest < min_len and (parts[-1][1] - parts[-1][0]) + rest <= max_len * 1.3:
        parts[-1] = (parts[-1][0], round(duration, 2))
    elif rest > 1:
        parts.append((round(cur, 2), round(duration, 2)))
    return parts


# ---------------- расшифровка: срезы и текст ----------------
def slice_transcript(tr: dict, start: float, end: float) -> dict:
    """Кусок расшифровки всего видео со временем относительно start."""
    segs = []
    for s in tr.get("segments", []):
        if s["end"] <= start or s["start"] >= end:
            continue
        words = [{**w, "start": round(w["start"] - start, 2), "end": round(w["end"] - start, 2)}
                 for w in s.get("words", []) if w["end"] > start and w["start"] < end]
        segs.append({"start": round(max(s["start"], start) - start, 2), "end": round(min(s["end"], end) - start, 2),
                     "text": s["text"], "words": words})
    loud = [{**e, "start": e["start"] - start, "end": e["end"] - start}
            for e in tr.get("loud", []) if e["end"] > start and e["start"] < end]
    return {"language": tr.get("language"), "segments": segs, "loud": loud}


def fmt_ts(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def transcript_chunks(tr: dict, marks: list[tuple[float, str]], chunk: float = 600, overlap: float = 30,
                      duration: float = 0) -> list[tuple[float, float, str]]:
    """Расшифровка по кускам ~10 минут с таймкодами и пометками (пик просмотров, крик, чат) для Haiku."""
    lines = [(s["start"], f"[{fmt_ts(s['start'])}] {s['text']}") for s in tr.get("segments", [])]
    lines += [(t, f"[{fmt_ts(t)}] ({m})") for t, m in marks]
    lines.sort(key=lambda x: x[0])
    total = duration or (lines[-1][0] + 1 if lines else 0)
    out, cur = [], 0.0
    while cur < total:
        end = min(total, cur + chunk)
        text = "\n".join(l for t, l in lines if cur - overlap <= t < end)
        if text.strip():
            out.append((cur, end, text[:12000]))
        cur = end
    return out


def parse_ts(v) -> float | None:
    """«12:34», «1:02:03», «754» или число → секунды."""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    v = v.strip()
    if re.fullmatch(r"\d+(\.\d+)?", v):
        return float(v)
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", v)
    if not m:
        return None
    h, mnt, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    return float(h * 3600 + mnt * 60 + s)


def channel_slug(info: dict) -> str:
    """Канал для кандидатов: ник Twitch (тогда работают настройки канала) или имя автора на YouTube."""
    if "twitch" in (info.get("extractor") or "").lower() and info.get("uploader_id"):
        return info["uploader_id"]
    return (info.get("uploader") or "видео").strip()[:40]


def credit_line(info: dict) -> str:
    ex = (info.get("extractor") or "").lower()
    if "twitch" in ex and info.get("uploader_id"):
        return f"twitch.tv/{info['uploader_id']}"
    if "youtube" in ex and info.get("uploader"):
        return f"YouTube: {info['uploader']}"
    return info.get("uploader") or ""


def cleanup_source(job_dir: Path) -> None:
    for p in job_dir.glob("source.*"):
        p.unlink(missing_ok=True)
    shutil.rmtree(job_dir / "tmp", ignore_errors=True)
