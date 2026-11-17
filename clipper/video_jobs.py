"""Задания «видео по ссылке»: скачать → распознать речь → найти моменты или нарезать на части →
кандидаты в общий конвейер (ИИ, монтаж, модерация)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from .cutter import extract_audio
from .llm import LLMError
from .tools import encoder_args, probe, run
from .transcribe import loud_events, read_wav
from .videos import (channel_slug, chat_peaks, cleanup_source, credit_line, download, fmt_ts, heat_peaks,
                     merge_windows, parse_rechat, slice_transcript, split_parts, transcript_chunks)

log = logging.getLogger(__name__)

MODES = {"best": "лучшие моменты", "parts": "нарезка на части"}


class VideoJobs:
    def __init__(self, app):
        self.app = app
        self.db = app.db
        self.cfg = app.cfg.section("videos")
        self.root = app.cfg.data_dir / "videos"
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- очередь ----------
    def add(self, url: str, mode: str) -> int:
        now = time.time()
        vid = self.db.execute("INSERT INTO videos(url,mode,status,progress,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                              (url.strip(), mode, "queued", "в очереди", now, now)).lastrowid
        log.info("🔗 видео #%d в очереди (%s): %s", vid, MODES.get(mode, mode), url)
        return vid

    def update(self, vid: int, **fields) -> None:
        fields["updated_at"] = time.time()
        sets = ",".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE videos SET {sets} WHERE id=?", list(fields.values()) + [vid])

    def list(self, limit: int = 8) -> list[dict]:
        return self.db.all("SELECT * FROM videos ORDER BY id DESC LIMIT ?", (limit,))

    async def loop(self) -> None:
        # после перезапуска незаконченные задания начинаем заново
        self.db.execute("UPDATE videos SET status='queued', progress='в очереди (после перезапуска)' "
                        "WHERE status NOT IN ('queued','done','failed')")
        while True:
            job = self.db.one("SELECT * FROM videos WHERE status='queued' ORDER BY id LIMIT 1")
            if not job:
                await asyncio.sleep(5)
                continue
            try:
                n = await self.process(job)
                self.update(job["id"], status="done", progress=f"готово: {n} клип.", n_clips=n)
                log.info("🔗 видео #%d готово: %d клипов отправлено в обработку", job["id"], n)
                if self.app.bot:
                    await self.app.bot.notify(f"🔗 Видео #{job['id']} обработано ({MODES.get(job['mode'])}): "
                                              f"{n} клип. — придут на модерацию.")
            except Exception as e:  # noqa: BLE001
                log.exception("видео #%d", job["id"])
                self.update(job["id"], status="failed", progress="ошибка", error=str(e)[:500])
                await self.app.alert(f"video:{job['id']}", f"Видео #{job['id']} не обработано: {str(e)[:200]}")

    # ---------- обработка ----------
    async def process(self, job: dict) -> int:
        vid = job["id"]
        d = self.root / f"{vid:05d}"
        d.mkdir(parents=True, exist_ok=True)
        self.update(vid, status="downloading", progress="скачиваю…", dir=str(d))
        log.info("🔗 видео #%d: скачиваю…", vid)
        info = await asyncio.to_thread(download, job["url"], d, self.app.ffmpeg,
                                       int(self.cfg.get("max_height", 1080)),
                                       lambda s: self.update(vid, progress=s))
        src = Path(info["path"])
        meta = await probe(self.app.ffprobe, src)
        duration = float(info.get("duration") or meta["duration"])
        max_min = float(self.cfg.get("max_minutes", 360))
        if duration > max_min * 60:
            raise RuntimeError(f"видео длиннее {max_min:.0f} мин (videos.max_minutes в config.yaml)")
        channel = channel_slug(info)
        self.update(vid, title=info["title"][:200], uploader=info["uploader"][:100], channel=channel,
                    credit=credit_line(info), source_url=info["webpage_url"], duration=duration,
                    game=info.get("game"))
        log.info("🔗 видео #%d: «%s» (%s, %s)", vid, info["title"][:60], info["uploader"] or "свой файл", fmt_ts(duration))

        # речь всего видео — один раз; дальше куски берутся из этой расшифровки
        self.update(vid, status="transcribing", progress="распознаю речь…")
        log.info("🗣  видео #%d: распознаю речь всего видео (%s)…", vid, fmt_ts(duration))
        wav = d / "audio.wav"
        await extract_audio(self.app.ffmpeg, src, wav)
        t0 = time.time()
        tr = await asyncio.to_thread(self.app.transcriber.transcribe, wav)
        tr["loud"] = await asyncio.to_thread(lambda: loud_events(read_wav(wav)))
        wav.unlink(missing_ok=True)
        (d / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        log.info("🗣  видео #%d: речь распознана за %.0f с", vid, time.time() - t0)

        chat = parse_rechat(info.get("chat"))
        self.update(vid, status="analyzing", progress="ищу моменты…" if job["mode"] == "best" else "нарежу на части…")
        job = self.db.one("SELECT * FROM videos WHERE id=?", (vid,))
        if job["mode"] == "parts":
            n = await self._parts(job, src, tr, duration, chat)
        else:
            n = await self._best(job, src, tr, duration, chat, info.get("heatmap") or [])
        if self.cfg.get("delete_source", True):
            cleanup_source(d)  # исходник большой (запись стрима — гигабайты); куски уже вырезаны
        return n

    async def _cut(self, src: Path, out_dir: Path, start: float, end: float) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = out_dir / "raw.mp4"
        code, out = await run([self.app.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
                               "-i", str(src), "-t", f"{end - start:.2f}", "-map", "0:v:0", "-map", "0:a:0?",
                               *encoder_args(self.app.encoder, 8000), "-pix_fmt", "yuv420p",
                               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(raw)], timeout=900)
        if code != 0 or not raw.exists():
            raise RuntimeError(f"не удалось вырезать кусок видео: {out[-300:]}")
        return raw

    async def _make_candidate(self, job: dict, src: Path, tr: dict, chat, start: float, end: float,
                              **fields) -> int:
        cid = self.db.add_candidate(kind="clip", channel=job["channel"], game=job.get("game"), video_id=job["id"],
                                    credit=job["credit"], source_url=job["source_url"], video_pos=start,
                                    status="cutting", created_at=time.time())
        d = self.app.cfg.data_dir / "candidates" / f"{cid:06d}_video{job['id']}"
        await self._cut(src, d, start, end)
        (d / "transcript.json").write_text(json.dumps(slice_transcript(tr, start, end), ensure_ascii=False),
                                           encoding="utf-8")
        (d / "chat.json").write_text(json.dumps([(t - start, m) for t, m in chat if start <= t <= end],
                                                ensure_ascii=False), encoding="utf-8")
        self.db.update_candidate(cid, dir=str(d), raw_start_ts=0.0, raw_duration=end - start,
                                 win_start=start, win_end=end, **fields)
        return cid

    # ---------- режим «лучшие моменты» ----------
    async def _best(self, job: dict, src: Path, tr: dict, duration: float, chat, heatmap) -> int:
        name = self.app.streamer_name(job["channel"]) if self.db.channel(job["channel"]) else (job["uploader"] or "автор")
        heat = heat_peaks(heatmap, duration)
        rage = [e for e in tr.get("loud", []) if e["over_db"] >= 9 and e["end"] - e["start"] >= 1.0]
        chats = chat_peaks(chat, duration)
        marks = ([(t, "ПИК ПРОСМОТРОВ") for t, _ in heat] + [(e["start"], "КРИК") for e in rage]
                 + [(t - 5, "ЧАТ") for t, _ in chats])
        items = []
        for t, v in heat:
            items.append({"start": t - 40, "end": t + 25, "score": 4 + 4 * v, "sources": ["просмотры"], "peak": t})
        for e in rage:
            items.append({"start": e["start"] - 35, "end": e["end"] + 20, "score": 6, "sources": ["крик"],
                          "peak": e["start"]})
        for t, z in chats:
            items.append({"start": t - 45, "end": t + 20, "score": 5 + min(z, 6) / 2, "sources": ["чат"], "peak": t - 8})
        if self.app.llm.enabled:
            chunks = transcript_chunks(tr, marks, duration=duration)
            for i, (c0, c1, text) in enumerate(chunks):
                self.update(job["id"], progress=f"ИИ ищет моменты… {i + 1}/{len(chunks)}")
                try:
                    found = await self.app.llm.find_moments(job["title"] or "", name, job.get("game") or "", text)
                except LLMError as e:
                    log.warning("видео #%d: ИИ не ответил на кусок %d: %s", job["id"], i + 1, e)
                    continue
                for m in found:
                    s, e = max(0.0, m["start"] - 6), min(duration, m["end"] + 6)
                    if e - s < 12:
                        continue
                    items.append({"start": s, "end": min(e, s + 100), "score": m["score"], "sources": ["ИИ"],
                                  "title": m["title"], "why": m["why"], "peak": (m["start"] + m["end"]) / 2})
        windows = merge_windows([{**it, "start": max(0.0, it["start"]), "end": min(duration, it["end"])}
                                 for it in items if it["end"] - it["start"] >= 12])
        limit = int(max(3, min(float(self.cfg.get("max_moments", 30)), duration / 180)))
        windows = sorted(windows, key=lambda w: -w["score"])[:limit]
        log.info("🎯 видео #%d: найдено %d моментов (просмотры %d, крики %d, чат %d, ИИ %d), беру %d",
                 job["id"], len(items), len(heat), len(rage), len(chats),
                 sum(1 for it in items if "ИИ" in it["sources"]), len(windows))
        n = 0
        for w in sorted(windows, key=lambda w: w["start"]):
            self.update(job["id"], progress=f"вырезаю моменты… {n + 1}/{len(windows)}")
            peak_rel = max(0.0, min(w["end"], w.get("peak", w["start"])) - w["start"])
            fields = {"trigger": "video", "peak_ts": peak_rel, "score": round(w["score"], 1),
                      "signals": {"sources": w.get("sources", [])}}
            if w.get("title"):  # ИИ уже нашёл и описал момент — сразу к Sonnet за границами и заголовками
                fields.update(status="screened", screen={"score": int(min(10, round(w["score"]))),
                                                         "category": "другое", "summary": w.get("why") or w["title"]})
            else:
                fields.update(status="transcribed")
            await self._make_candidate(job, src, tr, [], w["start"], w["end"], **fields)
            n += 1
        return n

    # ---------- режим «на части» ----------
    async def _parts(self, job: dict, src: Path, tr: dict, duration: float, chat) -> int:
        c = self.cfg
        parts = split_parts(tr.get("segments", []), duration, float(c.get("part_seconds", 75)),
                            float(c.get("part_min", 45)), float(c.get("part_max", 120)))
        total = len(parts)
        name = self.app.streamer_name(job["channel"]) if self.db.channel(job["channel"]) else (job["uploader"] or "")
        meta: dict[int, dict] = {}
        if self.app.llm.enabled and c.get("ai_titles", True):
            texts = [(i + 1, " ".join(s["text"] for s in slice_transcript(tr, a, b)["segments"]))
                     for i, (a, b) in enumerate(parts)]
            for k in range(0, total, 15):
                self.update(job["id"], progress=f"ИИ придумывает заголовки… {min(k + 15, total)}/{total}")
                try:
                    meta.update(await self.app.llm.title_parts(job["title"] or "", name, texts[k:k + 15]))
                except LLMError as e:
                    log.warning("видео #%d: заголовки частей без ИИ: %s", job["id"], e)
        skip = int(c.get("skip_score", 3))
        base = (job["title"] or "Видео")[:60]
        log.info("🧩 видео #%d: нарезаю на %d частей", job["id"], total)
        for i, (a, b) in enumerate(parts):
            n = i + 1
            self.update(job["id"], progress=f"нарезаю на части… {n}/{total}")
            m = meta.get(n, {})
            title = m.get("title") or base
            fields = {"trigger": "video_part", "part": f"{n}/{total}", "peak_ts": 0.0, "title": title,
                      "titles": [title, base], "category": "другое", "clip_start": 0.0, "clip_end": b - a,
                      "score": m.get("score") or 0}
            if m and m.get("score", 10) <= skip:  # проходная часть — в «Отсеянные», вернуть можно оттуда
                fields.update(status="discarded", screen={"score": m["score"], "summary": "проходная часть"})
            else:
                fields.update(status="to_render", screen={"score": m["score"], "summary": title} if m else {"skipped": True})
            await self._make_candidate(job, src, tr, chat, a, b, **fields)
        return total
