"""Ядро программы: наблюдение за каналами, запись, детектор, конвейер обработки кандидатов, уборка."""
from __future__ import annotations

import asyncio
import html
import json
import logging
import shutil
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from .chat import ChatHub
from .config import Config
from .cutter import CutError, cut_candidate, extract_audio, extract_frames, spread
from .db import DB, day_start
from .detector import ChannelDetector, OpenWindow, extend_window, plan_window
from .llm import LLMError, RouterAI, build_context
from .recorder import Recorder
from .render import render_clip, render_compilation
from .segments import load_session_segments
from .tools import encoder_args, find_ffmpeg, find_ffprobe, pick_encoder, probe, run, streamlink_cmd
from .transcribe import Transcriber, loud_events, read_wav, transcript_text, words_in
from .twitch import TwitchAPI, parse_ts

log = logging.getLogger(__name__)

FINAL = ("rejected", "discarded", "failed", "done")


def describe_trigger(trigger: str, score: float, signals: dict) -> str:
    if trigger == "viewer_clip":
        return "зрители сделали клип"
    parts = []
    if signals.get("rage"):
        parts.append(f"КРИК +{signals.get('loud_by_db', 0):.0f} дБ")
    if signals.get("chat_z", 0) >= 1:
        parts.append(f"всплеск чата ×{signals['chat_z']:.1f}")
    if signals.get("emote_score", 0) >= 0.5:
        parts.append("эмоуты")
    if not signals.get("rage") and signals.get("audio_z", 0) >= 1:
        parts.append("громкий звук")
    return f"балл {score:.1f}" + (" (" + ", ".join(parts) + ")" if parts else "")


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.data_dir / "app.db")
        self.ffmpeg = find_ffmpeg(cfg.get("ffmpeg", "auto"))
        self.ffprobe = find_ffprobe(cfg.get("ffprobe", "auto"), self.ffmpeg)
        self.encoder = pick_encoder(self.ffmpeg, cfg.section("render").get("encoder", "auto"))
        s = cfg.secrets
        self.twitch = TwitchAPI(s.twitch_client_id, s.twitch_client_secret, streamlink_cmd(cfg.get("streamlink", "auto")))
        self.chat = ChatHub()
        self.llm = RouterAI(cfg.section("llm"), s.routerai_key, on_usage=self._on_usage)
        self.transcriber = Transcriber(cfg.section("whisper"))
        self.recorders: dict[str, Recorder] = {}
        self.detectors: dict[str, ChannelDetector] = {}
        self.open_windows: dict[str, OpenWindow] = {}
        self.last_end: dict[str, float] = {}
        self._last_status: dict[str, float] = {}  # конец окна последнего кандидата канала
        self.clip_moments: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.seen_clips: set[str] = set()
        self.offline_checks: dict[str, int] = defaultdict(int)
        self.live: dict[str, dict] = {}  # логин -> {"game", "title"}
        self.wake = asyncio.Event()
        self.bot = None  # BotUI, если задан токен Telegram
        self._alerted: dict[str, float] = {}
        self._tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()
        self._started_at = time.time()
        (cfg.data_dir / "sessions").mkdir(exist_ok=True)
        # сессии, оборванные падением или закрытием окна, считаем завершёнными — иначе уборка их не тронет
        self.db.execute("UPDATE sessions SET ended_at=? WHERE ended_at IS NULL", (time.time(),))
        (cfg.data_dir / "candidates").mkdir(exist_ok=True)
        log.info("⚙️  ffmpeg: %s · кодировщик видео: %s", self.ffmpeg, self.encoder)

    # ---------- настройки ----------
    def setting(self, name: str):
        return self.db.kv_get(f"set:{name}", self.cfg.section("defaults").get(name))

    def set_setting(self, name: str, value) -> None:
        self.db.kv_set(f"set:{name}", value)

    # ---------- уведомления ----------
    async def alert(self, key: str, text: str, buttons=None, every: float = 600) -> None:
        now = time.time()
        if now - self._alerted.get(key, 0) < every:
            return
        self._alerted[key] = now
        log.warning("%s", text)
        if self.bot:
            await self.bot.notify("⚠️ " + html.escape(text), buttons)

    def spawn(self, coro) -> None:
        """Фоновая задача с сохранённой ссылкой и логом ошибок."""
        task = asyncio.create_task(coro)
        self._bg.add(task)

        def done(t: asyncio.Task):
            self._bg.discard(t)
            if not t.cancelled() and t.exception():
                log.error("фоновая задача упала", exc_info=t.exception())
        task.add_done_callback(done)

    def _on_usage(self, model: str, in_tok: int, out_tok: int, cost: float) -> None:
        self.db.add_spend(model, in_tok, out_tok, cost)

    # ---------- каналы ----------
    async def add_channel(self, login: str) -> tuple[bool, str]:
        login = login.strip().lower().removeprefix("@")
        for pref in ("https://", "http://", "www.", "twitch.tv/", "m.twitch.tv/"):
            login = login.removeprefix(pref)
        login = login.split("/")[0].split("?")[0]
        if not login or not all(ch.isalnum() or ch == "_" for ch in login):
            return False, "Не похоже на ник Twitch."
        display, user_id = login, None
        if self.twitch.enabled:
            try:
                user = await self.twitch.user(login)
            except Exception as e:  # noqa: BLE001
                return False, f"Twitch API не ответил: {e}"
            if not user:
                return False, f"Канал «{login}» не найден на Twitch."
            display, user_id = user.get("display_name") or login, user.get("id")
        existing = self.db.channel(login)
        if existing and existing["state"] in ("active", "paused"):
            return False, f"{display} уже в списке."
        if existing:
            self.db.update_channel(login, state="active", removed_at=None, display_name=display,
                                   user_id=user_id or existing.get("user_id"))
        else:
            self.db.execute("INSERT INTO channels(login,display_name,user_id,state,added_at) VALUES(?,?,?,?,?)",
                            (login, display, user_id, "active", time.time()))
        self.wake.set()
        return True, display

    async def remove_channel(self, login: str, now: bool) -> None:
        if now or login not in self.recorders:
            await self._stop_recording(login)
            self.db.update_channel(login, state="removed", removed_at=time.time())
        else:
            self.db.update_channel(login, state="removing")

    async def set_paused(self, login: str, paused: bool) -> None:
        ch = self.db.channel(login)
        if ch and ch["state"] == "removing" and paused:  # пауза удаляемого канала = удалить сейчас
            return await self.remove_channel(login, now=True)
        self.db.update_channel(login, state="paused" if paused else "active")
        if paused:
            await self._stop_recording(login)
        self.wake.set()

    # ---------- запуск ----------
    async def start(self) -> None:
        loops = [self.chat.run(), self._watch_loop(), self._detect_loop(), self._janitor_loop(),
                 self._stage("to_cut", self._do_cut), self._stage("cut", self._do_transcribe),
                 self._stage("transcribed", self._do_screen, max_attempts=6),
                 self._stage("screened", self._do_review, max_attempts=6),
                 self._stage("to_render", self._do_render)]
        if self.twitch.enabled:
            loops.append(self._clips_loop())
        self._tasks = [asyncio.create_task(c) for c in loops]

    async def shutdown(self) -> None:
        for login in list(self.recorders):
            await self._stop_recording(login)
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---------- наблюдатель ----------
    async def _watch_loop(self) -> None:
        interval = float(self.cfg.section("recording").get("live_check_seconds", 60))
        while True:
            try:
                await self._watch_once()
            except Exception:  # noqa: BLE001
                log.exception("ошибка наблюдателя")
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), interval)
            except asyncio.TimeoutError:
                pass

    async def _watch_once(self) -> None:
        if self.setting("paused"):
            for login in list(self.recorders):
                await self._stop_recording(login)
            return
        chans = {c["login"]: c for c in self.db.channels(("active", "removing"))}
        for login in list(self.recorders):
            if login not in chans:
                await self._stop_recording(login)
        self.live = await self.twitch.live_info(list(chans))
        for login, info in self.live.items():
            rec = self.recorders.get(login)
            if rec and info.get("game") and info["game"] != rec.game:
                if rec.game:
                    log.info("🎮 [%s] сменил игру: %s → %s", login, rec.game, info["game"])
                rec.game = info["game"]
        for login, ch in chans.items():
            rec = self.recorders.get(login)
            if rec and not rec.running:  # задача записи умерла — перезапускаем
                log.warning("[%s] запись не работает — перезапускаю", login)
                await self._stop_recording(login)
                rec = None
            if login in self.live:
                self.offline_checks[login] = 0
                if not rec and ch["state"] == "active":
                    await self._start_recording(login)
            else:
                if ch["state"] == "removing" and not rec:
                    self.db.update_channel(login, state="removed", removed_at=time.time())
                if rec:
                    self.offline_checks[login] += 1
                    if self.offline_checks[login] >= 2:
                        log.info("⚫ [%s] эфир закончился", login)
                        await self._stop_recording(login)
                        if ch["state"] == "removing":
                            self.db.update_channel(login, state="removed", removed_at=time.time())

    async def _start_recording(self, login: str) -> None:
        session_dir = self.cfg.data_dir / "sessions" / login / time.strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        sid = self.db.execute("INSERT INTO sessions(channel,dir,started_at) VALUES(?,?,?)",
                              (login, str(session_dir), time.time())).lastrowid
        rec = Recorder(login, sid, session_dir, self.cfg, self.ffmpeg, on_alert=self._rec_alert)
        self.recorders[login] = rec
        self.detectors[login] = ChannelDetector(self.cfg.section("detector"))
        self.chat.set_log(login, session_dir / "chat.jsonl")
        rec.start()
        await self.chat.join(login)
        log.info("🔴 [%s] в эфире — начинаю запись", login)

    async def _stop_recording(self, login: str) -> None:
        rec = self.recorders.pop(login, None)
        self.open_windows.pop(login, None)
        if rec:
            await rec.stop()
            self.db.execute("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), rec.session_id))
            log.info("⏹  [%s] запись остановлена", login)
        self.chat.set_log(login, None)
        await self.chat.part(login)

    async def _rec_alert(self, login: str, reason: str) -> None:
        await self.alert(f"rec:{login}", f"Запись {login} прерывается: {reason}",
                         [[("🔄 Перезапустить", f"ch:restart:{login}"), ("📺 Открыть канал", f"ch:open:{login}")]])

    async def restart_recording(self, login: str) -> None:
        await self._stop_recording(login)
        self.wake.set()

    # ---------- детектор ----------
    async def _detect_loop(self) -> None:
        dcfg = self.cfg.section("detector")
        tick = float(dcfg.get("tick_seconds", 2))
        seg = float(self.cfg.section("recording").get("segment_seconds", 4))
        while True:
            await asyncio.sleep(tick)
            try:
                now = time.time()
                thr = float(dcfg.get("thresholds", {}).get(self.setting("sensitivity"), 6.0))
                for login, rec in list(self.recorders.items()):
                    if not rec.healthy:
                        continue
                    det = self.detectors[login]
                    score, signals = det.tick(now, self.chat.history(login), rec.audio, list(self.clip_moments[login]))
                    self._status_line(login, rec, score, signals, thr)
                    self._maybe_calibrate(login, rec, now)
                    win = self.open_windows.get(login)
                    if score >= thr or signals.get("rage"):
                        # момент «по звуку» привязан к секунде крика, по чату — к «сейчас минус задержка чата»
                        peak = signals.get("audio_peak_t") or now - dcfg.get("reaction_delay", 10)
                        if win:
                            extend_window(win, now, dcfg, peak)
                            self.last_end[login] = win.end
                            win.max_score = max(win.max_score, score)
                            self.db.update_candidate(win.cid, win_end=win.end, score=win.max_score,
                                                     next_try_at=win.end + seg + 3)
                        elif peak > self.last_end.get(login, 0) + 5:
                            # новый момент — только после конца окна предыдущего, без дублей
                            self._new_candidate(login, rec, peak, score, signals,
                                                "rage" if signals.get("rage") else "detector", track=True)
                    if win and now > win.end + seg + 3:
                        self.open_windows.pop(login, None)
            except Exception:  # noqa: BLE001
                log.exception("ошибка детектора")

    def _status_line(self, login: str, rec: Recorder, score: float, signals: dict, thr: float) -> None:
        """Раз в минуту — что видит детектор, чтобы было понятно, что программа жива."""
        now = time.time()
        if now - self._last_status.get(login, 0) < 60:
            return
        self._last_status[login] = now
        if signals.get("warmup"):
            log.info("👂 [%s] изучаю обычный уровень чата и звука… (чат %d сообщ./10 с)", login, signals["chat"])
            return
        audio = signals.get("audio_db")
        log.info("👂 [%s] чат %d сообщ./10 с · звук %s · балл %.1f из %.1f нужных", login, signals["chat"],
                 f"{audio:.0f} дБ" if audio is not None else "тишина", score, thr)

    def _new_candidate(self, login: str, rec: Recorder, peak: float, score: float, signals: dict,
                       trigger: str, track: bool) -> int:
        dcfg = self.cfg.section("detector")
        seg = float(self.cfg.section("recording").get("segment_seconds", 4))
        start, end = plan_window(peak, dcfg)
        start = max(start, rec.started_at, self.last_end.get(login, 0) if track else 0)
        cid = self.db.add_candidate(channel=login, session_id=rec.session_id, status="to_cut", trigger=trigger,
                                    game=rec.game or None,
                                    peak_ts=peak, win_start=start, win_end=end, score=score, signals=signals,
                                    next_try_at=end + seg + 3)
        if track:
            self.last_end[login] = end
            self.open_windows[login] = OpenWindow(cid, peak, start, end, score)
        log.info("🎯 [%s] момент #%d найден: %s", login, cid, describe_trigger(trigger, score, signals))
        return cid

    # ---------- клипы зрителей ----------
    async def _clips_loop(self) -> None:
        interval = float(self.cfg.get("clips_poll_seconds", 30))
        while True:
            await asyncio.sleep(interval)
            for login, rec in list(self.recorders.items()):
                try:
                    await self._poll_clips(login, rec)
                except Exception as e:  # noqa: BLE001 — ошибка одного канала не должна останавливать опрос
                    log.warning("клипы %s: %s", login, e)

    async def _poll_clips(self, login: str, rec: Recorder) -> None:
        ch = self.db.channel(login)
        if not ch or not ch.get("user_id"):
            return
        for clip in await self.twitch.recent_clips(ch["user_id"], time.time() - 900):
            if clip["id"] in self.seen_clips:
                continue
            self.seen_clips.add(clip["id"])
            moment = parse_ts(clip["created_at"]) - 12
            if moment < rec.started_at:
                continue
            self.clip_moments[login].append(moment)
            hit = self.db.one("SELECT * FROM candidates WHERE channel=? AND kind='clip' AND win_start<=? "
                              "AND win_end>=? ORDER BY id DESC LIMIT 1", (login, moment, moment))
            if hit:
                sig = hit.get("signals") or {}
                sig["viewer_clips"] = int(sig.get("viewer_clips", 0)) + 1
                self.db.update_candidate(hit["id"], signals=sig)
            else:
                self._new_candidate(login, rec, moment, 3.0, {"viewer_clips": 1, "clip_title": clip.get("title")},
                                    "viewer_clip", track=False)

    # ---------- конвейер ----------
    async def _stage(self, status: str, fn, max_attempts: int = 3) -> None:
        while True:
            try:
                c = self.db.next_candidate(status)
            except Exception:  # noqa: BLE001
                log.exception("очередь %s", status)
                await asyncio.sleep(5)
                continue
            if not c:
                await asyncio.sleep(2)
                continue
            try:
                await fn(c)
            except Exception as e:  # noqa: BLE001
                try:
                    await self._stage_failed(c, status, e, max_attempts)
                except Exception:  # noqa: BLE001
                    log.exception("обработка ошибки кандидата #%d", c["id"])
                    await asyncio.sleep(5)

    async def _stage_failed(self, c: dict, status: str, e: Exception, max_attempts: int) -> None:
        attempts = c["attempts"] + 1
        log.warning("кандидат #%d (%s): %s", c["id"], status, e)
        permanent = isinstance(e, CutError) and e.permanent
        if attempts >= max_attempts or permanent:
            old = c.get("render_path")
            if status == "to_render" and old and Path(old).exists():
                # неудачный перерендер: возвращаем прежний готовый клип на модерацию
                self.db.update_candidate(c["id"], status="rendered", edit_start=None, edit_end=None,
                                         card_msg_id=None, attempts=0, error=str(e)[:500])
                await self.alert(f"rerender:{c['id']}", f"Перерендер клипа #{c['id']} не удался, "
                                                        f"оставил прежнюю версию: {str(e)[:200]}")
                return
            self.db.update_candidate(c["id"], status="failed", error=str(e)[:500], attempts=attempts)
            if c["kind"] == "compilation":
                self.release_parts(c)
            if not permanent:
                await self.alert(f"fail:{status}",
                                 f"Кандидат #{c['id']} не обработан ({status}): {str(e)[:200]}")
        else:
            self.db.update_candidate(c["id"], attempts=attempts, error=str(e)[:500],
                                     next_try_at=time.time() + 30 * 2 ** (attempts - 1))
            if isinstance(e, LLMError) and attempts >= 3:
                await self.alert("llm", f"routerai не отвечает: {str(e)[:150]}",
                                 [[("🔁 Повторить сейчас", "st:retry"), ("📊 Статус", "st:show")]])

    def cand_dir(self, c: dict) -> Path:
        return Path(c["dir"]) if c.get("dir") else self.cfg.data_dir / "candidates" / f"{c['id']:06d}_{c['channel']}"

    async def _do_cut(self, c: dict) -> None:
        sess = self.db.one("SELECT * FROM sessions WHERE id=?", (c["session_id"],))
        out_dir = self.cand_dir(c)
        raw, raw_start, dur = await cut_candidate(self.ffmpeg, Path(sess["dir"]), out_dir,
                                                  c["win_start"], c["win_end"], c["peak_ts"])
        if dur < 8:
            raise CutError(f"слишком короткий кусок записи ({dur:.0f} с)", permanent=True)
        log.info("✂️  #%d вырезан фрагмент %d с", c["id"], dur)
        self.db.update_candidate(c["id"], status="cut", dir=str(out_dir), raw_start_ts=raw_start,
                                 raw_duration=dur, attempts=0, error=None)

    async def _do_transcribe(self, c: dict) -> None:
        d = Path(c["dir"])
        wav = d / "audio.wav"
        await extract_audio(self.ffmpeg, d / "raw.mp4", wav)
        log.info("🗣  #%d распознаю речь…", c["id"])
        t0 = time.time()
        tr = await asyncio.to_thread(self.transcriber.transcribe, wav)
        try:  # крики — по самому звуку: Whisper громкость не передаёт
            tr["loud"] = await asyncio.to_thread(lambda: loud_events(read_wav(wav)))
        except Exception as e:  # noqa: BLE001
            log.warning("громкость кандидата #%d: %s", c["id"], e)
        (d / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        wav.unlink(missing_ok=True)
        words = sum(len(s.get("words", [])) for s in tr.get("segments", []))
        loud = len(tr.get("loud", []))
        log.info("🗣  #%d речь распознана за %.0f с: %d слов%s", c["id"], time.time() - t0, words,
                 f", криков: {loud}" if loud else "")
        self.db.update_candidate(c["id"], status="transcribed", attempts=0, error=None)

    async def _context(self, c: dict, screen: dict | None = None) -> str:
        d = Path(c["dir"])
        tr = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
        t0, dur = c["raw_start_ts"], c["raw_duration"]
        chat = [(t - t0, m) for t, m in self.chat.window(c["channel"], t0, t0 + dur)]
        if not chat and t0 < self._started_at:  # окно до перезапуска программы — читаем лог сессии
            chat = await asyncio.to_thread(self._chat_from_log, c, t0, dur)
        return build_context(c["channel"], c.get("game") or "", transcript_text(tr), chat, c.get("signals") or {},
                             c["peak_ts"] - t0, dur, screen)

    def _chat_from_log(self, c: dict, t0: float, dur: float) -> list[tuple[float, str]]:
        sess = self.db.one("SELECT dir FROM sessions WHERE id=?", (c["session_id"],))
        path = Path(sess["dir"]) / "chat.jsonl" if sess else None
        out = []
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if t0 <= m["t"] <= t0 + dur:
                    out.append((m["t"] - t0, m["m"]))
        return out

    async def _budget_ok(self, c: dict) -> bool:
        limit = float(self.setting("api_daily_limit_rub") or 0)
        if limit > 0 and self.db.spend_since(day_start()) >= limit:
            self.db.update_candidate(c["id"], next_try_at=time.time() + 300)
            await self.alert("budget", f"Дневной лимит API ({limit:.0f} ₽) исчерпан, отбор на паузе. "
                                       "Запись продолжается.", every=6 * 3600)
            return False
        return True

    async def _do_screen(self, c: dict) -> None:
        if not self.llm.enabled:  # без ключа routerai: пропускаем ИИ, на модерацию идёт всё окно
            log.info("🤖 #%d ИИ-отбор выключен (нет ключа routerai) — беру фрагмент целиком", c["id"])
            self.db.update_candidate(c["id"], status="to_render", screen={"skipped": True}, clip_start=0.0,
                                     clip_end=c["raw_duration"], titles=["Момент со стрима"],
                                     title="Момент со стрима",
                                     category="рейдж" if c.get("trigger") == "rage" else "другое")
            return
        if not await self._budget_ok(c):
            return
        d = Path(c["dir"])
        peak_rel = c["peak_ts"] - c["raw_start_ts"]
        frames = await extract_frames(self.ffmpeg, d / "raw.mp4", d, spread(c["raw_duration"], 3, peak_rel, 30), "s")
        log.info("🤖 #%d Haiku оценивает момент…", c["id"])
        res = await self.llm.screen(await self._context(c), frames)
        thr = int(self.cfg.section("llm").get("screen_threshold", 6))
        status = "screened" if res["score"] >= thr else "discarded"
        verdict = "дальше в Sonnet" if status == "screened" else f"отсеян (порог {thr})"
        log.info("🤖 #%d Haiku: %d/10, %s — %s · %s", c["id"], res["score"], res["category"],
                 res.get("summary", "")[:80], verdict)
        self.db.update_candidate(c["id"], status=status, screen=res, category=res["category"], attempts=0, error=None)

    async def _do_review(self, c: dict) -> None:
        if not await self._budget_ok(c):
            return
        d = Path(c["dir"])
        frames = await extract_frames(self.ffmpeg, d / "raw.mp4", d, spread(c["raw_duration"], 6), "r")
        log.info("🎬 #%d Sonnet выбирает границы и заголовки…", c["id"])
        res = await self.llm.review(await self._context(c, c.get("screen")), frames, c["raw_duration"])
        if not res["keep"]:
            log.info("🎬 #%d Sonnet: не годится для ролика — отсеян", c["id"])
            self.db.update_candidate(c["id"], status="discarded", review=res)
            return
        log.info("🎬 #%d Sonnet: %d/10, %s, %.0f–%.0f с (%.0f с)%s, «%s»", c["id"], res["score"], res["category"],
                 res["start"], res["end"], res["end"] - res["start"], ", СПОЙЛЕР" if res["spoiler"] else "",
                 res["titles"][0])
        self.db.update_candidate(c["id"], status="to_render", review=res, clip_start=res["start"],
                                 clip_end=res["end"], titles=res["titles"], title=res["titles"][0],
                                 category=res["category"], spoiler=int(res["spoiler"]), tag=res["tag"],
                                 description=res["description"], score=res["score"], attempts=0, error=None)

    async def _do_render(self, c: dict) -> None:
        if c["kind"] == "compilation":
            return await self._render_compilation(c)
        d = Path(c["dir"])
        raw = d / "raw.mp4"
        info = await probe(self.ffprobe, raw)
        start, end = (c["edit_start"], c["edit_end"]) if c.get("edit_start") is not None else (c["clip_start"], c["clip_end"])
        start, end = max(0.0, start), min(info["duration"] or c["raw_duration"], end)
        ch = self.db.channel(c["channel"]) or {}
        mode = self.cam_mode(c["channel"])
        cam = self.cam_rect(c["channel"], ch)
        top_media = None
        if mode == "off":
            cam = None
        elif mode == "media":
            cam = None
            top_media, _ = self.channel_overlay(c["channel"])
            if not top_media:
                log.warning("#%d режим «заставка», но файла заставки нет — монтирую без вебки", c["id"])
        if cam and not await self._cam_present(raw, d, cam, start, end):
            # стример поменял раскладку (полноэкранная вебка, сцена «отошёл») — макет без вебки
            log.info("🎥 #%d в рамке вебки нет лица — монтирую без вебки", c["id"])
            cam = None
        tr = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
        out = d / "clip.mp4"
        layout = "заставка сверху" if top_media else "вебка сверху" if cam else "без вебки"
        log.info("🎞  #%d монтирую 9:16 (%s, %.0f с)…", c["id"], layout, end - start)
        t0 = time.time()
        await render_clip(self.ffmpeg, self.encoder, raw, out, start, end, info["width"], info["height"], cam,
                          words_in(tr, start, end), c.get("title") or "",
                          f"twitch.tv/{c['channel']}", self.cfg.section("render"), top_media=top_media)
        log.info("✅ #%d клип готов за %.0f с (%.1f МБ) → на модерацию", c["id"], time.time() - t0,
                 out.stat().st_size / 1e6)
        self.db.update_candidate(c["id"], status="rendered", render_path=str(out), render_duration=end - start,
                                 clip_start=start, clip_end=end, edit_start=None, edit_end=None,
                                 card_msg_id=None, attempts=0, error=None)

    async def _cam_present(self, raw: Path, d: Path, cam, start: float, end: float) -> bool:
        if not self.cfg.section("render").get("check_cam", True):
            return True
        frames = await extract_frames(self.ffmpeg, raw, d, spread(end - start, 8, None, None), "cam", scale=None,
                                      offset=start)
        if not frames:
            return True
        code, out = await run([sys.executable, "-m", "clipper.camcal", "check", *map(str, cam),
                               *(str(p) for _, p in frames)], timeout=120)
        for _, p in frames:
            p.unlink(missing_ok=True)
        try:
            return bool(json.loads(out.strip().splitlines()[-1])["present"]) if code == 0 else True
        except (ValueError, KeyError, IndexError):
            log.warning("проверка вебки: %s", out[-300:])
            return True  # проверка сломалась — не портим клип, оставляем вебку

    # ---------- подборки ----------
    def compilation_tags(self) -> list[tuple[str, str, int]]:
        """Темы для подборок: (игра, тема, сколько клипов) — клипы разных игр не смешиваются."""
        rows = self.db.all("SELECT COALESCE(game,'') g, tag, COUNT(*) n FROM candidates WHERE kind='clip' "
                           "AND status IN ('approved','sent','done') AND in_compilation=0 AND tag IS NOT NULL "
                           "AND tag!='' GROUP BY g, tag ORDER BY n DESC")
        need = int(self.cfg.section("compilation").get("min_clips", 3))
        return [(r["g"], r["tag"], r["n"]) for r in rows if r["n"] >= need]

    def create_compilation(self, game: str, tag: str) -> int | None:
        rows = self.db.all("SELECT * FROM candidates WHERE kind='clip' AND status IN ('approved','sent','done') "
                           "AND in_compilation=0 AND tag=? AND COALESCE(game,'')=? ORDER BY score DESC", (tag, game))
        max_s = float(self.cfg.section("compilation").get("max_seconds", 90))
        picked, total = [], 0.0
        for r in rows:
            dur = r.get("render_duration") or 0
            if r.get("render_path") and Path(r["render_path"]).exists() and total + dur <= max_s:
                picked.append(r)
                total += dur
        if len(picked) < int(self.cfg.section("compilation").get("min_clips", 3)):
            return None
        picked.sort(key=lambda r: r["id"])
        streamers = sorted({r["channel"] for r in picked})
        title = f"{tag.capitalize()}: {len(streamers)} стрим." if len(streamers) > 1 else tag.capitalize()
        cid = self.db.add_candidate(kind="compilation", channel=",".join(streamers), status="to_render", game=game or None,
                                    parts=[r["id"] for r in picked], title=title, titles=[title], tag=tag,
                                    category="подборка", score=max(r["score"] or 0 for r in picked),
                                    description="Стримеры: " + ", ".join(f"twitch.tv/{s}" for s in streamers))
        for r in picked:
            self.db.update_candidate(r["id"], in_compilation=1)
        return cid

    def release_parts(self, c: dict) -> None:
        """Подборка не состоялась — её клипы снова доступны для других подборок."""
        for i in c.get("parts") or []:
            self.db.update_candidate(i, in_compilation=0)

    async def _render_compilation(self, c: dict) -> None:
        parts = [p for p in (self.db.candidate(i) for i in c["parts"])
                 if p and p.get("render_path") and Path(p["render_path"]).exists()]
        if len(parts) < 2:
            raise CutError("клипы для подборки уже удалены", permanent=True)
        paths = [Path(p["render_path"]) for p in parts]
        total = sum(p.get("render_duration") or 0 for p in parts)
        d = self.cand_dir(c)
        d.mkdir(parents=True, exist_ok=True)
        out = d / "clip.mp4"
        await render_compilation(self.ffmpeg, self.encoder, paths, out, self.cfg.section("render"), total)
        self.db.update_candidate(c["id"], status="rendered", dir=str(d), render_path=str(out),
                                 render_duration=total, card_msg_id=None, attempts=0, error=None)

    # ---------- вебка ----------
    async def calibrate_cam(self, login: str) -> None:
        try:
            if await self._calibrate_cam(login):
                return
        except Exception:  # noqa: BLE001
            log.exception("[%s] поиск вебки", login)
        # не получилось — попробуем снова через 5 минут
        self.db.kv_set(f"cam_next:{login}", time.time() + 120)
        ch = self.db.channel(login)
        if ch and ch["cam_status"] == "pending":
            self.db.update_channel(login, cam_status="unknown")

    async def _calibrate_cam(self, login: str) -> bool:
        rec = self.recorders.get(login)
        if not rec:
            return False
        # 24 кадра из последних ~3 минут: край вебки постоянен, игра вокруг меняется
        segs = (await asyncio.to_thread(load_session_segments, rec.session_dir))[-45:]
        if len(segs) < 8:
            return False
        n = min(24, len(segs))
        picks = [segs[int(i * (len(segs) - 1) / (n - 1))] for i in range(n)]
        work = self.cfg.data_dir / "cam" / login
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, s in enumerate(picks):
            p = work / f"f{i:02d}.jpg"
            code, _ = await run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(s.path),
                                 "-frames:v", "1", "-q:v", "3", str(p)], timeout=60)
            if code == 0 and p.exists():
                frames.append(str(p))
        if len(frames) < 6:
            return False
        preview = work / "preview.jpg"
        code, out = await run([sys.executable, "-m", "clipper.camcal", "find", str(preview), *frames], timeout=180)
        try:
            res = json.loads(out.strip().splitlines()[-1]) if code == 0 else {}
        except (ValueError, IndexError):
            res = {}
        if not res:
            log.warning("camcal: %s", out[-500:])
            return False
        if not res.get("rect"):
            # лица нет (стример отошёл, заставка между играми) — не беспокоим, попробуем позже
            log.info("🎥 [%s] лица в кадре нет (стример отошёл?) — поищу вебку через 5 минут", login)
            return False
        # последний кадр — основа для перерисовки рамки при ручной подстройке
        shutil.copy(frames[-1], work / "base.jpg")
        self.db.kv_set(f"cam:{login}", res.get("rect"))
        self.db.kv_set(f"cam_method:{login}", res.get("method"))
        log.info("🎥 [%s] вебка найдена (%s) — %s", login,
                 "по краям рамки" if res.get("method") == "edges" else "примерно, по лицу",
                 "жду подтверждения в боте" if self.bot else
                 ("принята автоматически" if res.get("method") == "edges" else "без бота не принята"))
        if self.bot:
            await self.bot.send_cam_proposal(login, preview if preview.exists() else None, res.get("rect"),
                                             res.get("method"))
        elif res.get("method") == "edges":  # без бота принимаем только уверенную рамку
            self.accept_cam(login, res.get("rect"))
        else:
            self.accept_cam(login, None)
        return True

    def _maybe_calibrate(self, login: str, rec: Recorder, now: float) -> None:
        """Искать вебку, как только накопилось ~35 с записи (8+ сегментов), а не через минуту."""
        if now - rec.started_at < 35 or now < (self.db.kv_get(f"cam_next:{login}") or 0):
            return
        ch = self.db.channel(login)
        if ch and ch["cam_status"] == "unknown" and self.cam_mode(login) == "auto":
            self.db.update_channel(login, cam_status="pending")
            self.spawn(self.calibrate_cam(login))

    def cam_rect(self, login: str, ch: dict | None = None):
        """Рамка вебки для монтажа. Пока вы не ответили в боте, используется уже найденная по краям или
        по вырезке рамка — иначе клипы до подтверждения монтировались бы без вебки."""
        ch = ch or self.db.channel(login) or {}
        if ch.get("cam_status") == "ok":
            return ch["cam_x"], ch["cam_y"], ch["cam_w"], ch["cam_h"]
        if ch.get("cam_status") == "pending" and self.db.kv_get(f"cam_method:{login}") in ("edges", "template", "manual"):
            rect = self.db.kv_get(f"cam:{login}")
            if rect:
                return tuple(rect)
        return None

    async def grab_frames(self, login: str, n: int = 3) -> list[Path]:
        """Свежие кадры канала: из идущей записи, иначе — сохранённый кадр последнего поиска вебки."""
        work = self.cfg.data_dir / "cam" / login
        work.mkdir(parents=True, exist_ok=True)
        rec = self.recorders.get(login)
        frames: list[Path] = []
        if rec:
            segs = (await asyncio.to_thread(load_session_segments, rec.session_dir))[-12:]
            for i, sgm in enumerate(segs[:: max(1, len(segs) // n)][:n]):
                p = work / f"g{i}.jpg"
                code, _ = await run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(sgm.path),
                                     "-frames:v", "1", "-q:v", "3", str(p)], timeout=60)
                if code == 0 and p.exists():
                    frames.append(p)
        if not frames and (work / "base.jpg").exists():
            frames = [work / "base.jpg"]
        return frames

    async def match_cam_template(self, login: str, template: Path) -> tuple[Path | None, dict]:
        """Рамка по присланной вырезке вебки. Возвращает (кадр с рамкой, результат)."""
        frames = await self.grab_frames(login)
        if not frames:
            return None, {"rect": None, "error": "нет кадров: канал не в эфире и ещё ни разу не записывался"}
        work = self.cfg.data_dir / "cam" / login
        preview = work / "preview.jpg"
        code, out = await run([sys.executable, "-m", "clipper.camcal", "match", str(template), str(preview),
                               *map(str, frames)], timeout=180)
        try:
            res = json.loads(out.strip().splitlines()[-1]) if code == 0 else {"rect": None, "error": out[-300:]}
        except (ValueError, IndexError):
            res = {"rect": None, "error": out[-300:]}
        if res.get("rect"):
            if frames[-1] != work / "base.jpg":
                shutil.copy(frames[-1], work / "base.jpg")
            self.db.kv_set(f"cam:{login}", res["rect"])
            self.db.kv_set(f"cam_method:{login}", "template")
            if (self.db.channel(login) or {}).get("cam_status") != "ok":
                self.db.update_channel(login, cam_status="pending")
            log.info("🎥 [%s] вебка найдена по присланной вырезке (совпадение %.2f)", login, res.get("score", 0))
        return (preview if preview.exists() else None), res

    async def draw_cam_preview(self, login: str, rect) -> Path | None:
        """Перерисовывает рамку на сохранённом кадре (для ручной подстройки)."""
        work = self.cfg.data_dir / "cam" / login
        base, out = work / "base.jpg", work / "adjust.jpg"
        if not base.exists():
            return None
        code, log_out = await run([sys.executable, "-m", "clipper.camcal", "draw", str(base), str(out),
                                   *map(str, rect)], timeout=60)
        if code != 0:
            log.warning("рамка вебки: %s", log_out[-300:])
            return None
        return out

    def accept_cam(self, login: str, rect) -> None:
        if rect:
            x, y, w, h = rect
            self.db.update_channel(login, cam_status="ok", cam_x=x, cam_y=y, cam_w=w, cam_h=h)
        else:
            self.db.update_channel(login, cam_status="none")

    # ---------- отсеянные кандидаты ----------
    def rejected_list(self, limit: int = 40) -> list[dict]:
        """Кандидаты, которых отсеял ИИ или которые упали с ошибкой (пока их файлы не удалены уборкой)."""
        return self.db.all("SELECT * FROM candidates WHERE kind='clip' AND status IN ('discarded','failed') "
                           "AND dir IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))

    @staticmethod
    def reject_reason(c: dict) -> str:
        screen, review = c.get("screen") or {}, c.get("review") or {}
        if c["status"] == "failed":
            return f"ошибка обработки: {(c.get('error') or 'неизвестно')[:300]}"
        if isinstance(review, dict) and review and not review.get("keep", True):
            return f"Sonnet решил, что для ролика не годится (оценка {review.get('score', '?')}/10)"
        if isinstance(screen, dict) and "score" in screen:
            return f"Haiku: {screen['score']}/10 — ниже порога. {screen.get('summary', '')}".strip()
        return "отсеян"

    async def preview(self, c: dict) -> Path:
        """Лёгкая версия фрагмента (480p) для просмотра в Telegram: исходник 1080p бывает больше 50 МБ."""
        d = Path(c["dir"])
        out = d / "preview.mp4"
        if out.exists():
            return out
        raw = d / "raw.mp4"
        if not raw.exists():
            raise CutError("фрагмент уже удалён уборкой", permanent=True)
        code, log_out = await run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                                   "-vf", "scale=-2:480,fps=30", *encoder_args(self.encoder, 1200),
                                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                                   str(out)], timeout=300)
        if code != 0 or not out.exists():
            raise RuntimeError(f"превью не получилось: {log_out[-300:]}")
        return out

    def rescue(self, cid: int) -> str | None:
        """Вернуть отсеянного кандидата в работу. Возвращает описание, куда он пошёл, или None."""
        c = self.db.candidate(cid)
        if not c or c["status"] not in ("discarded", "failed") or not c.get("dir"):
            return None
        d = Path(c["dir"])
        if not (d / "raw.mp4").exists():
            self.db.update_candidate(cid, status="to_cut", attempts=0, error=None, next_try_at=0)
            return "заново на вырезку"
        if not (d / "transcript.json").exists():
            self.db.update_candidate(cid, status="cut", attempts=0, error=None, next_try_at=0)
            return "на распознавание речи"
        screen, review = c.get("screen") or {}, c.get("review") or {}
        if isinstance(review, dict) and review.get("titles"):  # Sonnet уже выбрал границы — берём их
            start, end, titles = review["start"], review["end"], review["titles"]
        else:
            summary = (screen.get("summary") if isinstance(screen, dict) else "") or "Момент со стрима"
            start, end, titles = 0.0, c["raw_duration"], [summary[:60]]
        self.db.update_candidate(cid, status="to_render", clip_start=start, clip_end=end, titles=titles,
                                 title=titles[0], category=c.get("category") or "другое", attempts=0,
                                 error=None, next_try_at=0, edit_start=None, edit_end=None)
        log.info("♻️  #%d возвращён из отсеянных — в монтаж и на модерацию", cid)
        return "в монтаж, потом придёт на модерацию"

    # ---------- режим вебки и заставки ----------
    def cam_mode(self, login: str) -> str:
        """auto | off | media — режим канала, если задан, иначе общий по умолчанию."""
        ch = self.db.channel(login) or {}
        mode = ch.get("cam_mode") or self.setting("cam_default") or "auto"
        return mode if mode in ("auto", "off", "media") else "auto"

    def overlay_dir(self) -> Path:
        d = self.cfg.data_dir / "overlays"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def overlay_path(self, key: str) -> Path | None:
        """Файл заставки: key — логин канала или "_default" (общая)."""
        files = sorted(self.overlay_dir().glob(f"{key}.*"))
        return files[0] if files else None

    def channel_overlay(self, login: str) -> tuple[Path | None, str]:
        own = self.overlay_path(login)
        if own:
            return own, "своя"
        common = self.overlay_path("_default")
        return (common, "общая") if common else (None, "нет файла")

    # ---------- уборка ----------
    async def _janitor_loop(self) -> None:
        cl = self.cfg.section("cleanup")
        while True:
            try:
                await asyncio.to_thread(self._cleanup, cl)  # rmtree тысяч файлов не должен тормозить запись
                free = shutil.disk_usage(self.cfg.data_dir).free / 1e9
                if free < float(cl.get("min_free_gb", 20)):
                    await self.alert("disk", f"На диске осталось {free:.0f} ГБ", every=3600)
            except Exception:  # noqa: BLE001
                log.exception("ошибка уборки")
            await asyncio.sleep(600)

    def _cleanup(self, cl: dict) -> None:
        now = time.time()
        short = now - float(cl.get("keep_rejected_days", 1)) * 86400
        long = now - float(cl.get("keep_clips_days", 7)) * 86400
        rows = self.db.all("SELECT id, dir, status, updated_at FROM candidates WHERE dir IS NOT NULL")
        for r in rows:
            old = (r["status"] in ("rejected", "discarded", "failed") and r["updated_at"] < short) or r["updated_at"] < long
            if old and Path(r["dir"]).exists() and r["status"] in FINAL + ("sent", "approved"):
                shutil.rmtree(r["dir"], ignore_errors=True)
                self.db.update_candidate(r["id"], dir=None, render_path=None)
        # записи закончившихся сессий старше буфера, если на них не ссылаются необработанные кандидаты
        buf = float(self.cfg.section("recording").get("buffer_hours", 3)) * 3600
        for s in self.db.all("SELECT * FROM sessions WHERE ended_at IS NOT NULL AND ended_at<?", (now - buf,)):
            busy = self.db.scalar("SELECT COUNT(*) FROM candidates WHERE session_id=? AND status IN ('to_cut')",
                                  (s["id"],))
            if not busy and Path(s["dir"]).exists():
                for part in Path(s["dir"]).glob("part_*"):
                    shutil.rmtree(part, ignore_errors=True)
        # идущие сессии: старые части (после переподключений) ffmpeg сам не чистит
        for rec in list(self.recorders.values()):
            current = rec.session_dir / f"part_{rec.part:03d}"
            for part in rec.session_dir.glob("part_*"):
                seg_log = part / "segments.jsonl"
                last = seg_log.stat().st_mtime if seg_log.exists() else part.stat().st_mtime
                if part != current and last < now - buf - 600:
                    shutil.rmtree(part, ignore_errors=True)
            # сегменты текущей части, которые ffmpeg не смог удалить (файл был занят на Windows)
            for seg in current.glob("seg_*.ts"):
                try:
                    if seg.stat().st_mtime < now - buf - 1800:
                        seg.unlink()
                except OSError:
                    pass

    # ---------- статус ----------
    def status(self) -> dict:
        today = day_start()
        q = lambda sql, *a: self.db.scalar(sql, a) or 0  # noqa: E731
        return {
            "recording": len(self.recorders),
            "healthy": sum(1 for r in self.recorders.values() if r.healthy),
            "channels": len(self.db.channels(("active", "paused", "removing"))),
            "moderation": self.db.count("rendered"),
            "publish": q("SELECT COUNT(*) FROM candidates WHERE status='approved'"),
            "processing": q("SELECT COUNT(*) FROM candidates WHERE status IN ('to_cut','cut','transcribed','screened','to_render')"),
            "cand_today": q("SELECT COUNT(*) FROM candidates WHERE kind='clip' AND created_at>=?", today),
            "approved_today": q("SELECT COUNT(*) FROM candidates WHERE decided_at>=? AND status IN ('approved','sent','done')", today),
            "posted_today": q("SELECT COUNT(*) FROM candidates WHERE sent_at>=? AND (pub_shorts=1 OR pub_tiktok=1)", today),
            "free_gb": shutil.disk_usage(self.cfg.data_dir).free / 1e9,
            "spend_today": self.db.spend_since(today),
            "chat": self.chat.connected,
            "llm": self.llm.enabled,
            "twitch_api": self.twitch.enabled,
            "encoder": self.encoder,
            "paused": bool(self.setting("paused")),
        }
