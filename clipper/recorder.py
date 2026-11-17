"""Запись одного канала: streamlink -> ffmpeg (HLS-сегменты без перекодирования + звук для анализа)."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from .segments import PartIndexer
from .tools import streamlink_cmd

log = logging.getLogger(__name__)

AUDIO_RATE = 16000
AUDIO_CHUNK = AUDIO_RATE // 2 * 2  # 0.5 с моно s16le в байтах


class Recorder:
    def __init__(self, login: str, session_id: int, session_dir: Path, cfg, ffmpeg: str,
                 on_alert=None):
        self.login = login
        self.session_id = session_id
        self.session_dir = session_dir
        self.cfg = cfg
        self.rec = cfg.section("recording")
        self.ffmpeg = ffmpeg
        self.on_alert = on_alert
        self.audio: deque[tuple[float, float]] = deque(maxlen=4 * 3600)  # (время, громкость дБ) каждые 0.5 с
        self.started_at = time.time()
        self.game = ""  # категория Twitch, обновляется наблюдателем
        self.part = 0
        self.restarts = 0
        self.stop_requested = False
        self.healthy = False
        self.last_segment_at = 0.0
        self._procs: list[asyncio.subprocess.Process] = []
        self._task: asyncio.Task | None = None

    # --- управление ---
    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"rec-{self.login}")

    async def stop(self) -> None:
        self.stop_requested = True
        await self._kill()
        if self._task and not self._task.done():
            try:  # даём задаче корректно закрыть логи и дописать индекс
                await asyncio.wait_for(asyncio.shield(self._task), 20)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                self._task.cancel()
        self.healthy = False

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done())

    # --- основной цикл с перезапусками ---
    async def _run(self) -> None:
        backoff = 5
        while not self.stop_requested:
            self.part += 1
            part_dir = self.session_dir / f"part_{self.part:03d}"
            t0 = time.time()
            try:
                part_dir.mkdir(parents=True, exist_ok=True)
                reason = await self._run_part(part_dir)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — запись не должна умирать молча
                log.exception("[%s] ошибка записи", self.login)
                reason = f"ошибка: {e}"
            if self.stop_requested:
                break
            self.healthy = False
            self.restarts += 1
            log.warning("[%s] запись прервалась (%s), перезапуск через %d с", self.login, reason, backoff)
            if self.on_alert and self.restarts % 3 == 0:
                try:
                    await self.on_alert(self.login, reason)
                except Exception:  # noqa: BLE001
                    log.exception("уведомление о записи")
            await asyncio.sleep(backoff)
            backoff = 5 if time.time() - t0 > 300 else min(backoff * 2, 60)

    async def _run_part(self, part_dir: Path) -> str:
        seg = int(self.rec.get("segment_seconds", 4))
        list_size = int(self.rec.get("buffer_hours", 3) * 3600 / seg)
        r_fd, w_fd = os.pipe()
        sl_log = open(part_dir / "streamlink.log", "ab")
        ff_log = open(part_dir / "ffmpeg.log", "ab")
        try:
            streamlink = await asyncio.create_subprocess_exec(
                *streamlink_cmd(self.cfg.get("streamlink", "auto")), *(self.rec.get("streamlink_extra_args") or []),
                "--loglevel", "warning", "-O", f"https://twitch.tv/{self.login}", self.rec.get("quality", "best"),
                stdout=w_fd, stderr=sl_log)
            ffmpeg = await asyncio.create_subprocess_exec(
                self.ffmpeg, "-hide_banner", "-loglevel", "warning", "-i", "pipe:0",
                "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                "-f", "hls", "-hls_time", str(seg), "-hls_list_size", str(list_size),
                "-hls_flags", "delete_segments+independent_segments+temp_file",
                "-hls_segment_filename", "seg_%06d.ts", "index.m3u8",
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "pipe:1",
                stdin=r_fd, stdout=asyncio.subprocess.PIPE, stderr=ff_log, cwd=str(part_dir))
        finally:
            os.close(r_fd)
            os.close(w_fd)
        self._procs = [streamlink, ffmpeg]
        indexer = PartIndexer(part_dir)
        audio_task = asyncio.create_task(self._read_audio(ffmpeg.stdout))
        reason = "процесс завершился"
        try:
            stall = float(self.rec.get("stall_seconds", 30))
            started = time.time()
            while not self.stop_requested:
                await asyncio.sleep(1)
                if indexer.poll():
                    self.last_segment_at = time.time()
                    if not self.healthy:
                        log.info("📼 [%s] запись идёт, сегменты пишутся", self.login)
                    self.healthy = True
                if streamlink.returncode is not None or ffmpeg.returncode is not None:
                    reason = streamlink_reason(part_dir / "streamlink.log") or "стрим закончился или пропало соединение"
                    break
                ref = indexer.last_new_at if indexer.seen else started
                if time.time() - ref > stall + (30 if not indexer.seen else 0):
                    reason = f"нет новых сегментов {int(stall)} с"
                    break
        finally:
            try:
                await self._kill()
            finally:
                audio_task.cancel()
                indexer.poll()
                sl_log.close()
                ff_log.close()
        return reason

    async def _read_audio(self, stream: asyncio.StreamReader) -> None:
        buf = b""
        try:
            while True:
                data = await stream.read(AUDIO_CHUNK)
                if not data:
                    return
                buf += data
                while len(buf) >= AUDIO_CHUNK:
                    chunk, buf = buf[:AUDIO_CHUNK], buf[AUDIO_CHUNK:]
                    self.audio.append((time.time(), loudness_db(chunk)))
        except asyncio.CancelledError:
            pass

    async def _kill(self) -> None:
        for p in self._procs:
            if p.returncode is None:
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass
        for p in self._procs:
            try:
                await asyncio.wait_for(p.wait(), 10)
            except asyncio.TimeoutError:
                p.kill()
                await p.wait()
        self._procs = []


def streamlink_reason(log_path: Path) -> str:
    """Причина обрыва из последних строк лога streamlink — понятным языком."""
    from .errors import explain
    try:
        lines = [l for l in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    errs = [l for l in lines[-15:] if "error" in l.lower() or "unable" in l.lower() or "no playable" in l.lower()]
    return explain(errs[-1]) if errs else ""


def loudness_db(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    return 20 * math.log10(max(rms, 1e-5))
