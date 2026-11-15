"""Поиск ffmpeg/ffprobe, выбор кодировщика, запуск внешних команд."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_FFMPEG_CANDIDATES = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",   # macOS: brew install ffmpeg-full
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
    "ffmpeg",
]


def _has_libass(ffmpeg: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split()[1:2] == ["ass"] for line in out.splitlines() if line.strip())


@lru_cache(maxsize=None)
def find_ffmpeg(setting: str = "auto") -> str:
    if setting and setting != "auto":
        return setting
    found = []
    candidates = list(_FFMPEG_CANDIDATES)
    if sys.platform == "win32":
        # winget прописывает PATH только для новых окон терминала — ищем и в его папках напрямую
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates.insert(0, str(local / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
        candidates += [str(p) for p in sorted((local / "Microsoft" / "WinGet" / "Packages").glob(
            "Gyan.FFmpeg*/*/bin/ffmpeg.exe"), reverse=True)]
    for cand in candidates:
        path = cand if Path(cand).is_absolute() else shutil.which(cand)
        if path and Path(path).exists():
            found.append(path)
            if _has_libass(path):
                return path
    if found:
        log.warning("ffmpeg без libass (субтитры не будут работать): %s", found[0])
        return found[0]
    raise RuntimeError("ffmpeg не найден. Установите его (Windows: winget install Gyan.FFmpeg) и откройте "
                       "новое окно терминала, либо укажите полный путь в config.yaml (ffmpeg: ...)")


def streamlink_cmd(setting: str = "auto") -> list[str]:
    """Команда запуска streamlink. По умолчанию — из виртуального окружения программы
    (python -m streamlink), так он не зависит от PATH терминала."""
    if setting and setting != "auto":
        return [setting]
    try:
        import streamlink  # noqa: F401
        return [sys.executable, "-m", "streamlink"]
    except ImportError as e:
        path = shutil.which("streamlink")
        if path:
            return [path]
        raise RuntimeError(f"streamlink не найден в окружении {sys.executable} ({e}). "
                           "Активируйте окружение и выполните: pip install -e .") from e


@lru_cache(maxsize=None)
def find_ffprobe(setting: str = "auto", ffmpeg: str = "ffmpeg") -> str:
    if setting and setting != "auto":
        return setting
    exe = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    sibling = Path(ffmpeg).with_name(exe)
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"


@lru_cache(maxsize=None)
def pick_encoder(ffmpeg: str, setting: str = "auto") -> str:
    if setting and setting != "auto":
        return setting
    for enc in ("h264_nvenc", "h264_videotoolbox"):
        try:
            r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                "color=c=black:s=320x240:d=0.2", "-c:v", enc, "-f", "null", "-"],
                               capture_output=True, timeout=30)
            if r.returncode == 0:
                return enc
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "libx264"


def encoder_args(encoder: str, kbps: int) -> list[str]:
    rate = [f"-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.3)}k", "-bufsize", f"{kbps * 2}k"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", *rate]
    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{kbps}k"]
    return ["-c:v", "libx264", "-preset", "veryfast", *rate]


async def run(cmd: list[str], cwd: Path | None = None, timeout: float = 600, tail: int | None = 3000) -> tuple[int, str]:
    """Запуск команды, возвращает (код, вывод stderr+stdout; по умолчанию только хвост)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "timeout"
    text = out.decode("utf-8", "replace")
    return proc.returncode, text[-tail:] if tail else text


async def probe(ffprobe: str, path: Path) -> dict:
    code, out = await run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                          timeout=60, tail=None)
    if code != 0:
        raise RuntimeError(f"ffprobe: {out[-300:]}")
    data = json.loads(out)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "duration": float(data.get("format", {}).get("duration") or 0)}
