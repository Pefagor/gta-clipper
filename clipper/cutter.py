"""Быстрая вырезка кандидата из сегментов (без перекодирования), кадры и звук для анализа."""
from __future__ import annotations

from pathlib import Path

from .segments import load_session_segments, pick_window
from .tools import run


class CutError(Exception):
    pass


async def cut_candidate(ffmpeg: str, session_dir: Path, out_dir: Path, start: float, end: float,
                        anchor: float) -> tuple[Path, float, float]:
    """Склеивает сегменты окна в raw.mp4. Возвращает (путь, время начала, длительность)."""
    segs = pick_window(load_session_segments(session_dir), start, end, anchor)
    if not segs:
        raise CutError("нет сегментов для этого окна (запись удалена или ещё не началась)")
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / "concat.txt"
    lst.write_text("".join(f"file '{s.path.as_posix()}'\n" for s in segs), encoding="utf-8")
    raw = out_dir / "raw.mp4"
    code, out = await run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                           "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(raw)], timeout=120)
    if code != 0 or not raw.exists():
        raise CutError(f"ffmpeg concat: {out[-300:]}")
    return raw, segs[0].start, segs[-1].end - segs[0].start


async def extract_audio(ffmpeg: str, raw: Path, wav: Path) -> None:
    code, out = await run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-vn",
                           "-ac", "1", "-ar", "16000", str(wav)], timeout=120)
    if code != 0:
        raise CutError(f"ffmpeg audio: {out[-300:]}")


async def extract_frames(ffmpeg: str, raw: Path, out_dir: Path, times: list[float], prefix: str) -> list[tuple[float, Path]]:
    frames = []
    for i, t in enumerate(times):
        p = out_dir / f"{prefix}_{i:02d}.jpg"
        code, _ = await run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(t, 0):.2f}",
                             "-i", str(raw), "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", str(p)],
                            timeout=60)
        if code == 0 and p.exists():
            frames.append((t, p))
    return frames


def spread(duration: float, n: int, center: float | None = None, span: float | None = None) -> list[float]:
    """n моментов равномерно по [0, duration] или вокруг center в пределах span."""
    if center is not None and span:
        lo, hi = max(0.0, center - span / 2), min(duration, center + span / 2)
    else:
        lo, hi = 0.5, max(duration - 0.5, 0.5)
    if n == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]
