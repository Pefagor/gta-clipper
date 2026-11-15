"""Индекс записанных сегментов: какой файл какой отрезок реального времени покрывает.

Время сегмента считается по моменту его появления на диске, а не по меткам в потоке:
так вырезанные рекламные паузы и обрывы не сбивают сопоставление с чатом.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

GAP_TOLERANCE = 3.0  # секунд: больший разрыв между сегментами считается дырой


@dataclass
class Segment:
    path: Path
    start: float
    end: float


def parse_playlist(text: str) -> list[tuple[str, float]]:
    """[(имя файла, длительность)] из m3u8."""
    out, dur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            try:
                dur = float(line[8:].split(",")[0])
            except ValueError:
                dur = None
        elif line and not line.startswith("#") and dur is not None:
            out.append((line, dur))
            dur = None
    return out


class PartIndexer:
    """Следит за плейлистом одной части записи и дописывает новые сегменты в segments.jsonl."""

    def __init__(self, part_dir: Path):
        self.part_dir = part_dir
        self.playlist = part_dir / "index.m3u8"
        self.log_path = part_dir / "segments.jsonl"
        self.seen: set[str] = set()
        self.last_new_at = time.time()
        self.last_end = 0.0

    def poll(self, now: float | None = None) -> list[Segment]:
        now = now or time.time()
        try:
            entries = parse_playlist(self.playlist.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):  # на Windows файл может быть занят в момент переименования
            return []
        new = [(n, d) for n, d in entries if n not in self.seen and (self.part_dir / n).exists()]
        if not new:
            return []
        # Несколько сегментов разом (например, на старте): раскладываем их назад от текущего момента.
        segs: list[Segment] = []
        end = now
        for name, dur in reversed(new):
            start = end - dur
            segs.append(Segment(self.part_dir / name, start, end))
            end = start
        segs.reverse()
        # не допускаем наложения на предыдущие сегменты
        if segs and segs[0].start < self.last_end:
            shift = self.last_end - segs[0].start
            if shift < 2.0:
                segs = [Segment(s.path, s.start + shift, s.end + shift) for s in segs]
        with self.log_path.open("a", encoding="utf-8") as f:
            for s in segs:
                self.seen.add(s.path.name)
                f.write(json.dumps({"file": s.path.name, "start": s.start, "end": s.end}) + "\n")
        self.last_new_at = now
        self.last_end = segs[-1].end
        return segs


def load_session_segments(session_dir: Path) -> list[Segment]:
    segs: list[Segment] = []
    for log_path in sorted(session_dir.glob("part_*/segments.jsonl")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            p = log_path.parent / d["file"]
            if p.exists():
                segs.append(Segment(p, d["start"], d["end"]))
    segs.sort(key=lambda s: s.start)
    return segs


def pick_window(segs: list[Segment], start: float, end: float, anchor: float) -> list[Segment]:
    """Сплошной (без дыр и смены части) кусок сегментов внутри [start, end], содержащий anchor."""
    inside = [s for s in segs if s.end > start and s.start < end]
    if not inside:
        return []
    runs: list[list[Segment]] = [[inside[0]]]
    for s in inside[1:]:
        prev = runs[-1][-1]
        if s.path.parent != prev.path.parent or s.start - prev.end > GAP_TOLERANCE:
            runs.append([s])
        else:
            runs[-1].append(s)
    for run in runs:
        if run[0].start - 1 <= anchor <= run[-1].end + 1:
            return run
    return max(runs, key=lambda r: r[-1].end - r[0].start)
