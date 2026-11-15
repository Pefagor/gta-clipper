"""Детектор интересных моментов: всплески чата, громкости, эмоуты и клипы зрителей."""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field


SILENCE_DB = -60.0


@dataclass
class Sample:
    t: float
    chat: int
    emotes: int
    audio: float | None


def count_recent(history, start: float, end: float, emotes: list[str]) -> tuple[int, int]:
    """Сообщений и сообщений с эмоутами в [start, end]. history — по возрастанию времени."""
    n = e = 0
    lowered = [x.lower() for x in emotes]
    for t, text in reversed(history):
        if t < start:
            break
        if t > end:
            continue
        n += 1
        low = text.lower()
        if any(x in low for x in lowered):
            e += 1
    return n, e


def mean_audio(audio, start: float, end: float) -> float | None:
    vals = []
    for t, db in reversed(audio):
        if t < start:
            break
        if t <= end:
            vals.append(db)
    return sum(vals) / len(vals) if vals else None


def zscore(x: float, values: list[float], min_std: float) -> float:
    if len(values) < 5:
        return 0.0
    mu = statistics.fmean(values)
    sd = max(statistics.pstdev(values), min_std)
    return (x - mu) / sd


@dataclass
class ChannelDetector:
    cfg: dict
    samples: deque = field(default_factory=lambda: deque(maxlen=2000))
    started_at: float | None = None

    def tick(self, now: float, chat_history, audio, clip_moments: list[float]) -> tuple[float, dict]:
        """Возвращает (балл, сигналы) для окна, заканчивающегося сейчас."""
        c = self.cfg
        win = float(c.get("window_seconds", 10))
        if self.started_at is None:
            self.started_at = now
        chat_n, emote_n = count_recent(chat_history, now - win, now, c.get("emotes", []))
        audio_db = mean_audio(audio, now - 3, now)
        if audio_db is not None and audio_db < SILENCE_DB:
            audio_db = None  # тишина (BRB, выключенный микрофон) не участвует в сравнении
        base = [s for s in self.samples if now - c.get("baseline_seconds", 600) <= s.t <= now - win]
        self.samples.append(Sample(now, chat_n, emote_n, audio_db))

        signals = {"chat": chat_n, "emotes": emote_n, "audio_db": round(audio_db, 1) if audio_db is not None else None}
        if now - self.started_at < c.get("warmup_seconds", 120) or len(base) < 15:
            signals["warmup"] = True
            return 0.0, signals

        base_chat = [s.chat for s in base]
        chat_z = zscore(chat_n, base_chat, min_std=max(1.0, statistics.fmean(base_chat) * 0.25))
        audio_vals = [s.audio for s in base if s.audio is not None]
        audio_z = zscore(audio_db, audio_vals, min_std=3.0) if audio_db is not None and len(audio_vals) >= 15 else 0.0
        if chat_n == 0:  # громкий звук без всякой реакции чата — не момент
            audio_z = min(audio_z, 1.0)
        mean_chat = statistics.fmean(base_chat)
        emote_score = min(3.0, emote_n / max(4.0, mean_chat)) if chat_n else 0.0
        # Клипы зрителей приходят из API с опозданием в минуты: в живой балл их не добавляем
        # (они сами создают кандидатов в App._clips_loop), здесь только для информации.
        clips = sum(1 for t in clip_moments if now - 300 <= t <= now)

        w = c.get("weights", {})
        score = (w.get("chat", 2) * min(max(chat_z, 0), 5)
                 + w.get("audio", 1.5) * min(max(audio_z, 0), 5)
                 + w.get("emotes", 1) * emote_score)
        signals.update(chat_z=round(chat_z, 2), audio_z=round(audio_z, 2), emote_score=round(emote_score, 2),
                       clips=clips)
        return round(score, 2), signals


@dataclass
class OpenWindow:
    """Кандидат, окно которого ещё может расшириться."""
    cid: int
    peak: float
    start: float
    end: float
    max_score: float


def plan_window(peak: float, cfg: dict) -> tuple[float, float]:
    return peak - cfg.get("pre_roll", 45), peak + cfg.get("post_roll", 30)


def extend_window(w: OpenWindow, now: float, cfg: dict) -> None:
    reaction = cfg.get("reaction_delay", 10)
    new_end = now - reaction + cfg.get("post_roll", 30) * 0.66
    w.end = min(max(w.end, new_end), w.start + cfg.get("max_window", 90))
