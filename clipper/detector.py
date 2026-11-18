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


def recent_chunks(audio, start: float, end: float) -> list[tuple[float, float]]:
    out = []
    for t, db in reversed(audio):
        if t < start:
            break
        if t <= end:
            out.append((t, db))
    return out


def loudest(audio, start: float, end: float) -> float | None:
    """Время самого громкого фрагмента в [start, end]."""
    best_t, best_db = None, -1e9
    for t, db in reversed(audio):
        if t < start:
            break
        if t <= end and db > best_db:
            best_t, best_db = t, db
    return best_t


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
        # громкость — по последним 3 с полученного звука: он приходит пачками, и «последние 3 с до сейчас»
        # между пачками бывают пустыми (так YouTube-эфиры выглядели «тишиной»)
        a_end = audio[-1][0] if audio and 0 <= now - audio[-1][0] < 10 else now
        audio_db = mean_audio(audio, a_end - 3, a_end)
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
        # Рейдж: стример орёт заметно громче обычного (по умолчанию +9 дБ ≈ в 2 раза громче на слух).
        # Такой момент берём даже при молчащем чате.
        # Среднее за 3 с размывает короткий крик, поэтому рейдж смотрим по кускам по 0.5 с:
        # хотя бы секунда (2 куска) должна быть громче обычного уровня на rage_db.
        loud_by, rage = 0.0, False
        if len(audio_vals) >= 15:
            usual = statistics.median(audio_vals)
            spread = max(statistics.pstdev(audio_vals), 3.0)
            recent = sorted((db for t, db in recent_chunks(audio, now - 3, now) if db > SILENCE_DB), reverse=True)
            if len(recent) >= 2:
                loud_by = recent[1] - usual  # второй по громкости кусок: крик длится хотя бы 1 с
                rage = loud_by >= c.get("rage_db", 9.0) and loud_by / spread >= c.get("rage_min_z", 2.5)
                if rage:
                    audio_z = max(audio_z, loud_by / spread)
        if chat_n == 0 and not rage:  # просто громкий звук без всякой реакции чата — не момент
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
                       clips=clips, loud_by_db=round(loud_by, 1))
        # Момент «по звуку»: крик сильнее реакции чата — тогда окно привязываем к секунде крика,
        # а не к «сейчас минус задержка чата».
        audio_part = w.get("audio", 1.5) * min(max(audio_z, 0), 5)
        chat_part = w.get("chat", 2) * min(max(chat_z, 0), 5)
        if rage or (audio_part > 0 and audio_part >= chat_part):
            peak_t = loudest(audio, now - 4, now)
            if peak_t is not None:
                signals["audio_peak_t"] = round(peak_t, 2)
        if rage:
            signals["rage"] = True
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


def extend_window(w: OpenWindow, now: float, cfg: dict, peak: float | None = None) -> None:
    """Активность продолжается — растягиваем окно, чтобы в клип попала развязка (не дальше max_window)."""
    if peak is None:
        peak = now - cfg.get("reaction_delay", 10)
    new_end = peak + cfg.get("post_roll", 30) * 0.66
    w.end = min(max(w.end, new_end), w.start + cfg.get("max_window", 90))
