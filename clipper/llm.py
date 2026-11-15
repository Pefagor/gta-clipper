"""Отбор моментов: Haiku отсеивает, Sonnet доводит. Запросы идут в routerai.ru (формат OpenAI API)."""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

CATEGORIES = ["смешное", "баг", "погоня", "реакция", "эпичное", "другое"]

SCREEN_SYSTEM = """Ты отбираешь моменты со стримов по GTA 6 для вертикальных роликов (TikTok, YouTube Shorts).
Тебе дают фрагмент стрима: расшифровку речи стримера с таймкодами, сообщения чата и несколько кадров.
Детектор заметил тут всплеск активности. Реши, есть ли здесь момент, который зайдёт как короткий ролик
для русскоязычной аудитории: смешное, неожиданное, эпичное, баг, яркая реакция стримера.
Скучная болтовня, загрузочные экраны, меню, реклама, просто спам эмоутами без события — низкая оценка.
Ответь ТОЛЬКО JSON без пояснений:
{"score": <целое 1-10>, "category": "<одно из: %s>", "summary": "<одна фраза: что происходит>"}""" % ", ".join(CATEGORIES)

REVIEW_SYSTEM = """Ты монтажёр коротких вертикальных роликов со стримов по GTA 6 для русскоязычной аудитории.
Тебе дают фрагмент стрима (таймкоды в секундах от начала фрагмента): расшифровку речи, чат, кадры и
вывод первичного отбора. Задачи:
1. Реши, стоит ли делать ролик (keep).
2. Выбери точные границы start/end: ролик должен начинаться с завязки (не обрывай фразу) и заканчиваться
   развязкой или реакцией. Обычно 20-45 секунд; до 90 секунд, только если история без этого не понятна.
3. Отметь spoiler=true, если в ролике видно или обсуждается развитие сюжета, концовка миссии, сюжетный поворот.
4. Придумай 3 коротких цепляющих заголовка на русском (до 60 символов, без кликбейта-лжи, без хэштегов).
5. tag — короткая тема для подборок (2-4 слова, например "погоня с полицией", "первый взгляд на город").
6. description — одна фраза для описания ролика.
Ответь ТОЛЬКО JSON без пояснений:
{"keep": true|false, "score": <1-10>, "start": <число>, "end": <число>, "category": "<одно из: %s>",
 "spoiler": true|false, "titles": ["...", "...", "..."], "tag": "...", "description": "..."}""" % ", ".join(CATEGORIES)


class LLMError(Exception):
    pass


def as_bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "yes", "да", "1"):
            return True
        if low in ("false", "no", "нет", "0"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def as_score(v) -> int:
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    if not m:
        raise LLMError(f"нет оценки в ответе: {v!r}")
    return max(0, min(10, round(float(m.group()))))


def as_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"в ответе нет JSON: {text[:200]}")
    try:
        return json.loads(text[start:end + 1])
    except ValueError as e:
        raise LLMError(f"не удалось разобрать JSON: {e}: {text[:200]}") from e


def _image_part(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def build_user_content(context: str, frames: list[tuple[float, Path]]) -> list[dict]:
    parts: list[dict] = [{"type": "text", "text": context}]
    for t, p in frames:
        parts.append({"type": "text", "text": f"Кадр на {t:.1f} с:"})
        parts.append(_image_part(p))
    return parts


class RouterAI:
    def __init__(self, cfg: dict, api_key: str, on_usage=None):
        self.cfg = cfg
        self.api_key = api_key
        self.on_usage = on_usage  # callback(model, in_tok, out_tok, cost_rub)
        self.http = httpx.AsyncClient(base_url=cfg.get("base_url", "https://routerai.ru/api/v1"),
                                      timeout=float(cfg.get("timeout", 120)))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def cost(self, model: str, in_tok: int, out_tok: int) -> float:
        p_in, p_out = (self.cfg.get("prices_rub", {}) or {}).get(model, [0, 0])
        return (in_tok * p_in + out_tok * p_out) / 1_000_000

    async def chat(self, model: str, system: str, content: list[dict], max_tokens: int = 2000) -> dict:
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        try:
            r = await self.http.post("/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as e:
            raise LLMError(f"сеть: {e}") from e
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage") or {}
        in_tok, out_tok = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        if self.on_usage:
            self.on_usage(model, in_tok, out_tok, self.cost(model, in_tok, out_tok))
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"неожиданный ответ: {str(data)[:300]}") from e
        if isinstance(text, list):  # некоторые прокси отдают список блоков
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        return parse_json(text)

    async def screen(self, context: str, frames) -> dict:
        res = await self.chat(self.cfg["screen_model"], SCREEN_SYSTEM, build_user_content(context, frames), 800)
        res["score"] = as_score(res.get("score"))
        if res.get("category") not in CATEGORIES:
            res["category"] = "другое"
        return res

    async def review(self, context: str, frames, duration: float) -> dict:
        res = await self.chat(self.cfg["review_model"], REVIEW_SYSTEM, build_user_content(context, frames), 2500)
        start = max(0.0, as_float(res.get("start"), 0.0))
        end = min(duration, as_float(res.get("end"), duration))
        if end - start < 8:  # модель выдала странные границы — берём всё окно
            start, end = 0.0, duration
        if end - start > 90:
            end = start + 90
        titles = [str(t).strip() for t in (res.get("titles") or []) if str(t).strip()][:3] or ["Момент со стрима"]
        return {"keep": as_bool(res.get("keep"), True), "score": as_score(res.get("score", 0)),
                "start": round(start, 2), "end": round(end, 2),
                "category": res.get("category") if res.get("category") in CATEGORIES else "другое",
                "spoiler": as_bool(res.get("spoiler"), False), "titles": titles,
                "tag": str(res.get("tag") or "").strip()[:40], "description": str(res.get("description") or "")[:200]}

    async def list_models(self) -> list[str]:
        r = await self.http.get("/models", headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]


def build_context(channel: str, transcript: str, chat: list[tuple[float, str]], signals: dict,
                  peak_rel: float, duration: float, screen: dict | None = None) -> str:
    if len(chat) > 60:
        step = len(chat) / 60
        chat = [chat[int(i * step)] for i in range(60)]
    chat_txt = "\n".join(f"[{t:.1f}] {m[:120]}" for t, m in chat) or "(сообщений нет)"
    s = (f"Стример: {channel}\nДлина фрагмента: {duration:.0f} с. Всплеск активности примерно на {peak_rel:.0f} с.\n"
         f"Сигналы детектора: {json.dumps(signals, ensure_ascii=False)}\n\n"
         f"Речь стримера:\n{transcript}\n\nЧат:\n{chat_txt}\n")
    if screen:
        s += f"\nПервичный отбор: {json.dumps(screen, ensure_ascii=False)}\n"
    return s
