"""Telegram-бот: модерация, публикация, каналы, статус, настройки — всё на кнопках."""
from __future__ import annotations

import asyncio
import html
import logging
import time
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import BaseFilter, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto,
                           KeyboardButton, Message, ReplyKeyboardMarkup)

from .db import day_start

log = logging.getLogger(__name__)

BTN_MOD, BTN_PUB, BTN_CH, BTN_ST, BTN_SET = "🎬 Модерация", "📤 Публикация", "📺 Каналы", "📊 Статус", "⚙️ Настройки"
CAT_ICON = {"смешное": "😂", "рейдж": "🤬", "баг": "🐞", "погоня": "🚓", "реакция": "😱", "эпичное": "🔥", "подборка": "🎞",
            "другое": "🎬"}
SENS = {"low": "Меньше кандидатов", "mid": "Средне", "high": "Больше"}


def kb(rows) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=d) for t, d in row]
                                                 for row in rows])


MAIN_KB = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=BTN_MOD), KeyboardButton(text=BTN_PUB)],
                                        [KeyboardButton(text=BTN_CH), KeyboardButton(text=BTN_ST)],
                                        [KeyboardButton(text=BTN_SET)]],
                              resize_keyboard=True, is_persistent=True)


class Input(StatesGroup):
    add_channel = State()
    title = State()
    hashtags = State()
    limit = State()


class Owner(BaseFilter):
    def __init__(self, owner_id: int | None):
        self.owner_id = owner_id

    async def __call__(self, event) -> bool:
        user = getattr(event, "from_user", None)
        return bool(self.owner_id and user and user.id == self.owner_id)


def fmt_dur(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"


def ago(ts: float | None) -> str:
    if not ts:
        return "—"
    m = int((time.time() - ts) / 60)
    return f"{m} мин назад" if m < 60 else f"{m // 60} ч {m % 60} мин назад"


def game_meta(game: str | None, games: dict) -> tuple[str, str]:
    """(короткое название для заголовка Shorts, хэштеги игры). Из config.yaml → games, иначе по названию."""
    if not game:
        return "", ""
    meta = games.get(game) or {}
    short = meta["short"] if "short" in meta else game
    tags = meta.get("tags")
    if tags is None:  # #dota2 из «Dota 2»; для категорий вроде Just Chatting хэштег не нужен
        slug = "".join(ch for ch in game.lower() if ch.isalnum())
        tags = f"#{slug}" if slug and game not in ("Just Chatting", "Music", "IRL") else ""
    return short, tags


def adjust_rect(rect: list[float], op: str, move: float = 0.01, grow: float = 0.05) -> list[float]:
    """Сдвиг/размер рамки вебки (доли кадра); рамка остаётся внутри кадра."""
    x, y, w, h = rect
    if op == "l":
        x -= move
    elif op == "r":
        x += move
    elif op == "u":
        y -= move
    elif op == "d":
        y += move
    elif op in ("w+", "w-"):
        nw = w * (1 + grow if op == "w+" else 1 - grow)
        x, w = x - (nw - w) / 2, nw
    elif op in ("h+", "h-"):
        nh = h * (1 + grow if op == "h+" else 1 - grow)
        y, h = y - (nh - h) / 2, nh
    w, h = min(max(w, 0.03), 1.0), min(max(h, 0.03), 1.0)
    x, y = min(max(x, 0.0), 1.0 - w), min(max(y, 0.0), 1.0 - h)
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def rating(c: dict) -> str:
    """Оценка ИИ из 10, если она есть; иначе — балл всплеска детектора (у него нет верхней границы)."""
    for key in ("review", "screen"):
        res = c.get(key)
        if isinstance(res, dict) and isinstance(res.get("score"), (int, float)):
            return f"⭐ {int(res['score'])}/10"
    if c["kind"] == "compilation":
        return f"⭐ {int(c.get('score') or 0)}/10"
    return f"📈 всплеск {c.get('score') or 0:.0f}"


class BotUI:
    def __init__(self, app, token: str, owner_id: int | None):
        self.app = app
        self.db = app.db
        self.owner_id = owner_id
        # большой таймаут: клип до 45 МБ грузится долго, а повтор по таймауту отправил бы видео дважды
        self.bot = Bot(token, session=AiohttpSession(timeout=600),
                       default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self.dp = Dispatcher()
        self.router = Router()
        self._card_lock = asyncio.Lock()
        self._pub_lock = asyncio.Lock()
        self._bg: set[asyncio.Task] = set()
        self._register()

    def spawn(self, coro) -> None:
        """Фоновая задача с сохранённой ссылкой и логом ошибок."""
        task = asyncio.create_task(coro)
        self._bg.add(task)

        def done(t: asyncio.Task):
            self._bg.discard(t)
            if not t.cancelled() and t.exception():
                log.error("фоновая задача упала", exc_info=t.exception())
        task.add_done_callback(done)

    # ================= запуск =================
    async def run(self) -> None:
        log.info("🤖 Telegram-бот запущен%s", "" if self.owner_id else " (TELEGRAM_OWNER_ID не задан — напишите боту /start)")
        tasks = [asyncio.create_task(self._card_loop()), asyncio.create_task(self._publish_loop())]
        try:
            await self.dp.start_polling(self.bot, handle_signals=False)
        finally:
            for t in tasks:
                t.cancel()

    @property
    def quiet(self) -> bool:
        return bool(self.app.setting("quiet"))

    async def notify(self, text: str, buttons=None) -> None:
        if not self.owner_id:
            return
        try:
            await self.bot.send_message(self.owner_id, text, reply_markup=kb(buttons) if buttons else None,
                                        disable_notification=self.quiet)
        except Exception as e:  # noqa: BLE001
            log.warning("telegram: %s", e)

    # ================= регистрация обработчиков =================
    def _register(self) -> None:
        r = self.router
        # /start для всех: помогает узнать свой ID, пока он не вписан в .env
        public = Router()

        @public.message(Command("start", "id"))
        async def start(m: Message, state: FSMContext):
            if self.owner_id and m.from_user.id == self.owner_id:
                await state.clear()
                await m.answer("Готов к работе. Меню внизу экрана.", reply_markup=MAIN_KB)
            elif not self.owner_id:
                await m.answer(f"Ваш Telegram ID: <code>{m.from_user.id}</code>\n"
                               "Впишите его в .env как TELEGRAM_OWNER_ID и перезапустите программу.")

        owner = Owner(self.owner_id)
        r.message.filter(owner)
        r.callback_query.filter(owner)

        r.message.register(self.on_mod, F.text == BTN_MOD)
        r.message.register(self.on_pub, F.text == BTN_PUB)
        r.message.register(self.on_channels, F.text == BTN_CH)
        r.message.register(self.on_status, F.text == BTN_ST)
        r.message.register(self.on_settings, F.text == BTN_SET)
        r.message.register(self.in_add_channel, Input.add_channel)
        r.message.register(self.in_title, Input.title)
        r.message.register(self.in_hashtags, Input.hashtags)
        r.message.register(self.in_limit, Input.limit)
        r.callback_query.register(self.cb, F.data)
        self.dp.include_router(public)
        self.dp.include_router(r)

    # ================= модерация =================
    def _card_caption(self, c: dict) -> str:
        cat = c.get("category") or "другое"
        queue = max(self.db.count("rendered") - 1, 0)
        lines = []
        if c.get("spoiler"):
            lines.append("🔞 <b>СПОЙЛЕР</b>")
        who = c["channel"] if c["kind"] == "clip" else "подборка: " + c["channel"]
        lines.append(f"📺 <b>{html.escape(who)}</b> · {CAT_ICON.get(cat, '🎬')} {html.escape(cat)} · {rating(c)}"
                     f" · 🕐 {ago(c.get('peak_ts') or c.get('created_at'))}")
        if c.get("game"):
            lines.append(f"🎮 {html.escape(c['game'])}")
        lines.append(f"«{html.escape(c.get('title') or '')}»")
        lines.append(f"⏱ {fmt_dur(c.get('render_duration'))} · В очереди ещё: {queue}")
        return "\n".join(lines)

    def _card_kb(self, cid: int) -> InlineKeyboardMarkup:
        return kb([[("✅ Одобрить", f"m:ok:{cid}"), ("❌ Отклонить", f"m:no:{cid}")],
                   [("✂️ Обрезать", f"m:cut:{cid}"), ("📝 Заголовок", f"m:title:{cid}")]])

    async def _send_card(self, c: dict) -> bool:
        """Отправляет карточку. False — клип отправить нельзя (он помечен failed), нужно брать следующий."""
        path = c.get("render_path")
        if not path or not Path(path).exists():
            self.db.update_candidate(c["id"], status="failed", error="файл клипа пропал")
            return False
        try:
            msg = await self.bot.send_video(self.owner_id, FSInputFile(path), caption=self._card_caption(c)[:1024],
                                            reply_markup=self._card_kb(c["id"]), supports_streaming=True,
                                            width=1080, height=1920, disable_notification=self.quiet)
        except TelegramForbiddenError:
            log.warning("Telegram не даёт писать: откройте бота и нажмите /start")
            await asyncio.sleep(60)
            return True
        except TelegramBadRequest as e:
            self.db.update_candidate(c["id"], status="failed", error=f"telegram: {e}"[:500])
            await self.notify(f"⚠️ Клип #{c['id']} не отправился в Telegram: {html.escape(str(e)[:200])}")
            return False
        self.db.update_candidate(c["id"], card_msg_id=msg.message_id)
        log.info("📨 #%d карточка отправлена в Telegram на модерацию", c["id"])
        return True

    async def _card_loop(self) -> None:
        while True:
            await asyncio.sleep(3)
            if not self.owner_id:
                continue
            try:
                async with self._card_lock:
                    await self._ensure_card()
            except Exception:  # noqa: BLE001
                log.exception("карточка модерации")

    async def _ensure_card(self, resend: bool = False) -> str:
        """Следит, чтобы на экране была ровно одна карточка. Возвращает, что сейчас происходит."""
        for _ in range(20):  # пропускаем клипы, которые отправить нельзя
            cur_id = self.db.kv_get("card_id")
            cur = self.db.candidate(cur_id) if cur_id else None
            if cur and cur["status"] == "to_render":
                return "rerender"  # перерендер после обрезки или смены заголовка — ждём
            if cur and cur["status"] == "rendered":
                if cur.get("card_msg_id") and not resend:
                    return "shown"
                if cur.get("card_msg_id"):  # старая копия карточки больше не активна
                    await self._safe(self.bot.edit_message_reply_markup(chat_id=self.owner_id,
                                                                        message_id=cur["card_msg_id"],
                                                                        reply_markup=None))
                if await self._send_card(cur):
                    return "shown"
                continue
            nxt = self.db.one("SELECT * FROM candidates WHERE status='rendered' ORDER BY id LIMIT 1")
            self.db.kv_set("card_id", nxt["id"] if nxt else None)
            if not nxt:
                return "empty"
            resend = False
        return "empty"

    async def on_mod(self, m: Message, state: FSMContext):
        await state.clear()
        async with self._card_lock:
            result = await self._ensure_card(resend=True)
        if result == "rerender":
            await m.answer("Клип перерендеривается, карточка придёт через несколько секунд.", reply_markup=MAIN_KB)
        elif result == "empty":
            busy = self.db.scalar("SELECT COUNT(*) FROM candidates WHERE status IN "
                                  "('to_cut','cut','transcribed','screened','to_render')") or 0
            await m.answer("Очередь модерации пуста." + (f" В обработке: {busy}." if busy else "")
                           + " Новые клипы придут сами.", reply_markup=MAIN_KB)

    async def _decide(self, q: CallbackQuery, cid: int, ok: bool) -> None:
        c = self.db.candidate(cid)
        if not c or c["status"] != "rendered":
            await q.answer("Этот клип уже обработан")
            return
        self.db.update_candidate(cid, status="approved" if ok else "rejected", decided_at=time.time())
        log.info("%s #%d %s", "👍" if ok else "👎", cid, "одобрен — в очередь публикации" if ok else "отклонён")
        if not ok and c["kind"] == "compilation":
            self.app.release_parts(c)
        text = self._card_caption(c) + ("\n\n✅ <b>Одобрено</b>" if ok else "\n\n❌ <b>Отклонено</b>")
        markup = None if ok else kb([[("↩️ Вернуть", f"m:undo:{cid}")]])
        await self._safe(q.message.edit_caption(caption=text, reply_markup=markup))
        await q.answer("Одобрено" if ok else "Отклонено")
        async with self._card_lock:
            if self.db.kv_get("card_id") == cid:
                self.db.kv_set("card_id", None)
            await self._ensure_card()

    # ================= обработчик кнопок =================
    async def cb(self, q: CallbackQuery, state: FSMContext):
        if not isinstance(q.message, Message):  # сообщение слишком старое, Telegram его не отдаёт
            return await self._safe(q.answer("Сообщение устарело, откройте раздел заново из меню"))
        parts = q.data.split(":")
        kind, action = parts[0], parts[1] if len(parts) > 1 else ""
        arg = parts[2] if len(parts) > 2 else ""
        try:
            handler = getattr(self, f"cb_{kind}", None)
            if handler:
                await handler(q, state, action, arg, parts)
            else:
                await q.answer()
        except TelegramBadRequest as e:
            log.debug("telegram: %s", e)
            await self._safe(q.answer())
        except Exception:  # noqa: BLE001 — ошибка в одной кнопке не должна ронять бота
            log.exception("кнопка %s", q.data)
            await self._safe(q.answer("Ошибка, подробности в логе"))

    async def cb_m(self, q, state, action, arg, parts):
        cid = int(arg)
        c = self.db.candidate(cid)
        if action in ("ok", "no"):
            return await self._decide(q, cid, action == "ok")
        if not c:
            return await q.answer("Клип не найден")
        if action == "undo":
            if c["status"] != "rejected" or not c.get("render_path") or not Path(c["render_path"]).exists():
                return await q.answer("Уже нельзя вернуть: клип удалён", show_alert=True)
            self.db.update_candidate(cid, status="rendered", card_msg_id=None, decided_at=None)
            await self._safe(q.message.edit_caption(caption=self._card_caption(c) + "\n\n↩️ Возвращён в очередь",
                                                    reply_markup=None))
            return await q.answer("Вернул в очередь")
        if action == "cut":
            if c["kind"] != "clip":
                return await q.answer("Подборку обрезать нельзя")
            self.db.update_candidate(cid, edit_start=c["clip_start"], edit_end=c["clip_end"])
            await self._safe(q.message.edit_reply_markup(reply_markup=self._cut_kb(self.db.candidate(cid))))
            return await q.answer()
        if action == "title":
            rows = [[(f"{i + 1}. {t[:40]}", f"t:{cid}:{i}")] for i, t in enumerate(c.get("titles") or [])]
            rows.append([("✏️ Свой вариант", f"t:{cid}:custom"), ("← Назад", f"t:{cid}:back")])
            await self._safe(q.message.edit_reply_markup(reply_markup=kb(rows)))
            return await q.answer()
        await q.answer()

    def _cut_kb(self, c: dict) -> InlineKeyboardMarkup:
        cid, s, e = c["id"], c["edit_start"], c["edit_end"]
        return kb([[("Начало −5 с", f"c:{cid}:s:-5"), ("Начало +5 с", f"c:{cid}:s:5")],
                   [("Конец −5 с", f"c:{cid}:e:-5"), ("Конец +5 с", f"c:{cid}:e:5")],
                   [(f"Сейчас: {fmt_dur(s)}–{fmt_dur(e)} ({int(e - s)} с)", f"c:{cid}:noop:0")],
                   [("🎬 Перерендерить", f"c:{cid}:go:0"), ("← Назад", f"c:{cid}:back:0")]])

    async def cb_c(self, q, state, action, arg, parts):
        cid, what, delta = int(action), arg, float(parts[3]) if len(parts) > 3 else 0
        c = self.db.candidate(cid)
        if not c or c["status"] != "rendered":
            return await q.answer("Клип уже обработан")
        if what in ("s", "e"):
            s, e = c["edit_start"], c["edit_end"]
            if what == "s":
                s = min(max(0.0, s + delta), e - 5)
            else:
                e = max(min(c["raw_duration"], e + delta), s + 5)
            self.db.update_candidate(cid, edit_start=s, edit_end=e)
            await self._safe(q.message.edit_reply_markup(reply_markup=self._cut_kb(self.db.candidate(cid))))
            return await q.answer()
        if what == "go":
            async with self._card_lock:
                self.db.update_candidate(cid, status="to_render", card_msg_id=None)
                self.db.kv_set("card_id", cid)
            await self._safe(q.message.delete())
            return await q.answer("Перерендериваю, ~15 с")
        if what == "back":
            self.db.update_candidate(cid, edit_start=None, edit_end=None)
            await self._safe(q.message.edit_reply_markup(reply_markup=self._card_kb(cid)))
        await q.answer()

    async def cb_t(self, q, state, action, arg, parts):
        cid = int(action)
        c = self.db.candidate(cid)
        if not c:
            return await q.answer()
        if arg == "back":
            await self._safe(q.message.edit_reply_markup(reply_markup=self._card_kb(cid)))
            return await q.answer()
        if arg == "custom":
            await state.set_state(Input.title)
            await state.update_data(cid=cid)
            await q.message.answer("Пришлите свой заголовок одним сообщением.")
            return await q.answer()
        titles = c.get("titles") or []
        idx = int(arg)
        if c["status"] != "rendered" or idx >= len(titles):
            return await q.answer("Клип уже обработан")
        await self._safe(q.message.delete())
        await self._set_title(cid, titles[idx])
        await q.answer("Заголовок выбран, перерендериваю")

    async def _set_title(self, cid: int, title: str) -> bool:
        """Только для клипа на модерации. Карточку перед вызовом удаляют; она придёт заново."""
        async with self._card_lock:
            c = self.db.candidate(cid)
            if not c or c["status"] != "rendered":
                return False
            self.db.update_candidate(cid, title=title, card_msg_id=None)
            self.db.kv_set("card_id", cid)
            if c["kind"] == "clip":  # заголовок вшит в видео — перерендер
                self.db.update_candidate(cid, status="to_render", edit_start=c["clip_start"], edit_end=c["clip_end"])
            return True

    async def in_title(self, m: Message, state: FSMContext):
        data = await state.get_data()
        await state.clear()
        title = (m.text or "").strip()[:80]
        if not title:
            return await m.answer("Пустой заголовок, оставил как было.")
        cid = data.get("cid")
        c = self.db.candidate(cid) if cid else None
        if not c or c["status"] != "rendered":
            return await m.answer("Этот клип уже обработан, заголовок не изменён.", reply_markup=MAIN_KB)
        if c.get("card_msg_id"):
            await self._safe(self.bot.delete_message(self.owner_id, c["card_msg_id"]))
        await self._set_title(cid, title)
        await m.answer(f"Заголовок: «{html.escape(title)}». Перерендериваю.", reply_markup=MAIN_KB)

    # ================= публикация =================
    def _pkg_caption(self, c: dict) -> str:
        short, game_tags = game_meta(c.get("game"), self.app.cfg.get("games") or {})
        tags = " ".join(t for t in (game_tags, self.app.setting("hashtags") or "") if t)[:200]
        title = (c.get("title") or "Момент со стрима")[:90]
        if c["kind"] == "clip":
            credit = f"Стрим: twitch.tv/{c['channel']}"
        else:
            credit = (c.get("description") or "")[:250]
        shorts_t = (f"{title} | {short}" if short else title)[:100]
        shorts_d = f"{credit} {tags} #shorts".strip()
        tiktok = f"{title} {credit} {tags}".strip()
        done = []
        if c.get("pub_shorts"):
            done.append("✅ Shorts")
        if c.get("pub_tiktok"):
            done.append("✅ TikTok")
        spoiler = "🔞 <b>СПОЙЛЕР</b>\n" if c.get("spoiler") else ""
        return (f"{spoiler}▶️ <b>YouTube Shorts</b>\nНазвание: <code>{html.escape(shorts_t)}</code>\n"
                f"Описание: <code>{html.escape(shorts_d)}</code>\n\n"
                f"🎵 <b>TikTok</b>\n<code>{html.escape(tiktok)}</code>"
                + (f"\n\n{' · '.join(done)}" if done else ""))

    def _pkg_kb(self, c: dict) -> InlineKeyboardMarkup:
        cid = c["id"]
        return kb([[("✅ Выложил в Shorts" if not c.get("pub_shorts") else "☑️ Shorts", f"p:s:{cid}"),
                    ("✅ Выложил в TikTok" if not c.get("pub_tiktok") else "☑️ TikTok", f"p:t:{cid}")],
                   [("⏰ Через 1 час", f"p:later:{cid}"), ("🗑 Не публиковать", f"p:drop:{cid}")]])

    async def _send_package(self, cid: int) -> bool:
        """Отправляет пакет, если клип всё ещё ждёт публикации. Под замком — без двойной отправки."""
        async with self._pub_lock:
            c = self.db.candidate(cid)
            if not c or c["status"] != "approved":
                return False
            path = c.get("render_path")
            if not path or not Path(path).exists():
                self.db.update_candidate(cid, status="failed", error="файл клипа пропал")
                return False
            try:
                msg = await self.bot.send_video(self.owner_id, FSInputFile(path), caption=self._pkg_caption(c)[:1024],
                                                reply_markup=self._pkg_kb(c), supports_streaming=True, width=1080,
                                                height=1920, disable_notification=self.quiet)
            except TelegramForbiddenError:
                log.warning("Telegram не даёт писать: откройте бота и нажмите /start")
                self.db.update_candidate(cid, defer_until=time.time() + 300)
                return False
            except TelegramBadRequest as e:
                self.db.update_candidate(cid, status="failed", error=f"telegram: {e}"[:500])
                await self.notify(f"⚠️ Пакет клипа #{cid} не отправился: {html.escape(str(e)[:200])}")
                return False
            self.db.update_candidate(cid, status="sent", sent_at=time.time(), pkg_msg_id=msg.message_id)
            log.info("📤 #%d пакет для публикации отправлен в Telegram", cid)
            return True

    def _next_publish_at(self) -> float:
        interval = float(self.app.setting("interval_hours") or 1.5) * 3600
        last = self.db.scalar("SELECT MAX(sent_at) FROM candidates WHERE sent_at IS NOT NULL") or 0
        return last + interval

    def _sent_today(self) -> int:
        return self.db.scalar("SELECT COUNT(*) FROM candidates WHERE sent_at>=?", (day_start(),)) or 0

    def _next_approved(self) -> dict | None:
        return self.db.one("SELECT * FROM candidates WHERE status='approved' AND defer_until<=? ORDER BY "
                           "CASE kind WHEN 'compilation' THEN 0 ELSE 1 END, decided_at LIMIT 1", (time.time(),))

    async def _publish_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            if not self.owner_id or self.app.setting("paused"):
                continue
            try:
                c = self._next_approved()
                if not c:
                    continue
                if self.app.setting("publish_mode") == "now":
                    await self._send_package(c["id"])
                elif self._sent_today() < int(self.app.setting("per_day") or 10) and time.time() >= self._next_publish_at():
                    await self._send_package(c["id"])
            except Exception:  # noqa: BLE001
                log.exception("публикация")

    async def cb_p(self, q, state, action, arg, parts):
        if action == "next":
            c = self._next_approved()
            if not c:
                return await q.answer("Нет одобренных клипов в очереди")
            await q.answer("Отправляю")
            await self._send_package(c["id"])
            return
        if action == "comp":
            tags = self.app.compilation_tags()
            if not tags:
                need = self.app.cfg.section("compilation").get("min_clips", 3)
                return await q.answer(f"Нужно хотя бы {need} одобренных клипа на одну тему", show_alert=True)
            rows = [[(f"{(g + ': ') if g else ''}{t} ({n})"[:60], f"p:mk:{i}")] for i, (g, t, n) in enumerate(tags[:10])]
            self.db.kv_set("comp_tags", [[g, t] for g, t, _ in tags[:10]])
            await self._safe(q.message.edit_text("🎞 Выберите тему подборки:", reply_markup=kb(rows)))
            return await q.answer()
        if action == "mk":
            tags = self.db.kv_get("comp_tags") or []
            idx = int(arg)
            if idx >= len(tags):
                return await q.answer()
            game, tag = tags[idx]
            cid = self.app.create_compilation(game, tag)
            if not cid:
                return await q.answer("Не хватает клипов", show_alert=True)
            await self._safe(q.message.edit_text(f"🎞 Собираю подборку «{html.escape(tag)}»"
                                                 f"{' (' + html.escape(game) + ')' if game else ''}. "
                                                 "Она придёт на модерацию."))
            return await q.answer()
        cid = int(arg)
        c = self.db.candidate(cid)
        if not c:
            return await q.answer()
        if action in ("s", "t"):
            field = "pub_shorts" if action == "s" else "pub_tiktok"
            self.db.update_candidate(cid, **{field: 0 if c.get(field) else 1})
            c = self.db.candidate(cid)
            if c["pub_shorts"] and c["pub_tiktok"]:
                self.db.update_candidate(cid, status="done")
                await self._safe(q.message.edit_caption(caption=self._pkg_caption(c) + "\n\n✅ <b>Выложено</b>",
                                                        reply_markup=None))
            else:
                if c["status"] == "done":
                    self.db.update_candidate(cid, status="sent")
                await self._safe(q.message.edit_caption(caption=self._pkg_caption(c), reply_markup=self._pkg_kb(c)))
            return await q.answer()
        if action == "later":
            self.db.update_candidate(cid, status="approved", defer_until=time.time() + 3600)
            await self._safe(q.message.delete())
            return await q.answer("Пришлю снова через час")
        if action == "drop":
            self.db.update_candidate(cid, status="done")
            await self._safe(q.message.edit_caption(caption=self._pkg_caption(c) + "\n\n🗑 Не публикуем",
                                                    reply_markup=None))
            return await q.answer()

    async def on_pub(self, m: Message, state: FSMContext):
        await state.clear()
        queued = self.db.count("approved")
        per_day = int(self.app.setting("per_day") or 10)
        mode = self.app.setting("publish_mode")
        nxt = self._next_publish_at()
        when = ("сразу после ✅" if mode == "now" else
                "сейчас" if nxt <= time.time() else time.strftime("%H:%M", time.localtime(nxt)))
        text = (f"📤 <b>Публикация</b>\nВ очереди: {queued}\nОтправлено сегодня: {self._sent_today()} из {per_day}\n"
                f"Следующий пакет: {when if queued else '—'}")
        await m.answer(text, reply_markup=kb([[("📨 Следующий сейчас", "p:next:0")],
                                              [("🎞 Собрать подборку", "p:comp:0")]]))

    # ================= каналы =================
    def _channels_view(self) -> tuple[str, InlineKeyboardMarkup]:
        rows = []
        today = day_start()
        chans = self.db.channels()
        for ch in chans:
            login = ch["login"]
            rec = self.app.recorders.get(login)
            n = self.db.scalar("SELECT COUNT(*) FROM candidates WHERE channel=? AND decided_at>=? AND status IN "
                               "('approved','sent','done')", (login, today)) or 0
            if ch["state"] == "paused":
                label = f"⏸ {ch['display_name'] or login} · на паузе"
            elif rec:
                mins = int((time.time() - rec.started_at) / 60)
                label = f"🔴 {ch['display_name'] or login} · {mins // 60}ч {mins % 60:02d}м · {n} клип."
            else:
                label = f"⚫ {ch['display_name'] or login} · офлайн"
            if ch["state"] == "removing":
                label += " · удалится"
            rows.append([(label, f"ch:open:{login}")])
        rows.append([("➕ Добавить канал", "ch:add:0")])
        return f"📺 <b>Каналы ({len(chans)})</b>", kb(rows)

    async def on_channels(self, m: Message, state: FSMContext):
        await state.clear()
        text, markup = self._channels_view()
        await m.answer(text, reply_markup=markup)

    def _channel_card(self, login: str) -> tuple[str, InlineKeyboardMarkup]:
        ch = self.db.channel(login)
        if not ch:
            return "Канал не найден.", kb([[("← Назад", "ch:list:0")]])
        rec = self.app.recorders.get(login)
        name = html.escape(ch["display_name"] or login)
        if rec:
            mins = int((time.time() - rec.started_at) / 60)
            head = f"🔴 <b>{name}</b> — в эфире {mins // 60}ч {mins % 60:02d}м" + ("" if rec.healthy else " (переподключение)")
        elif ch["state"] == "paused":
            head = f"⏸ <b>{name}</b> — на паузе"
        else:
            head = f"⚫ <b>{name}</b> — офлайн"
        cam = {"ok": "✅ настроена", "none": "🚫 без вебки", "pending": "⏳ ждёт подтверждения",
               "unknown": "настроится при первом эфире"}[ch["cam_status"]]
        today = day_start()
        clips = self.db.scalar("SELECT COUNT(*) FROM candidates WHERE channel=? AND decided_at>=? AND status IN "
                               "('approved','sent','done')", (login, today)) or 0
        cands = self.db.scalar("SELECT COUNT(*) FROM candidates WHERE channel=? AND created_at>=?", (login, today)) or 0
        text = f"{head}\nВебка: {cam} · Клипов сегодня: {clips} · Кандидатов: {cands}"
        pause = ("▶️ Продолжить", f"ch:resume:{login}") if ch["state"] == "paused" else ("⏸ Пауза", f"ch:pause:{login}")
        return text, kb([[pause, ("🎥 Настроить вебку", f"ch:cam:{login}")],
                         [("🗑 Удалить", f"ch:del:{login}"), ("← Назад", "ch:list:0")]])

    async def cb_ch(self, q, state, action, arg, parts):
        login = arg
        if action == "list":
            text, markup = self._channels_view()
            await self._safe(q.message.edit_text(text, reply_markup=markup))
        elif action == "add":
            await state.set_state(Input.add_channel)
            await q.message.answer("Пришлите ник канала или ссылку вида twitch.tv/ник.")
        elif action == "open":
            text, markup = self._channel_card(login)
            await self._safe(q.message.edit_text(text, reply_markup=markup))
        elif action in ("pause", "resume"):
            await self.app.set_paused(login, action == "pause")
            text, markup = self._channel_card(login)
            await self._safe(q.message.edit_text(text, reply_markup=markup))
        elif action == "cam":
            if login not in self.app.recorders:
                return await q.answer("Настроить вебку можно, когда канал в эфире", show_alert=True)
            self.db.update_channel(login, cam_status="pending")
            self.spawn(self.app.calibrate_cam(login))
            return await q.answer("Ищу вебку, пришлю кадр")
        elif action == "del":
            name = html.escape((self.db.channel(login) or {}).get("display_name") or login)
            if login in self.app.recorders:
                await self._safe(q.message.edit_text(
                    f"Удалить {name}? Сейчас идёт запись.",
                    reply_markup=kb([[("Остановить сейчас", f"ch:delnow:{login}")],
                                     [("Дописать до конца стрима", f"ch:delend:{login}")],
                                     [("Отмена", f"ch:open:{login}")]])))
            else:
                await self._safe(q.message.edit_text(
                    f"Удалить {name}? Клипы в очереди останутся.",
                    reply_markup=kb([[("🗑 Да, удалить", f"ch:delnow:{login}"), ("Отмена", f"ch:open:{login}")]])))
        elif action in ("delnow", "delend"):
            await self.app.remove_channel(login, now=action == "delnow")
            text, markup = self._channels_view()
            note = "Удалён." if action == "delnow" else "Удалится после конца стрима."
            await self._safe(q.message.edit_text(f"{note}\n\n{text}", reply_markup=markup))
        elif action == "restart":
            await self.app.restart_recording(login)
            return await q.answer("Перезапускаю запись")
        await self._safe(q.answer())

    async def in_add_channel(self, m: Message, state: FSMContext):
        await state.clear()
        ok, info = await self.app.add_channel(m.text or "")
        if not ok:
            return await m.answer(f"❌ {html.escape(info)}", reply_markup=MAIN_KB)
        await m.answer(f"✅ Добавлен: <b>{html.escape(info)}</b>. Проверяю эфир, запись начнётся автоматически.",
                       reply_markup=MAIN_KB)
        if len(self.db.channels(("active",))) > 6:
            await m.answer("⚠️ Каналов больше 6: проверьте, хватит ли интернета (~8 Мбит/с на канал) и диска.")

    # ================= вебка =================
    def _cam_kb(self, login: str, rect) -> InlineKeyboardMarkup:
        rows = [[("✅ Верно", f"cam:ok:{login}"), ("✏️ Подправить", f"cam:adj:{login}")]] if rect else []
        rows.append([("🔄 Искать снова", f"cam:retry:{login}"), ("🚫 Без вебки", f"cam:none:{login}")])
        return kb(rows)

    def _cam_adjust_kb(self, login: str) -> InlineKeyboardMarkup:
        return kb([[("⬅️", f"cm:l:{login}"), ("⬆️", f"cm:u:{login}"), ("⬇️", f"cm:d:{login}"), ("➡️", f"cm:r:{login}")],
                   [("↔️ шире", f"cm:w+:{login}"), ("↔️ уже", f"cm:w-:{login}"),
                    ("↕️ выше", f"cm:h+:{login}"), ("↕️ ниже", f"cm:h-:{login}")],
                   [("✅ Готово", f"cam:ok:{login}"), ("← Назад", f"cm:back:{login}")]])

    @staticmethod
    def _cam_caption(name: str, rect, method: str | None) -> str:
        if not rect:
            return f"🎥 <b>{name}</b>: вебку не нашёл."
        if method == "estimate":
            return (f"🎥 <b>{name}</b>: нашёл лицо, но края вебки не видны — рамка примерная (жёлтая). "
                    "Проверьте и при необходимости подправьте.")
        return f"🎥 <b>{name}</b>: вебка здесь? (зелёная рамка)"

    async def send_cam_proposal(self, login: str, photo: Path | None, rect, method: str | None = None) -> None:
        if not self.owner_id:
            return
        name = html.escape((self.db.channel(login) or {}).get("display_name") or login)
        text = self._cam_caption(name, rect, method)
        buttons = self._cam_kb(login, rect)
        try:
            if photo:
                await self.bot.send_photo(self.owner_id, FSInputFile(photo), caption=text, reply_markup=buttons)
            else:
                await self.bot.send_message(self.owner_id, text, reply_markup=buttons)
        except Exception as e:  # noqa: BLE001
            log.warning("telegram: %s", e)

    async def cb_cam(self, q, state, action, arg, parts):
        login = arg
        if action == "ok":
            self.app.accept_cam(login, self.db.kv_get(f"cam:{login}"))
            await self._edit(q.message, "✅ Вебка сохранена")
        elif action == "none":
            self.app.accept_cam(login, None)
            await self._edit(q.message, "🚫 Сохранено: без вебки")
        elif action == "retry":
            if login not in self.app.recorders:
                return await q.answer("Канал сейчас не в эфире", show_alert=True)
            await self._safe(q.message.delete())
            self.spawn(self.app.calibrate_cam(login))
            return await q.answer("Ищу снова")
        elif action == "adj":
            await self._safe(q.message.edit_reply_markup(reply_markup=self._cam_adjust_kb(login)))
        await self._safe(q.answer())

    async def cb_cm(self, q, state, action, arg, parts):
        """Ручная подстройка рамки вебки: сдвиг на 1% кадра, размер на 5%."""
        login = arg
        rect = self.db.kv_get(f"cam:{login}")
        if not rect:
            return await q.answer("Рамки нет — нажмите «Искать снова»", show_alert=True)
        if action == "back":
            await self._safe(q.message.edit_reply_markup(reply_markup=self._cam_kb(login, rect)))
            return await q.answer()
        rect = adjust_rect(rect, action)
        self.db.kv_set(f"cam:{login}", rect)
        self.db.kv_set(f"cam_method:{login}", "manual")
        photo = await self.app.draw_cam_preview(login, rect)
        name = html.escape((self.db.channel(login) or {}).get("display_name") or login)
        caption = f"🎥 <b>{name}</b>: подстройка рамки. Нажмите «Готово», когда рамка совпадёт с вебкой."
        if photo:
            await self._safe(q.message.edit_media(
                media=InputMediaPhoto(media=FSInputFile(photo), caption=caption),
                reply_markup=self._cam_adjust_kb(login)))
        await self._safe(q.answer())

    # ================= статус =================
    def _status_text(self) -> str:
        s = self.app.status()
        warn = []
        if not s["llm"]:
            warn.append("нет ключа routerai — ИИ-отбор выключен")
        if not s["twitch_api"]:
            warn.append("нет ключей Twitch — клипы зрителей не учитываются")
        if not s["chat"]:
            warn.append("чат не подключён")
        limit = float(self.app.setting("api_daily_limit_rub") or 0)
        lines = [f"📊 <b>Статус</b> · обновлено {time.strftime('%H:%M')}",
                 ("⏸ <b>Всё на паузе</b>" if s["paused"] else
                  f"Записи: 🔴 {s['recording']} из {s['channels']} каналов" +
                  (" · без ошибок" if s["healthy"] == s["recording"] else f" · переподключаются: {s['recording'] - s['healthy']}")),
                 f"Очереди: модерация {s['moderation']} · публикация {s['publish']} · обработка {s['processing']}",
                 f"Сегодня: {s['cand_today']} кандидатов → {s['approved_today']} одобрено → {s['posted_today']} выложено",
                 f"Диск: {s['free_gb']:.0f} ГБ свободно · API сегодня: {s['spend_today']:.1f} ₽" +
                 (f" из {limit:.0f} ₽" if limit else ""),
                 f"Кодировщик: {s['encoder']}"]
        if warn:
            lines.append("⚠️ " + "; ".join(warn))
        return "\n".join(lines)

    async def on_status(self, m: Message, state: FSMContext):
        await state.clear()
        await m.answer(self._status_text(), reply_markup=kb([[("🔄 Обновить", "st:show")]]))

    async def cb_st(self, q, state, action, arg, parts):
        if action == "retry":
            self.db.execute("UPDATE candidates SET next_try_at=0 WHERE status IN ('transcribed','screened')")
            return await q.answer("Повторяю")
        if action == "show":
            await self._safe(q.message.edit_text(self._status_text(), reply_markup=kb([[("🔄 Обновить", "st:show")]])))
        await self._safe(q.answer())

    # ================= настройки =================
    def _settings_view(self) -> tuple[str, InlineKeyboardMarkup]:
        a = self.app
        sens = a.setting("sensitivity")
        mode = a.setting("publish_mode")
        limit = float(a.setting("api_daily_limit_rub") or 0)
        mark = lambda on, t: ("● " if on else "") + t  # noqa: E731
        text = (f"⚙️ <b>Настройки</b>\nХэштеги: {html.escape(a.setting('hashtags') or '—')}\n"
                f"Лимит API в день: {f'{limit:.0f} ₽' if limit else 'не задан'}")
        rows = [[(mark(sens == k, v), f"s:sens:{k}") for k, v in SENS.items()],
                [("−", "s:pd:-1"), (f"Публикаций в день: {a.setting('per_day')}", "s:noop:0"), ("+", "s:pd:1")],
                [("−", "s:int:-0.5"), (f"Интервал: {a.setting('interval_hours')} ч", "s:noop:0"), ("+", "s:int:0.5")],
                [(mark(mode == "schedule", "По расписанию"), "s:mode:schedule"),
                 (mark(mode == "now", "Сразу после ✅"), "s:mode:now")],
                [("✏️ Хэштеги", "s:tags:0"), ("💰 Лимит API", "s:limit:0")],
                [(("🔔 Звук вкл" if a.setting("quiet") else "🌙 Тихий режим"), "s:quiet:0"),
                 (("▶️ Возобновить всё" if a.setting("paused") else "🛑 Пауза всего"), "s:pause:0")]]
        return text, kb(rows)

    async def on_settings(self, m: Message, state: FSMContext):
        await state.clear()
        text, markup = self._settings_view()
        await m.answer(text, reply_markup=markup)

    async def cb_s(self, q, state, action, arg, parts):
        a = self.app
        if action == "sens":
            a.set_setting("sensitivity", arg)
        elif action == "pd":
            a.set_setting("per_day", max(1, int(a.setting("per_day") or 10) + int(arg)))
        elif action == "int":
            a.set_setting("interval_hours", max(0.5, round(float(a.setting("interval_hours") or 1.5) + float(arg), 1)))
        elif action == "mode":
            a.set_setting("publish_mode", arg)
        elif action == "quiet":
            a.set_setting("quiet", not a.setting("quiet"))
        elif action == "pause":
            if not a.setting("paused"):
                await self._safe(q.message.edit_text("Остановить запись всех каналов и отбор?", reply_markup=kb(
                    [[("🛑 Да, пауза", "s:pauseok:0"), ("Отмена", "s:back:0")]])))
                return await q.answer()
            a.set_setting("paused", False)
            a.wake.set()
        elif action == "pauseok":
            a.set_setting("paused", True)
            a.wake.set()
        elif action == "tags":
            await state.set_state(Input.hashtags)
            await q.message.answer("Пришлите общие хэштеги через пробел, например: <code>#стрим #нарезка</code>.\n"
                                   "Хэштеги игры добавляются сами (настраиваются в config.yaml, раздел games).")
            return await q.answer()
        elif action == "limit":
            await state.set_state(Input.limit)
            await q.message.answer("Пришлите дневной лимит расходов на API в рублях (0 — без лимита).")
            return await q.answer()
        text, markup = self._settings_view()
        await self._safe(q.message.edit_text(text, reply_markup=markup))
        await self._safe(q.answer())

    async def in_hashtags(self, m: Message, state: FSMContext):
        await state.clear()
        tags = " ".join(t if t.startswith("#") else f"#{t}" for t in (m.text or "").split())[:200]
        self.app.set_setting("hashtags", tags)
        await m.answer(f"Хэштеги: {html.escape(tags) or '—'}", reply_markup=MAIN_KB)

    async def in_limit(self, m: Message, state: FSMContext):
        try:
            value = max(0.0, float((m.text or "").replace(",", ".").replace("₽", "").strip()))
        except ValueError:
            return await m.answer("Нужно число, например 500. Попробуйте ещё раз.")
        await state.clear()
        self.app.set_setting("api_daily_limit_rub", value)
        await m.answer(f"Лимит API: {f'{value:.0f} ₽ в день' if value else 'без лимита'}", reply_markup=MAIN_KB)

    # ================= утилиты =================
    async def _edit(self, msg: Message, text: str, markup=None) -> None:
        if msg.photo or msg.video:
            await self._safe(msg.edit_caption(caption=text, reply_markup=markup))
        else:
            await self._safe(msg.edit_text(text, reply_markup=markup))

    @staticmethod
    async def _safe(coro):
        try:
            return await coro
        except TelegramBadRequest as e:
            log.debug("telegram: %s", e)
        except Exception as e:  # noqa: BLE001
            log.warning("telegram: %s", e)
