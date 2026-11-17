"""Запуск: python -m clipper run | add <ник> | remove <ник> | list | doctor"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from logging.handlers import RotatingFileHandler

from .config import load_config


COLORS = {logging.WARNING: "\033[33m", logging.ERROR: "\033[31m", logging.CRITICAL: "\033[31m"}
DIM, RESET = "\033[2m", "\033[0m"


def _enable_colors() -> bool:
    """Цвета в консоли: на Windows их нужно включить (режим VT), при выводе в файл — не нужны."""
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not k.GetConsoleMode(h, ctypes.byref(mode)) or not k.SetConsoleMode(h, mode.value | 0x0004):
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


class ConsoleFormatter(logging.Formatter):
    """Консоль: «12:03:41 сообщение». Предупреждения — жёлтым с ⚠️, ошибки — красным с ❗ и объяснением,
    без трассировок (они в data/logs/app.log)."""

    def __init__(self, colors: bool):
        super().__init__()
        self.colors = colors

    def format(self, record: logging.LogRecord) -> str:
        from .errors import explain
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None and record.levelno >= logging.ERROR:
            msg = f"{msg}: {explain(record.exc_info[1])} (подробности — data/logs/app.log)"
        if record.levelno >= logging.ERROR:
            msg = f"❗ {msg}"
        elif record.levelno >= logging.WARNING:
            msg = f"⚠️  {msg}"
        ts = self.formatTime(record, "%H:%M:%S")
        if not self.colors:
            return f"{ts} {msg}"
        color = COLORS.get(record.levelno, "")
        return f"{DIM}{ts}{RESET} {color}{msg}{RESET if color else ''}"


def setup_logging(cfg) -> None:
    (cfg.data_dir / "logs").mkdir(exist_ok=True)
    # Windows: без этого эмодзи в консоли при перенаправлении вывода могут ронять логирование
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ConsoleFormatter(_enable_colors()))
    file = RotatingFileHandler(cfg.data_dir / "logs" / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file.setLevel(logging.DEBUG)  # в файле — всё, включая трассировки и полные тексты ошибок
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root.addHandler(console)
    root.addHandler(file)
    for noisy in ("httpx", "httpx2", "httpcore", "aiogram", "aiohttp", "asyncio", "faster_whisper", "urllib3",
                  "streamlink", "PIL", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _banner(app, cfg) -> None:
    """Сводка при запуске: что включено, что нет и почему."""
    from .version import full_version
    s, st = cfg.secrets, app.status()
    llm = cfg.section("llm")
    chans = app.db.channels()
    on, off = "✅", "➖"
    rows = [
        f"🏷  gta-clipper {full_version()}",
        f"{on} ffmpeg · кодировщик видео: {app.encoder}",
        f"{on if s.telegram_token else off} Telegram-бот" + ("" if s.telegram_token else " выключен — нет TELEGRAM_BOT_TOKEN, клипы копятся в data/candidates"),
        f"{on if s.routerai_key else off} ИИ-отбор" + (f": {llm.get('screen_model')} → {llm.get('review_model')}"
                                                   if s.routerai_key else " выключен — нет ROUTERAI_API_KEY, на модерацию идёт весь фрагмент"),
        f"{on if s.has_twitch else off} Twitch API" + ("" if s.has_twitch else " не настроен — без клипов зрителей и проверки ников"),
        f"📺 каналы ({len(chans)}): " + (", ".join(c["login"] + (" ⏸" if c["state"] == "paused" else "") for c in chans) or "нет — добавьте в боте"),
        f"💾 свободно на диске: {st['free_gb']:.0f} ГБ · данные: {cfg.data_dir}",
    ]
    if st["moderation"] or st["processing"]:
        rows.append(f"📦 с прошлого запуска: на модерации {st['moderation']}, в обработке {st['processing']}")
    width = max(len(r) for r in rows) + 2
    logging.info("─" * min(width, 100))
    for r in rows:
        logging.info(r)
    logging.info("─" * min(width, 100))


async def cmd_run(cfg, extra_channels: list[str]) -> None:
    from .app import App

    app = App(cfg)
    for login in extra_channels:
        ok, info = await app.add_channel(login)
        logging.info("📺 канал %s: %s", login, "добавлен" if ok else info)
    if cfg.secrets.telegram_token:
        from .bot import BotUI
        app.bot = BotUI(app, cfg.secrets.telegram_token, cfg.secrets.telegram_owner_id)
        if not cfg.secrets.telegram_owner_id:
            logging.warning("TELEGRAM_OWNER_ID не задан: напишите боту /start, он пришлёт ваш ID")
    await app.start()
    _banner(app, cfg)
    logging.info("▶️  программа запущена. Остановить — Ctrl+C")
    try:
        if app.bot:
            await app.bot.run()
        else:
            await asyncio.Event().wait()
    finally:
        logging.info("⏹  останавливаюсь — дописываю записи…")
        await app.shutdown()
        logging.info("⏹  программа остановлена")


async def cmd_doctor(cfg) -> int:

    from .tools import find_ffmpeg, find_ffprobe, pick_encoder, _has_libass

    ok = True
    from .version import full_version
    print(f"🏷  gta-clipper {full_version()}")

    def line(good: bool, text: str):
        nonlocal ok
        ok &= good
        print(("✅ " if good else "❌ ") + text)

    try:
        ff = find_ffmpeg(cfg.get("ffmpeg", "auto"))
        line(True, f"ffmpeg: {ff}")
        line(_has_libass(ff), "субтитры (libass) в ffmpeg")
        line(True, f"ffprobe: {find_ffprobe(cfg.get('ffprobe', 'auto'), ff)}")
        print(f"   кодировщик: {pick_encoder(ff, cfg.section('render').get('encoder', 'auto'))}")
    except RuntimeError as e:
        line(False, str(e))
    try:
        from .tools import streamlink_cmd
        line(True, "streamlink: " + " ".join(streamlink_cmd(cfg.get("streamlink", "auto"))))
    except RuntimeError as e:
        line(False, str(e))
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        from .transcribe import best_device
        dev = best_device()
        print("   whisper: " + {"cuda": f"видеокарта NVIDIA ({n})", "mps": "графическое ядро Apple (MPS)"}.get(
            dev, "процессор (видеокарта не найдена)"))
    except Exception as e:  # noqa: BLE001
        line(False, f"faster-whisper: {e}")
    s = cfg.secrets
    line(s.has_twitch, "ключи Twitch (без них нет клипов зрителей и проверки ников)")
    line(bool(s.telegram_token), "токен Telegram-бота")
    line(bool(s.telegram_owner_id), "TELEGRAM_OWNER_ID")
    line(bool(s.routerai_key), "ключ routerai")
    if s.routerai_key:
        from .llm import RouterAI
        llm = RouterAI(cfg.section("llm"), s.routerai_key)
        key = s.routerai_key
        print(f"   ключ routerai: {key[:6]}…{key[-4:]} ({len(key)} симв.)")
        try:
            models = set(await llm.list_models())
            for k in ("screen_model", "review_model"):
                name = cfg.section("llm")[k]
                line(name in models, f"модель {name} есть в каталоге routerai")
        except Exception as e:  # noqa: BLE001
            line(False, f"каталог моделей routerai не ответил: {e}")
        # настоящий пробный запрос: список моделей может открываться и без ключа
        for name in dict.fromkeys(cfg.section("llm")[k] for k in ("screen_model", "review_model")):
            try:
                r = await llm.http.post("/chat/completions", json={"model": name, "max_tokens": 5,
                                        "messages": [{"role": "user", "content": "Ответь словом ok"}]},
                                        headers={"Authorization": f"Bearer {key}"}, timeout=30)
                ok = r.status_code == 200
                line(ok, f"пробный запрос к {name}: " + ("работает" if ok else f"HTTP {r.status_code} {r.text[:200]}"))
            except Exception as e:  # noqa: BLE001
                line(False, f"пробный запрос к {name}: {e}")
    try:
        import yt_dlp
        line(True, f"yt-dlp {yt_dlp.version.__version__} (видео по ссылке)")
    except ImportError:
        line(False, "yt-dlp не установлен: pip install -e .")
    line(bool(shutil.which("deno")), "deno — нужен yt-dlp для YouTube (Windows: winget install DenoLand.Deno; "
                                     "Mac: brew install deno), без него часть качеств недоступна")
    free = shutil.disk_usage(cfg.data_dir).free / 1e9
    line(free > 50, f"свободно на диске: {free:.0f} ГБ")
    return 0 if ok else 1


STAGE = {"to_cut": "ждёт вырезки", "cut": "ждёт распознавания речи", "transcribed": "ждёт Haiku",
         "screened": "ждёт Sonnet", "to_render": "ждёт монтажа", "rendered": "на модерации",
         "approved": "ждёт публикации", "failed": "ошибка"}


def cmd_queue(cfg) -> None:
    import time

    from .db import DB
    for stream in (sys.stdout,):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    db = DB(cfg.data_dir / "app.db")
    rows = db.all("SELECT id, channel, status, attempts, error, next_try_at, created_at FROM candidates "
                  "WHERE status IN ('to_cut','cut','transcribed','screened','to_render','rendered','approved') "
                  "OR (status='failed' AND updated_at > ?) ORDER BY id", (time.time() - 86400,))
    if not rows:
        print("Очередь пуста.")
        return
    now = time.time()
    for r in rows:
        age = int((now - (r["created_at"] or now)) / 60)
        line = f"#{r['id']:<4} {r['channel'] or '':15} {STAGE.get(r['status'], r['status']):26} {age:>4} мин назад"
        if r["attempts"]:
            line += f" · попыток: {r['attempts']}"
        if r["next_try_at"] and r["next_try_at"] > now:
            line += f" · повтор через {int(r['next_try_at'] - now)} с"
        print(line)
        if r["error"]:
            print(f"      ошибка: {r['error'][:300]}")


def main() -> None:
    p = argparse.ArgumentParser(prog="clipper", description="Нарезка вертикальных клипов из стримов Twitch")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="запустить программу")
    r.add_argument("--channel", action="append", default=[], help="добавить канал при запуске")
    a = sub.add_parser("add", help="добавить канал")
    a.add_argument("login")
    d = sub.add_parser("remove", help="удалить канал")
    d.add_argument("login")
    sub.add_parser("list", help="список каналов")
    sub.add_parser("doctor", help="проверить установку и ключи")
    sub.add_parser("queue", help="что сейчас в обработке и почему")
    args = p.parse_args()

    try:
        cfg = load_config()
    except Exception as e:  # noqa: BLE001
        print(f"❗ Ошибка в настройках — {e}")
        print("   Исправьте файл и запустите снова. В YAML важны отступы (пробелами) и двоеточие после названия.")
        sys.exit(2)
    if args.cmd == "doctor":
        sys.exit(asyncio.run(cmd_doctor(cfg)))
    if args.cmd == "queue":
        return cmd_queue(cfg)
    setup_logging(cfg)
    if args.cmd == "run":
        from .errors import explain
        try:
            asyncio.run(cmd_run(cfg, args.channel))
        except KeyboardInterrupt:
            pass
        except Exception as e:  # noqa: BLE001 — понятная причина вместо трассировки
            logging.debug("программа остановилась", exc_info=True)
            logging.error("Программа остановилась: %s. Подробности — %s", explain(e), cfg.data_dir / "logs" / "app.log")
            sys.exit(1)
        return
    from .db import DB
    db = DB(cfg.data_dir / "app.db")
    if args.cmd == "list":
        for c in db.channels(("active", "paused", "removing")):
            print(f"{c['login']:25} {c['state']:9} вебка: {c['cam_status']}")
    elif args.cmd in ("add", "remove"):
        from .app import App

        async def go():
            app = App(cfg)
            if args.cmd == "add":
                ok, info = await app.add_channel(args.login)
                print(("добавлен: " if ok else "ошибка: ") + info)
            else:
                await app.remove_channel(args.login.lower(), now=True)
                print("удалён")
        asyncio.run(go())


if __name__ == "__main__":
    main()
