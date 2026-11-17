"""Понятные сообщения об ошибках: что случилось и что с этим делать — вместо трассировок и английских текстов."""
from __future__ import annotations

import errno

# (признаки в тексте ошибки, объяснение). Порядок важен: первые правила точнее.
RULES: list[tuple[tuple[str, ...], str]] = [
    # --- место и файлы ---
    (("no space left", "errno 28", "недостаточно места"),
     "закончилось место на диске — освободите место (data/ можно чистить) и перезапустите"),
    # --- routerai / нейросеть ---
    (("http 402", "insufficient", "balance", "баланс"),
     "на балансе routerai закончились деньги — пополните счёт в личном кабинете"),
    (("http 404", "model not found", "no such model", "model_not_found"),
     "в routerai нет такой модели — проверьте llm.screen_model / llm.review_model в config.yaml "
     "(python -m clipper doctor покажет, какие есть)"),
    (("http 429", "rate limit", "too many requests"),
     "routerai просит снизить частоту запросов — повторю позже сам"),
    (("http 500", "http 502", "http 503", "http 504", "bad gateway", "service unavailable"),
     "routerai временно сбоит (ошибка на их стороне) — повторю позже сам"),
    (("в ответе нет json", "не удалось разобрать json", "нет оценки в ответе"),
     "нейросеть ответила не по формату — повторю запрос"),
    (("сеть:", "connecterror", "connecttimeout", "readtimeout", "name or service not known", "getaddrinfo",
      "nodename nor servname"),
     "нет связи с сервером — проверьте интернет; повторю позже сам"),
    # --- YouTube / Twitch-записи (yt-dlp) ---
    (("private video", "this video is private"), "видео приватное — его может смотреть только автор"),
    (("confirm your age", "age-restricted", "inappropriate for some users"),
     "видео 18+ — YouTube не отдаёт его без входа в аккаунт"),
    (("not a bot", "sign in to confirm"),
     "YouTube требует подтвердить, что вы не бот — попробуйте позже, с другим интернетом или VPN"),
    (("not available in your country", "geo restrict", "geo-restricted"),
     "видео недоступно в вашей стране — нужен VPN"),
    (("subscriber", "only available to", "sub-only", "subscribers-only"),
     "запись доступна только подписчикам канала"),
    (("video unavailable", "has been removed", "does not exist", "http error 404", "this video is unavailable"),
     "видео удалено или недоступно по ссылке"),
    (("unsupported url",), "ссылка не поддерживается — нужна ссылка на видео YouTube или запись стрима Twitch"),
    (("requested format is not available", "no video formats", "only images are available"),
     "у видео не нашлось подходящего качества — установите deno (winget install DenoLand.Deno / brew install "
     "deno) и попробуйте снова"),
    # --- запись стримов (streamlink) ---
    (("no playable streams",), "стрим недоступен — канал офлайн или стрим только для подписчиков"),
    # --- Whisper ---
    (("cublas", "cudnn", "cudart", "libcuda", "cuda driver"),
     "Whisper не видит библиотеки видеокарты — выполните pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
     "и обновите драйвер NVIDIA (до этого распознавание идёт на процессоре)"),
    (("huggingface", "hf_hub", "model.bin", "localentrynotfound"),
     "не удалось скачать модель Whisper — при первом запуске нужен интернет (~1.6 ГБ)"),
    # --- ffmpeg ---
    (("no such filter: 'ass'", "no such filter: 'subtitles'"),
     "в ffmpeg нет поддержки субтитров — нужна полная сборка (Windows: winget install Gyan.FFmpeg, "
     "Mac: brew install ffmpeg-full)"),
    (("nvenc", "nvcuda", "no capable devices", "cannot load nvenc"),
     "кодировщик видеокарты NVIDIA недоступен — обновите драйвер или поставьте render.encoder: libx264 "
     "в config.local.yaml"),
    (("error opening input", "no such file or directory"),
     "файл не найден — возможно, его уже удалила уборка или запись прервалась"),
    (("invalid data found when processing input", "moov atom not found"),
     "файл видео повреждён или не дописан"),
]


def explain(e: BaseException | str) -> str:
    """Короткое объяснение на русском: что случилось и что делать."""
    if isinstance(e, BaseException):
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            return RULES[0][1]
        name = type(e).__name__
        if name == "LLMAuthError":
            return ("routerai не принял ключ (401) — проверьте ROUTERAI_API_KEY в .env: без кавычек и пробелов, "
                    "ключ не отозван в кабинете")
        if name == "TelegramUnauthorizedError":
            return "Telegram не принял токен бота — проверьте TELEGRAM_BOT_TOKEN в .env (токен от @BotFather)"
        if name == "TelegramConflictError":
            return ("этот бот уже запущен в другом месте (вторая копия программы на другом компьютере или в другом "
                    "окне) — остановите лишнюю копию")
        if name == "TelegramNetworkError":
            return "нет связи с Telegram (api.telegram.org) — проверьте интернет; если Telegram заблокирован — нужен VPN"
        if name == "TimeoutError" or name.endswith("Timeout"):
            return "операция не уложилась во время — повторю позже сам"
        text = f"{name}: {e}"
    else:
        text = str(e)
    low = text.lower()
    if "model" in low and ("not found" in low or "does not exist" in low or "invalid model" in low):
        return RULES[2][1]  # нет такой модели в routerai
    for keys, human in RULES:
        if any(k in low for k in keys):
            return human
    # уже понятное сообщение из нашего кода — оставляем как есть, без имени класса
    if isinstance(e, BaseException) and any("а" <= ch <= "я" for ch in str(e).lower()):
        return str(e)[:300]
    return text[:300]
