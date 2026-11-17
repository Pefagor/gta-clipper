"""Понятные ошибки: что случилось и что делать."""
from clipper.errors import explain
from clipper.llm import LLMAuthError, LLMError


def test_llm_errors():
    assert "ROUTERAI_API_KEY" in explain(LLMAuthError("routerai отклонил ключ (HTTP 401)"))
    assert "баланс" in explain(LLMError('HTTP 402: {"error":"insufficient balance"}'))
    assert "screen_model" in explain(LLMError('''HTTP 400: {"error":{"message":"Model 'x' not found"}}'''))
    assert "повторю" in explain(LLMError("HTTP 503: Service Unavailable"))
    assert "интернет" in explain(LLMError("сеть: ConnectError"))


def test_video_errors():
    assert "приватное" in explain("ERROR: [youtube] abc: Private video. Sign in if you've been granted access")
    assert "18+" in explain("ERROR: [youtube] abc: Sign in to confirm your age")
    assert "не бот" in explain("ERROR: [youtube] abc: Sign in to confirm you're not a bot")
    assert "удалено" in explain("ERROR: [youtube] aaaaaaaaaaa: This video is unavailable")
    assert "подписчикам" in explain("ERROR: [twitch:vod] v123: This video is only available to subscribers")
    assert "не поддерживается" in explain("ERROR: Unsupported URL: https://example.com/")


def test_system_errors():
    assert "место на диске" in explain(OSError(28, "No space left on device"))
    assert "nvidia-cublas" in explain(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    assert "libx264" in explain(RuntimeError("рендер не удался: [h264_nvenc] No capable devices found"))


def test_our_russian_messages_kept_as_is():
    from clipper.cutter import CutError
    assert explain(CutError("слишком короткий кусок записи (5 с)")) == "слишком короткий кусок записи (5 с)"


def test_telegram_errors_by_class_name():
    class TelegramConflictError(Exception):
        pass

    class TelegramUnauthorizedError(Exception):
        pass
    assert "уже запущен" in explain(TelegramConflictError("Conflict: terminated by other getUpdates request"))
    assert "TELEGRAM_BOT_TOKEN" in explain(TelegramUnauthorizedError("Unauthorized"))
