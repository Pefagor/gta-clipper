"""Распознавание речи faster-whisper (видеокарта NVIDIA, если есть; иначе процессор)."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self.device = None
        self.compute_type = None

    def _pick(self) -> list[tuple[str, str]]:
        device = self.cfg.get("device", "auto")
        compute = self.cfg.get("compute_type", "auto")
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        if compute != "auto":
            return [(device, compute)]
        if device == "cuda":
            # GTX 16xx иногда плохо работают с float16 — тогда откатываемся на int8
            return [("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")]
        return [("cpu", "int8")]

    def transcribe(self, wav: Path) -> dict:
        """Синхронно (вызывать через asyncio.to_thread). Возвращает сегменты со словами.

        Если видеокарта не справилась (нет CUDA/cuDNN, не тот режим вычислений), за один вызов
        перебираются режимы вплоть до процессора. Ошибки, не связанные с устройством, пробрасываются."""
        from faster_whisper import WhisperModel
        with self._lock:
            options = self._pick()
            if ("cpu", "int8") not in options:
                options.append(("cpu", "int8"))
            if self._model is not None:
                options = options[options.index((self.device, self.compute_type)):]
            last: Exception | None = None
            for device, compute in options:
                try:
                    if self._model is None or (device, compute) != (self.device, self.compute_type):
                        self._model = None
                        self._model = WhisperModel(self.cfg.get("model", "large-v3-turbo"), device=device,
                                                   compute_type=compute)
                        self.device, self.compute_type = device, compute
                        log.info("whisper: %s на %s (%s)", self.cfg.get("model"), device, compute)
                    result = self._run(wav)
                    if device == "cuda" and compute == "float16" and _looks_broken(result):
                        raise RuntimeError("cuda float16 дал пустой результат")
                    return result
                except Exception as e:  # noqa: BLE001
                    last = e
                    if device == "cpu" or not _device_error(e):
                        raise
                    log.warning("whisper %s/%s не справился (%s), пробую следующий режим", device, compute, e)
                    self._model = None
            raise RuntimeError(f"не удалось запустить whisper: {last}")

    def _run(self, wav: Path) -> dict:
        segments, info = self._model.transcribe(
            str(wav), language=self.cfg.get("language", "ru"), word_timestamps=True,
            vad_filter=True, beam_size=int(self.cfg.get("beam_size", 5)))
        out = []
        for s in segments:
            words = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                     for w in (s.words or []) if w.word.strip()]
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "words": words})
        return {"language": info.language, "segments": out}


def _device_error(e: Exception) -> bool:
    text = str(e).lower()
    return any(k in text for k in ("cuda", "cublas", "cudnn", "float16", "compute type", "gpu", "device",
                                   "out of memory", "пустой результат"))


def _looks_broken(result: dict) -> bool:
    """На GTX 16xx float16 иногда даёт мусор: одни и те же символы или пустые слова при наличии сегментов."""
    segs = result.get("segments", [])
    text = "".join(s["text"] for s in segs)
    return bool(segs) and (not text.strip() or len(set(text.replace(" ", ""))) <= 2)


def transcript_text(tr: dict, offset: float = 0.0, limit_chars: int = 6000) -> str:
    lines = [f"[{s['start'] - offset:.1f}] {s['text']}" for s in tr.get("segments", [])]
    text = "\n".join(lines)
    return text[:limit_chars] or "(речи нет)"


def words_in(tr: dict, start: float, end: float) -> list[dict]:
    return [w for s in tr.get("segments", []) for w in s.get("words", []) if w["end"] > start and w["start"] < end]
