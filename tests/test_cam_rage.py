"""Поиск вебки и рейдж-моменты."""
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from clipper.bot import adjust_rect
from clipper.detector import ChannelDetector
from clipper.transcribe import loud_events, transcript_text

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "tests" / "local_fixtures" / "stariy_bog"


def synthetic_frames(n=20, w=960, h=540, cam=(700, 0, 960, 220), seed=1):
    """«Игра» — меняющиеся размытые пятна; «вебка» — статичный фон с полкой и двигающимся человеком."""
    import cv2
    rng = np.random.default_rng(seed)
    cam_bg = cv2.resize(rng.integers(60, 120, (6, 8), dtype=np.uint8), (cam[2] - cam[0], cam[3] - cam[1]),
                        interpolation=cv2.INTER_CUBIC)
    cv2.line(cam_bg, (60, 10), (60, 100), 230, 3)  # полка за спиной: постоянная, но незамкнутая линия
    frames = []
    for _ in range(n):
        game = cv2.resize(rng.integers(0, 255, (9, 16), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        img = game.copy()
        img[cam[1]:cam[3], cam[0]:cam[2]] = cam_bg
        px = int(rng.integers(110, 150))
        cv2.circle(img, (cam[0] + px, 120), 35, int(rng.integers(150, 220)), -1)  # «голова» двигается
        frames.append(img)
    return frames


def test_find_rect_synthetic():
    from clipper.camcal import edge_persistence, find_rect
    frames = synthetic_frames()
    E = edge_persistence(frames)
    rect, score = find_rect(E, (700 + 95, 85, 70, 70))
    assert rect is not None and score >= 0.5
    L, T, R, B = rect
    assert abs(L - 700) <= 3 and T <= 2 and R >= 957 and abs(B - 220) <= 3


@pytest.mark.parametrize("folder", ["stariy_bog", "stariy_bog_menu"])  # игра и меню выбора героя (статичный UI)
def test_find_cam_real_stream(tmp_path, folder):
    frames = LOCAL.parent / folder
    if not frames.exists():
        pytest.skip("кадры реального стрима есть только локально (не в git)")
    out = subprocess.run([sys.executable, "-m", "clipper.camcal", "find", str(tmp_path / "preview.jpg"),
                          *map(str, sorted(frames.glob("*.jpg")))], capture_output=True, text=True, cwd=ROOT)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["method"] == "edges"
    x, y, w, h = res["rect"]
    # вебка stariy_bog — справа сверху; камера игрока внизу слева не должна выбираться
    assert abs(x - 0.793) < 0.01 and y < 0.01 and abs(x + w - 1.0) < 0.01 and abs(y + h - 0.264) < 0.01


def test_adjust_single_edges():
    r = [0.7, 0.0, 0.3, 0.3]
    x, y, w, h = adjust_rect(r, "Lr")            # левый край вправо — правый край на месте
    assert x > 0.7 and abs((x + w) - 1.0) < 1e-6
    x, y, w, h = adjust_rect(r, "Bu")            # нижний край вверх — верх на месте
    assert y == 0.0 and h < 0.3
    assert adjust_rect(r, "Rr") == [0.7, 0.0, 0.3, 0.3]   # правый край уже у края кадра


def test_adjust_rect_stays_inside_frame():
    assert adjust_rect([0.0, 0.0, 0.2, 0.2], "l") == [0.0, 0.0, 0.2, 0.2]
    assert adjust_rect([0.5, 0.5, 0.2, 0.2], "r") == [0.51, 0.5, 0.2, 0.2]
    x, y, w, h = adjust_rect([0.9, 0.0, 0.1, 0.2], "w+")
    assert x + w <= 1.0 and w > 0.1


def _talk(det, hist, audio, t0, seconds, db):
    score, sig = 0.0, {}
    for step in range(int(seconds / 2)):
        now = t0 + step * 2
        hist.append((now, "ну давай"))
        for k in range(4):
            audio.append((now - 1.5 + k * 0.5, db))
        score, sig = det.tick(now, hist, audio, [])
    return score, sig


def test_rage_triggers_without_chat():
    cfg = {"window_seconds": 10, "baseline_seconds": 600, "warmup_seconds": 0, "emotes": [],
           "weights": {"chat": 2, "audio": 1.5, "emotes": 1}, "rage_db": 9.0, "rage_min_z": 2.5}
    det = ChannelDetector(cfg)
    hist, audio = deque(), deque()
    _talk(det, hist, audio, 1000.0, 300, -30.0)  # 5 минут обычной речи
    now = 1300.0
    for k, db in enumerate([-29, -14, -12, -28]):  # крик длиной в секунду, чат молчит последние 10 с
        audio.append((now - 1.5 + k * 0.5, db))
    score, sig = det.tick(now, hist, audio, [])
    assert sig.get("rage") is True
    assert sig["audio_peak_t"] == pytest.approx(now - 0.5)  # окно привязывается к секунде крика


def test_loud_events_and_marks():
    rate = 16000
    t = np.arange(rate * 20) / rate
    samples = (0.03 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    samples[rate * 12: rate * 14] *= 10  # 2 секунды крика (+20 дБ)
    ev = loud_events(samples)
    assert len(ev) == 1 and ev[0]["start"] == 12.0 and ev[0]["end"] == 14.0 and ev[0]["over_db"] >= 15
    text = transcript_text({"segments": [{"start": 11.0, "text": "да ну нет"}], "loud": ev})
    assert text.splitlines()[1].startswith("[12.0] (КРИК")


@pytest.mark.skipif(not LOCAL.exists(), reason="кадры реального стрима есть только локально (не в git)")
def test_template_match_and_check_real_stream(tmp_path):
    """Вырезка вебки «со скриншота телефона» (уменьшена и пережата) находится в других кадрах стрима;
    проверка перед монтажом отличает настоящую вебку от пустого места в игре."""
    import cv2
    from clipper.camcal import check_cam, match_template
    frames = sorted(map(str, LOCAL.glob("*.jpg")))
    img = cv2.imread(frames[5])
    crop = cv2.resize(img[0:285, 1522:1920], None, fx=0.55, fy=0.55)
    tmpl = tmp_path / "tmpl.jpg"
    cv2.imwrite(str(tmpl), crop, [cv2.IMWRITE_JPEG_QUALITY, 60])
    res = match_template(str(tmpl), frames[12:15])
    x, y, w, h = res["rect"]
    assert abs(x - 0.7927) < 0.005 and y < 0.005 and abs(w - 0.2073) < 0.01 and abs(h - 0.2639) < 0.01
    assert check_cam([0.7927, 0.0, 0.2068, 0.2639], frames[:8])["present"] is True
    assert check_cam([0.35, 0.35, 0.2, 0.26], frames[:8])["present"] is False


def test_bursty_audio_is_not_silence():
    """Звук пачками по 5 с (YouTube): между пачками детектор не должен считать эфир тишиной."""
    cfg = {"window_seconds": 10, "baseline_seconds": 600, "warmup_seconds": 0, "emotes": [],
           "weights": {"chat": 2, "audio": 1.5, "emotes": 1}}
    det = ChannelDetector(cfg)
    hist, audio = deque(), deque()
    for k in range(300):  # пачки: 10 кусков по 0.5 с, приходят раз в 5 с
        if k % 10 == 0:
            for j in range(10):
                audio.append((1000 + k * 0.5 - 5 + j * 0.5, -20.0))
    _, sig = det.tick(1000 + 299 * 0.5 + 4.0, hist, audio, [])   # 4 с после последней пачки
    assert sig["audio_db"] == -20.0
