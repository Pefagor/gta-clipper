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


@pytest.mark.skipif(not LOCAL.exists(), reason="кадры реального стрима есть только локально (не в git)")
def test_find_cam_real_stream(tmp_path):
    out = subprocess.run([sys.executable, "-m", "clipper.camcal", "find", str(tmp_path / "preview.jpg"),
                          *map(str, sorted(LOCAL.glob("*.jpg")))], capture_output=True, text=True, cwd=ROOT)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["method"] == "edges"
    x, y, w, h = res["rect"]
    # вебка stariy_bog — справа сверху; камера игрока внизу слева не должна выбираться
    assert abs(x - 0.793) < 0.01 and y < 0.01 and abs(x + w - 1.0) < 0.01 and abs(y + h - 0.264) < 0.01


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
