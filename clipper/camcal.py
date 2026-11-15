"""Поиск вебки на кадрах стрима (OpenCV). Запускается отдельным процессом:

    python -m clipper.camcal out.jpg frame1.jpg frame2.jpg ...

Печатает JSON {"rect": [x, y, w, h] в долях кадра | null}. На out.jpg рисуется найденная рамка.
Отдельный процесс нужен, потому что на macOS OpenCV и PyAV (faster-whisper) конфликтуют в одном процессе.
"""
from __future__ import annotations

import json
import sys


def find_cam(paths: list[str]) -> tuple[list[float] | None, object | None]:
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []  # (cx, cy, size) в долях кадра
    last = None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        last = img
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, fw, fh) in cascade.detectMultiScale(gray, 1.1, 6, minSize=(max(24, w // 40), max(24, w // 40))):
            faces.append(((x + fw / 2) / w, (y + fh / 2) / h, fw / w))
    if last is None or not faces:
        return None, last
    # лицо, которое повторяется на большинстве кадров (вебка стоит на месте)
    best, best_n = None, 0
    for f in faces:
        n = sum(1 for g in faces if abs(g[0] - f[0]) < 0.04 and abs(g[1] - f[1]) < 0.04)
        if n > best_n:
            best, best_n = f, n
    if best_n < max(2, len(paths) // 2):
        return None, last
    cx, cy, size = best
    h, w = last.shape[:2]
    cam_w = min(size * 3.4, 0.6)
    cam_h = min(cam_w * w / h * 0.75, 0.7)   # примерно 4:3 по пикселям
    x = min(max(cx - cam_w / 2, 0), 1 - cam_w)
    y = min(max(cy - cam_h * 0.45, 0), 1 - cam_h)
    return [round(x, 4), round(y, 4), round(cam_w, 4), round(cam_h, 4)], last


def main() -> None:
    import cv2

    out, frames = sys.argv[1], sys.argv[2:]
    rect, img = find_cam(frames)
    if img is not None:
        if rect:
            h, w = img.shape[:2]
            x, y, rw, rh = rect
            cv2.rectangle(img, (int(x * w), int(y * h)), (int((x + rw) * w), int((y + rh) * h)), (0, 255, 0), 4)
        cv2.imwrite(out, img)
    print(json.dumps({"rect": rect}))


if __name__ == "__main__":
    main()
