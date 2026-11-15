"""Поиск вебки стримера на кадрах стрима. Запускается отдельным процессом (на macOS OpenCV и PyAV
конфликтуют в одном процессе):

    python -m clipper.camcal find  preview.jpg frame1.jpg frame2.jpg ...
        → {"rect": [x, y, w, h] в долях кадра | null, "method": "edges" | "estimate" | null, "face": [...]}
    python -m clipper.camcal check x y w h frame1.jpg ...
        → {"present": true|false, "hits": N}

Как ищется:
1. Лица — нейросетевым детектором YuNet (models/face_detection_yunet_2023mar.onnx).
2. Выбирается лицо, которое стоит на одном месте почти во всех кадрах и крупнее остальных:
   камеры игроков в трансляциях турниров и случайные «лица» в игре отсеиваются.
3. Рамка вебки — прямоугольник вокруг лица, все четыре стороны которого являются «постоянными»
   линиями: край, который виден почти на всех кадрах, пока игра вокруг меняется. Край кадра тоже
   считается стороной. Линии полок и мебели за спиной стримера не образуют замкнутой рамки.
4. Если рамку найти не удалось — оценка по размеру лица (method="estimate"), её стоит поправить вручную.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
MIN_FACE = 0.018      # минимальная ширина лица в долях кадра (меньше — иконки и мелочь в игре)
MIN_CONF = 0.8
MIN_PRESENCE = 0.5    # лицо должно быть хотя бы на половине кадров


def _detector(w: int, h: int, conf: float = MIN_CONF):
    import cv2
    return cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), conf)


def detect_faces(img, det) -> list[tuple[float, float, float, float, float]]:
    """[(x, y, w, h, уверенность)] в пикселях."""
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None:
        return []
    return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces
            if f[2] >= MIN_FACE * w]


def pick_face(per_frame: list[list[tuple]], n_frames: int) -> tuple[float, float, float, float] | None:
    """Лицо, которое стабильно стоит на месте. Возвращает медианный бокс (x, y, w, h)."""
    clusters: list[list[tuple]] = []
    for faces in per_frame:
        for f in faces:
            cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
            for cl in clusters:
                ref = np.median([[g[0] + g[2] / 2, g[1] + g[3] / 2, g[2]] for g in cl], axis=0)
                if abs(cx - ref[0]) <= 1.5 * ref[2] and abs(cy - ref[1]) <= 1.5 * ref[2]:
                    cl.append(f)
                    break
            else:
                clusters.append([f])
    best, best_score = None, 0.0
    for cl in clusters:
        presence = min(1.0, len(cl) / max(n_frames, 1))
        if presence < MIN_PRESENCE:
            continue
        size = float(np.median([g[2] for g in cl]))
        score = presence * size
        if score > best_score:
            best, best_score = cl, score
    if not best:
        return None
    return tuple(float(v) for v in np.median([g[:4] for g in best], axis=0))


def edge_persistence(gray_frames: list) -> np.ndarray:
    """Доля кадров, на которых пиксель — край (с допуском ±1 пиксель)."""
    import cv2
    acc = np.mean([cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120) > 0 for g in gray_frames], axis=0)
    return cv2.dilate(acc.astype(np.float32), np.ones((3, 3), np.uint8))


def find_rect(E: np.ndarray, face: tuple[float, float, float, float]) -> tuple[tuple[int, int, int, int] | None, float]:
    """Лучший прямоугольник (L, T, R, B) в пикселях вокруг лица и его оценка 0..1."""
    h, w = E.shape
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2

    def peaks(profile, lo, hi, edges):
        lo, hi = max(int(lo), 0), min(int(hi), len(profile) - 1)
        cand = [i for i in range(lo + 1, hi)
                if profile[i] >= 0.06 and profile[i] == profile[max(lo, i - 4):i + 5].max()]
        cand += [e for e in edges if lo <= e <= hi]
        inner = lambda i: 0 < i < len(profile) - 1  # noqa: E731
        return sorted(set(cand), key=lambda i: -profile[i] if inner(i) else -1.0)[:12]

    rows = slice(int(max(0, cy - 2 * fh)), int(min(h, cy + 2 * fh)))
    cols = slice(int(max(0, cx - 1.5 * fw)), int(min(w, cx + 1.5 * fw)))
    colp, rowp = E[rows].mean(axis=0), E[:, cols].mean(axis=1)
    Ls = peaks(colp, cx - 8 * fw, fx - 0.3 * fw, [0])
    Rs = peaks(colp, fx + 1.3 * fw, cx + 8 * fw, [w - 1])
    Ts = peaks(rowp, cy - 7 * fh, fy - 0.3 * fh, [0])
    Bs = peaks(rowp, fy + 1.6 * fh, cy + 8 * fh, [h - 1])

    def side(a0, a1, at, vertical):
        if vertical:
            if at <= 1 or at >= w - 2:
                return 1.0
            band = E[int(a0):int(a1), int(at)]
        else:
            if at <= 1 or at >= h - 2:
                return 1.0
            band = E[int(at), int(a0):int(a1)]
        return float((band >= 0.3).mean()) if band.size else 0.0

    best, best_score = None, 0.0
    for L, R, T, B in itertools.product(Ls, Rs, Ts, Bs):
        rw, rh = R - L, B - T
        if rh <= 0 or not (0.9 <= rw / rh <= 2.4 and 2.0 * fw <= rw <= 9 * fw):
            continue
        s = [side(T, B, L, True), side(T, B, R, True), side(L, R, T, False), side(L, R, B, False)]
        score = min(s) * 0.6 + float(np.mean(s)) * 0.4 + 0.02 * (rw * rh) / (w * h)
        if score > best_score:
            best, best_score = (L, T, R, B), score
    return best, round(best_score, 3)


def estimate_rect(face, w: int, h: int) -> tuple[int, int, int, int]:
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    rw = min(fw * 4.2, w * 0.45)
    rh = rw * 0.75
    L = int(min(max(cx - rw / 2, 0), w - rw))
    T = int(min(max(cy - rh * 0.45, 0), h - rh))
    return L, T, int(L + rw), int(T + rh)


def find_cam(paths: list[str]) -> dict:
    import cv2
    imgs = [img for img in (cv2.imread(p) for p in paths) if img is not None]
    if not imgs:
        return {"rect": None, "method": None, "face": None, "error": "нет кадров"}
    h, w = imgs[0].shape[:2]
    imgs = [i for i in imgs if i.shape[:2] == (h, w)]
    det = _detector(w, h)
    face = pick_face([detect_faces(i, det) for i in imgs], len(imgs))
    if not face:
        return {"rect": None, "method": None, "face": None, "frame": [w, h]}
    E = edge_persistence([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs])
    rect, score = find_rect(E, face)
    method = "edges"
    if rect is None or score < 0.5:
        rect, method = estimate_rect(face, w, h), "estimate"
    L, T, R, B = rect
    return {"rect": [round(L / w, 4), round(T / h, 4), round((R - L) / w, 4), round((B - T) / h, 4)],
            "method": method, "score": score, "face": [round(v, 1) for v in face], "frame": [w, h]}


def check_cam(rect: list[float], paths: list[str]) -> dict:
    """Есть ли лицо внутри сохранённой рамки вебки хотя бы на одном кадре."""
    import cv2
    hits = 0
    det = None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        det = det or _detector(w, h, conf=0.7)
        x, y, rw, rh = rect[0] * w, rect[1] * h, rect[2] * w, rect[3] * h
        for f in detect_faces(img, det):
            fcx, fcy = f[0] + f[2] / 2, f[1] + f[3] / 2
            if x <= fcx <= x + rw and y <= fcy <= y + rh:
                hits += 1
                break
    return {"present": hits > 0, "hits": hits}


def draw_preview(src: str, out: str, rect: list[float] | None, color=(0, 255, 0)) -> None:
    import cv2
    img = cv2.imread(src)
    if img is None:
        return
    if rect:
        h, w = img.shape[:2]
        x, y, rw, rh = rect
        cv2.rectangle(img, (int(x * w), int(y * h)), (int((x + rw) * w) - 1, int((y + rh) * h) - 1), color, 6)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    mode = sys.argv[1]
    if mode == "find":
        out, frames = sys.argv[2], sys.argv[3:]
        res = find_cam(frames)
        if frames:
            draw_preview(frames[-1], out, res.get("rect"),
                         (0, 255, 0) if res.get("method") == "edges" else (0, 200, 255))
    elif mode == "check":
        rect = [float(v) for v in sys.argv[2:6]]
        res = check_cam(rect, sys.argv[6:])
    elif mode == "draw":  # draw src out x y w h
        draw_preview(sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4:8]])
        res = {"ok": True}
    else:
        raise SystemExit(f"неизвестный режим: {mode}")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
