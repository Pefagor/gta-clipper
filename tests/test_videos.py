"""Видео по ссылке: нарезка на части, сигналы, чат записи стрима, задание целиком на своём файле."""
import asyncio
import json
import subprocess
from pathlib import Path

from clipper.videos import (chat_peaks, heat_peaks, merge_windows, parse_rechat, parse_ts, slice_transcript,
                            split_parts)


def _segments(duration: float, every: float = 6.0, pause: float = 0.8):
    t, segs = 0.0, []
    while t + every <= duration:
        segs.append({"start": t, "end": t + every - pause, "text": "фраза", "words": []})
        t += every
    return segs


def test_split_parts_cuts_in_pauses_and_merges_tail():
    segs = _segments(600)
    parts = split_parts(segs, 600, target=75, min_len=45, max_len=120)
    assert parts[0][0] == 0 and parts[-1][1] == 600
    for (a, b), (c, _) in zip(parts, parts[1:]):
        assert b == c and 45 <= b - a <= 120
        # разрез — в паузе между фразами, а не посреди фразы
        assert any(abs(b - (s["end"] + 0.4)) < 0.01 for s in segs)
    short = split_parts(_segments(130), 130, target=75, min_len=45, max_len=120)
    assert len(short) <= 2 and short[-1][1] == 130


def test_heat_and_chat_peaks():
    hm = [{"start_time": i, "end_time": i + 1, "value": v} for i, v in enumerate([1, .2, .2, .3, .9, .3, .2, .2, .7, .2])]
    peaks = heat_peaks(hm, duration=600)
    assert [round(t, 1) for t, _ in peaks] == []        # всё в первых 3% длины (начало видео всегда «горит»)
    hm = [{"start_time": i * 60, "end_time": i * 60 + 60, "value": v} for i, v in enumerate([1, .2, .2, .9, .3, .2, .8, .2])]
    assert [int(t) for t, _ in heat_peaks(hm, 480)] == [210, 390]
    chat = [(float(t), "x") for t in range(0, 600, 5)] + [(300.0 + i * 0.1, "KEKW") for i in range(60)]
    assert [int(t) for t, _ in chat_peaks(chat, 600)] == [305]


def test_merge_windows_and_parse_ts():
    ws = merge_windows([{"start": 0, "end": 40, "score": 5, "sources": ["чат"]},
                        {"start": 30, "end": 70, "score": 7, "sources": ["ИИ"], "title": "т"},
                        {"start": 200, "end": 240, "score": 4, "sources": ["крик"]}])
    assert len(ws) == 2 and ws[0]["end"] == 70 and ws[0]["title"] == "т" and ws[0]["sources"] == ["ИИ", "чат"]
    assert parse_ts("1:02:03") == 3723 and parse_ts("12:34") == 754 and parse_ts(90) == 90 and parse_ts("abc") is None


def test_parse_rechat_and_slice(tmp_path):
    p = tmp_path / "source.rechat.json"
    p.write_text(json.dumps({"comments": [{"content_offset_seconds": 12.5, "message": {"body": "KEKW"}},
                                          {"content_offset_seconds": 3, "message": {"fragments": [{"text": "привет"}]}}]}))
    assert parse_rechat(p) == [(3.0, "привет"), (12.5, "KEKW")]
    tr = {"segments": [{"start": 10, "end": 14, "text": "a", "words": [{"start": 10, "end": 11, "word": "a"}]},
                       {"start": 50, "end": 52, "text": "b", "words": []}],
          "loud": [{"start": 12, "end": 13, "over_db": 12}]}
    sl = slice_transcript(tr, 8, 30)
    assert len(sl["segments"]) == 1 and sl["segments"][0]["start"] == 2 and sl["segments"][0]["words"][0]["start"] == 2
    assert sl["loud"][0]["start"] == 4


class FakeTranscriber:
    def transcribe(self, wav):
        return {"language": "ru", "segments": _segments(150)}


def test_parts_job_on_local_file(tmp_path):
    """Задание «на части» на своём файле: части нарезаны, лежат в очереди монтажа и монтируются с «ЧАСТЬ N/M»."""
    from clipper import app as app_mod
    from clipper.config import load_config
    from clipper.tools import find_ffmpeg
    cfg = load_config()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.secrets.routerai_key = ""
    app = app_mod.App(cfg)
    app.transcriber = FakeTranscriber()
    src = tmp_path / "my video.mp4"
    subprocess.run([find_ffmpeg(), "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=150",
                    "-f", "lavfi", "-i", "sine=f=220:d=150", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    "-shortest", str(src)], check=True)

    async def go():
        vid = app.videos.add(str(src), "parts")
        n = await app.videos.process(app.db.one("SELECT * FROM videos WHERE id=?", (vid,)))
        cands = app.db.all("SELECT * FROM candidates WHERE video_id=? ORDER BY id", (vid,))
        assert n == len(cands) == 2
        assert [c["part"] for c in cands] == ["1/2", "2/2"] and all(c["status"] == "to_render" for c in cands)
        assert src.exists()  # свой файл не удаляется
        await app._do_render(app.db.candidate(cands[1]["id"]))
        c = app.db.candidate(cands[1]["id"])
        assert c["status"] == "rendered" and Path(c["render_path"]).exists()
        assert "ЧАСТЬ 2/2" in (Path(c["dir"]) / "subs.ass").read_text(encoding="utf-8")

    asyncio.run(go())
