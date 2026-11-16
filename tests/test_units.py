import time
from collections import deque
from pathlib import Path

from clipper.chat import parse_privmsg
from clipper.detector import ChannelDetector, count_recent
from clipper.llm import parse_json
from clipper.render import build_ass, group_words, target_kbps, video_filter
from clipper.segments import PartIndexer, Segment, parse_playlist, pick_window

DET_CFG = {"window_seconds": 10, "baseline_seconds": 600, "warmup_seconds": 0, "emotes": ["KEKW", "ахах"],
           "weights": {"clips": 3, "chat": 2, "audio": 1.5, "emotes": 1}}


def test_parse_playlist():
    text = "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4.000,\nseg_000001.ts\n#EXTINF:3.5,\nseg_000002.ts\n"
    assert parse_playlist(text) == [("seg_000001.ts", 4.0), ("seg_000002.ts", 3.5)]


def test_indexer_burst_spreads_backwards(tmp_path: Path):
    for i in (1, 2, 3):
        (tmp_path / f"seg_{i}.ts").write_bytes(b"x")
    (tmp_path / "index.m3u8").write_text("".join(f"#EXTINF:4.0,\nseg_{i}.ts\n" for i in (1, 2, 3)))
    idx = PartIndexer(tmp_path)
    segs = idx.poll(now=1000.0)
    assert [(s.start, s.end) for s in segs] == [(988.0, 992.0), (992.0, 996.0), (996.0, 1000.0)]
    assert idx.poll(now=1001.0) == []  # уже видели


def test_pick_window_stops_at_gaps():
    p = Path("/a/part_001")
    segs = [Segment(p / "1.ts", 0, 4), Segment(p / "2.ts", 4, 8), Segment(p / "3.ts", 30, 34), Segment(p / "4.ts", 34, 38)]
    run = pick_window(segs, 0, 40, anchor=32)
    assert [s.path.name for s in run] == ["3.ts", "4.ts"]
    run = pick_window(segs, 0, 40, anchor=2)
    assert [s.path.name for s in run] == ["1.ts", "2.ts"]


def test_parse_privmsg():
    line = ("@badge-info=;tmi-sent-ts=1700000000123;user-id=1 :bob!bob@bob.tmi.twitch.tv PRIVMSG #SomeChan :KEKW ахаха")
    ch, text, ts = parse_privmsg(line)
    assert ch == "somechan" and text == "KEKW ахаха" and abs(ts - 1700000000.123) < 1e-6
    assert parse_privmsg(":tmi.twitch.tv 001 justinfan :Welcome") is None


def test_detector_spike_scores_high():
    det = ChannelDetector(DET_CFG)
    hist, audio = deque(), deque()
    t0 = 10_000.0
    score = 0.0
    for step in range(200):  # 400 с спокойного чата: 2 сообщения за 10 с
        now = t0 + step * 2
        hist.append((now, "привет"))
        audio.append((now, -30.0))
        score, _ = det.tick(now, hist, audio, [])
    assert score < 3
    now = t0 + 400
    for i in range(40):  # всплеск: 40 сообщений с эмоутами и громкий звук
        hist.append((now - 5 + i * 0.1, "KEKW ахахах"))
    audio.append((now, -10.0))
    score, sig = det.tick(now, hist, audio, [now - 5])
    assert score > 8, sig
    assert sig["clips"] == 1


def test_count_recent():
    hist = [(1.0, "a"), (5.0, "KEKW"), (9.0, "b")]
    assert count_recent(hist, 4, 10, ["kekw"]) == (2, 1)


def test_parse_json_variants():
    assert parse_json('```json\n{"score": 7}\n```') == {"score": 7}
    assert parse_json('Вот ответ: {"a": 1} ок') == {"a": 1}


def test_group_and_ass():
    words = [{"start": 0.0, "end": 0.3, "word": "это"}, {"start": 0.3, "end": 0.6, "word": "было"},
             {"start": 0.6, "end": 1.0, "word": "{жёстко}"}, {"start": 3.0, "end": 3.4, "word": "да"}]
    # «это было {жёстко}» — 17 символов, в одну строку не влезет: третье слово уходит в новую фразу
    assert [len(g) for g in group_words(words)] == [2, 1, 1]
    ass = build_ass(words, "Заголовок", "twitch.tv/x", has_cam=True)
    assert "Style: Sub" in ass and "Заголовок" in ass and "twitch.tv/x" in ass
    assert "{жёстко}" not in ass and "(ЖЁСТКО)" in ass
    assert ass.count("Style: Sub") == 1 and ass.count(",Sub,") == 4
    # каждая строка субтитров прибита к одной точке
    assert ass.count(r"{\an2\pos(540,1500)}") == 4


def test_subtitle_events_never_overlap():
    """Наложение двух строк по времени заставляет libass раздвигать их — субтитры «прыгают»."""
    from clipper.render import subtitle_events
    words = [{"start": 0.0, "end": 0.5, "word": "раз"}, {"start": 0.5, "end": 1.4, "word": "два"},
             {"start": 1.2, "end": 1.6, "word": "три"},  # Whisper иногда даёт слова внахлёст
             {"start": 1.6, "end": 1.9, "word": "четыре"}, {"start": 2.2, "end": 2.5, "word": "пять"},
             {"start": 6.0, "end": 6.4, "word": "шесть"}]
    ev = subtitle_events(words)
    for (s1, e1, _), (s2, _, _) in zip(ev, ev[1:]):
        assert e1 <= s2 + 1e-9
    assert len(ev) == 6
    assert ev[-2][1] < 6.0  # длинная пауза — строка исчезает, а не висит до следующей фразы


def test_filters_and_bitrate():
    assert "vstack" in video_filter(1920, 1080, (0.7, 0.6, 0.25, 0.3))
    assert "boxblur" in video_filter(1920, 1080, None)
    assert target_kbps(30, 45, 8000) == 8000
    assert target_kbps(90, 45, 8000) < 4000
