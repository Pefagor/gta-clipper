"""Конвейер кандидата целиком: вырезка → (заглушка Whisper) → без ИИ → рендер → модерация."""
import asyncio
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import pytest

from clipper.config import load_config
from clipper.detector import ChannelDetector
from clipper.llm import as_bool, as_score
from clipper.tools import find_ffmpeg


class FakeTranscriber:
    def transcribe(self, wav: Path) -> dict:
        assert wav.exists()
        return {"language": "ru", "segments": [{"start": 10.0, "end": 11.0, "text": "вот это да",
                                                "words": [{"start": 10.0, "end": 10.3, "word": "вот"},
                                                          {"start": 10.3, "end": 10.6, "word": "это"},
                                                          {"start": 10.6, "end": 11.0, "word": "да"}]}]}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    from clipper import app as app_mod
    cfg = load_config()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.secrets.routerai_key = ""  # без ИИ
    a = app_mod.App(cfg)
    a.transcriber = FakeTranscriber()
    return a


def test_candidate_goes_through_pipeline(app, tmp_path):
    ff = find_ffmpeg()
    sess = tmp_path / "sess"
    part = sess / "part_001"
    part.mkdir(parents=True)
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=60",
                    "-f", "lavfi", "-i", "sine=f=300:d=60", "-c:v", "libx264", "-g", "60", "-preset", "ultrafast",
                    "-c:a", "aac", "-f", "hls", "-hls_time", "4", "-hls_list_size", "0",
                    "-hls_segment_filename", "seg_%06d.ts", "index.m3u8"], cwd=part, check=True)
    t0 = time.time() - 100
    segs = sorted(part.glob("seg_*.ts"))
    (part / "segments.jsonl").write_text("".join(
        json.dumps({"file": s.name, "start": t0 + i * 4, "end": t0 + (i + 1) * 4}) + "\n" for i, s in enumerate(segs)))
    sid = app.db.execute("INSERT INTO sessions(channel,dir,started_at) VALUES(?,?,?)", ("tester", str(sess), t0)).lastrowid
    app.db.execute("INSERT INTO channels(login,display_name,state,added_at) VALUES('tester','tester','active',?)", (t0,))
    peak = t0 + 30
    cid = app.db.add_candidate(channel="tester", session_id=sid, status="to_cut", trigger="detector", peak_ts=peak,
                               win_start=peak - 20, win_end=peak + 15, score=7, signals={"chat": 30})

    async def go():
        await app._do_cut(app.db.candidate(cid))
        assert app.db.candidate(cid)["status"] == "cut"
        await app._do_transcribe(app.db.candidate(cid))
        assert app.db.candidate(cid)["status"] == "transcribed"
        await app._do_screen(app.db.candidate(cid))
        c = app.db.candidate(cid)
        assert c["status"] == "to_render" and c["title"]
        await app._do_render(c)
        c = app.db.candidate(cid)
        assert c["status"] == "rendered" and Path(c["render_path"]).exists()
        assert 30 <= c["render_duration"] <= 40

        # неудачный перерендер возвращает прежний клип на модерацию, а не теряет его
        app.db.update_candidate(cid, status="to_render", edit_start=5.0, edit_end=1.0, attempts=2)
        await app._stage_failed(app.db.candidate(cid), "to_render", RuntimeError("boom"), 3)
        assert app.db.candidate(cid)["status"] == "rendered"

        # отсеянный ИИ кандидат: причина видна, превью смотрится, можно вернуть в работу
        app.db.update_candidate(cid, status="discarded", screen={"score": 3, "summary": "стример объясняет правила"})
        c = app.db.candidate(cid)
        assert [r["id"] for r in app.rejected_list()] == [cid]
        assert "Haiku: 3/10" in app.reject_reason(c)
        prev = await app.preview(c)
        assert prev.exists() and prev.stat().st_size < 20_000_000
        assert app.rescue(cid) == "в монтаж, потом придёт на модерацию"
        c = app.db.candidate(cid)
        assert c["status"] == "to_render" and c["title"] == "стример объясняет правила"
        assert app.rescue(cid) is None  # уже в работе — повторно не возвращается

    asyncio.run(go())


def test_unmute_after_silence_does_not_trigger():
    cfg = {"window_seconds": 10, "baseline_seconds": 600, "warmup_seconds": 0, "emotes": [],
           "weights": {"chat": 2, "audio": 1.5, "emotes": 1}}
    det = ChannelDetector(cfg)
    hist, audio = deque(), deque()
    t0 = 1000.0
    for step in range(150):  # 5 минут тишины (BRB), чат молчит
        now = t0 + step * 2
        audio.append((now, -100.0))
        det.tick(now, hist, audio, [])
    now = t0 + 300
    audio.append((now, -20.0))  # стример вернулся и заговорил
    score, _ = det.tick(now, hist, audio, [])
    assert score < 4.5


def test_llm_value_parsing():
    assert as_bool("false", True) is False and as_bool("да", False) is True and as_bool(None, True) is True
    assert as_score("8/10") == 8 and as_score(7.6) == 8 and as_score("15") == 10


def test_compilations_do_not_mix_games(app):
    for game in ("Dota 2", "Dota 2", "Dota 2", "Grand Theft Auto VI", "Grand Theft Auto VI"):
        app.db.add_candidate(kind="clip", channel="x", status="approved", tag="смешной баг", game=game)
    assert app.compilation_tags() == [("Dota 2", "смешной баг", 3)]


def test_db_migration_adds_game_column(tmp_path):
    import sqlite3
    from clipper.db import DB
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, channel TEXT, "
                "status TEXT NOT NULL, created_at REAL, updated_at REAL)")
    con.commit()
    con.close()
    db = DB(path)
    db.add_candidate(channel="x", status="to_cut", game="Dota 2")
    assert db.one("SELECT game FROM candidates")["game"] == "Dota 2"


def test_cam_mode_and_overlay_resolution(app):
    app.db.execute("INSERT INTO channels(login,display_name,state,added_at) VALUES('a','a','active',0)")
    assert app.cam_mode("a") == "auto"                  # по умолчанию — искать вебку
    app.set_setting("cam_default", "off")
    assert app.cam_mode("a") == "off"                   # общий режим
    app.db.update_channel("a", cam_mode="media")
    assert app.cam_mode("a") == "media"                 # режим канала важнее общего
    assert app.channel_overlay("a") == (None, "нет файла")
    (app.overlay_dir() / "_default.png").write_bytes(b"x")
    assert app.channel_overlay("a")[1] == "общая"
    (app.overlay_dir() / "a.mp4").write_bytes(b"x")
    assert app.channel_overlay("a")[1] == "своя"        # своя заставка канала важнее общей


def test_cam_rect_only_after_confirmation(app):
    app.db.execute("INSERT INTO channels(login,display_name,state,added_at,cam_status) VALUES('b','b','active',0,'pending')")
    app.db.kv_set("cam:b", [0.79, 0.0, 0.21, 0.26])
    app.db.kv_set("cam_method:b", "edges")
    assert app.cam_rect("b") is None            # до «✅ Верно» вебки в монтаже нет, даже найденной по краям
    app.accept_cam("b", [0.79, 0.0, 0.21, 0.26])
    assert app.cam_rect("b") == (0.79, 0.0, 0.21, 0.26)
