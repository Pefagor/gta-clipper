"""Сквозная проверка вырезки и рендера на синтетическом «стриме» (нужен ffmpeg с libass)."""
import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from clipper.cutter import cut_candidate, extract_frames
from clipper.render import render_clip, render_compilation
from clipper.tools import find_ffmpeg, find_ffprobe, pick_encoder, probe

FF = find_ffmpeg()
FP = find_ffprobe("auto", FF)
ENC = pick_encoder(FF)
RCFG = {"width": 1080, "height": 1920, "fps": 30, "max_file_mb": 45, "max_video_kbps": 4000}


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> Path:
    """Сессия как у рекордера: part_001 с HLS-сегментами по 4 с и segments.jsonl."""
    sess = tmp_path_factory.mktemp("sess")
    part = sess / "part_001"
    part.mkdir()
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=40",
                    "-f", "lavfi", "-i", "sine=f=440:d=40", "-c:v", "libx264", "-g", "60", "-preset", "ultrafast",
                    "-c:a", "aac", "-f", "hls", "-hls_time", "4", "-hls_list_size", "0",
                    "-hls_segment_filename", "seg_%06d.ts", "index.m3u8"], cwd=part, check=True)
    t0 = 1_000_000.0
    lines = []
    for i, seg in enumerate(sorted(part.glob("seg_*.ts"))):
        lines.append(json.dumps({"file": seg.name, "start": t0 + i * 4, "end": t0 + (i + 1) * 4}))
    (part / "segments.jsonl").write_text("\n".join(lines) + "\n")
    return sess


def test_cut_and_render(session: Path, tmp_path: Path):
    t0 = 1_000_000.0

    async def go():
        raw, raw_start, dur = await cut_candidate(FF, session, tmp_path, t0 + 6, t0 + 30, t0 + 15)
        assert raw.exists() and raw_start == t0 + 4 and 24 <= dur <= 32
        info = await probe(FP, raw)
        assert info["width"] == 1280 and info["duration"] > 20
        frames = await extract_frames(FF, raw, tmp_path, [1.0, 5.0], "f")
        assert len(frames) == 2
        words = [{"start": 2.0, "end": 2.4, "word": "привет"}, {"start": 2.4, "end": 3.0, "word": "мир"}]
        out_cam = await render_clip(FF, ENC, raw, tmp_path / "cam" / "clip.mp4", 1.0, 13.0, 1280, 720,
                                    (0.7, 0.6, 0.25, 0.35), words, "Тестовый заголовок", "twitch.tv/test", RCFG)
        out_nocam = await render_clip(FF, ENC, raw, tmp_path / "nocam" / "clip.mp4", 1.0, 11.0, 1280, 720,
                                      None, words, "Без вебки", "twitch.tv/test", RCFG)
        for out, expect in ((out_cam, 12), (out_nocam, 10)):
            v = await probe(FP, out)
            assert (v["width"], v["height"]) == (1080, 1920)
            assert abs(v["duration"] - expect) < 0.6
        comp = await render_compilation(FF, ENC, [out_cam, out_nocam], tmp_path / "comp.mp4", RCFG, 22)
        v = await probe(FP, comp)
        assert (v["width"], v["height"]) == (1080, 1920) and abs(v["duration"] - 22) < 1.0

    asyncio.run(go())


@pytest.mark.parametrize("kind", ["image", "video"])
def test_render_with_overlay(session: Path, tmp_path: Path, kind: str):
    """Вместо вебки сверху — заставка (картинка или видео, видео крутится по кругу)."""
    t0 = 1_000_000.0
    if kind == "image":
        media = tmp_path / "overlay.png"
        subprocess.run([FF, "-loglevel", "error", "-f", "lavfi", "-i", "color=c=orange:s=800x450", "-frames:v", "1",
                        str(media)], check=True)
    else:
        media = tmp_path / "overlay.mp4"  # короче клипа — должно зациклиться
        subprocess.run([FF, "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=640x360:r=30:d=3",
                        "-c:v", "libx264", "-preset", "ultrafast", str(media)], check=True)

    async def go():
        raw, _, _ = await cut_candidate(FF, session, tmp_path, t0 + 6, t0 + 30, t0 + 15)
        out = await render_clip(FF, ENC, raw, tmp_path / "ov" / "clip.mp4", 1.0, 11.0, 1280, 720, None,
                                [], "Заставка", "twitch.tv/test", RCFG, top_media=media)
        v = await probe(FP, out)
        assert (v["width"], v["height"]) == (1080, 1920) and abs(v["duration"] - 10) < 0.6

    asyncio.run(go())
