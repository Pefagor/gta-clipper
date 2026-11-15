"""Рендер вертикального ролика 1080x1920: вебка сверху, игра снизу (или игра на размытом фоне),
субтитры с подсветкой текущего слова, заголовок в начале и ник стримера."""
from __future__ import annotations

from pathlib import Path

from .tools import encoder_args, run

CAM_H = 640  # высота области вебки в макете с вебкой


def _ass_time(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[dict], max_words: int = 3, max_gap: float = 0.8) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in words:
        if groups and len(groups[-1]) < max_words and w["start"] - groups[-1][-1]["end"] <= max_gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def build_ass(words: list[dict], title: str, credit: str, has_cam: bool, font: str = "Arial",
              title_seconds: float = 2.5, width: int = 1080, height: int = 1920) -> str:
    """words — со временем относительно начала ролика."""
    sub_margin = 420 if has_cam else 520
    title_margin = CAM_H + 40 if has_cam else 300
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font},78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,{sub_margin},1
Style: Title,{font},62,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,3,14,0,8,70,70,{title_margin},1
Style: Credit,{font},40,&H30FFFFFF,&H30FFFFFF,&H80000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,40,40,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if title:
        events.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(title_seconds)},Title,,0,0,0,,{_esc(title)}")
    if credit:
        events.append(f"Dialogue: 0,{_ass_time(0)},{_ass_time(3600)},Credit,,0,0,0,,{_esc(credit)}")
    for group in group_words(words):
        texts = [_esc(w["word"]).upper() for w in group]
        for i, w in enumerate(group):
            start = w["start"]
            end = group[i + 1]["start"] if i + 1 < len(group) else max(w["end"], start + 0.3)
            if end <= start:
                continue
            line = " ".join((r"{\c&H00E5FF&}" + t + r"{\c&HFFFFFF&}") if j == i else t for j, t in enumerate(texts))
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Sub,,0,0,0,,{line}")
    return head + "\n".join(events) + "\n"


def video_filter(src_w: int, src_h: int, cam: tuple[float, float, float, float] | None,
                 width: int = 1080, height: int = 1920, fps: int = 30) -> str:
    """filter_complex: на выходе [v] (с наложенными субтитрами из subs.ass)."""
    if cam:
        cx, cy, cw, ch = cam
        x, y = int(cx * src_w), int(cy * src_h)
        w, h = max(int(cw * src_w) // 2 * 2, 32), max(int(ch * src_h) // 2 * 2, 32)
        game_h = height - CAM_H
        game_w_src = int(src_h * width / game_h) // 2 * 2
        return (f"[0:v]split=2[c][g];"
                f"[c]crop={w}:{h}:{x}:{y},scale={width}:{CAM_H}:force_original_aspect_ratio=increase,"
                f"crop={width}:{CAM_H}[cam];"
                f"[g]crop={min(game_w_src, src_w)}:{src_h},scale={width}:{game_h}[game];"
                f"[cam][game]vstack=inputs=2,fps={fps},setsar=1,ass=subs.ass[v]")
    fg_h = int(width * 3 / 4) // 2 * 2
    return (f"[0:v]split=2[b][f];"
            f"[b]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=20:2[bg];"
            f"[f]crop=ih*4/3:ih,scale={width}:{fg_h}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={fps},setsar=1,ass=subs.ass[v]")


def target_kbps(duration: float, max_mb: float, max_kbps: int, audio_kbps: int = 160) -> int:
    budget = max_mb * 8 * 1000 / max(duration, 1) - audio_kbps
    return int(max(1500, min(max_kbps, budget * 0.95)))


async def render_clip(ffmpeg: str, encoder: str, raw: Path, out: Path, start: float, end: float,
                      src_w: int, src_h: int, cam, words: list[dict], title: str, credit: str,
                      rcfg: dict) -> Path:
    work = out.parent
    work.mkdir(parents=True, exist_ok=True)
    rel_words = [{**w, "start": w["start"] - start, "end": w["end"] - start}
                 for w in words if w["end"] > start and w["start"] < end]
    (work / "subs.ass").write_text(build_ass(rel_words, title, credit, bool(cam), rcfg.get("font", "Arial")),
                                   encoding="utf-8")
    dur = end - start
    kbps = target_kbps(dur, rcfg.get("max_file_mb", 45), rcfg.get("max_video_kbps", 8000))
    fc = video_filter(src_w, src_h, cam, rcfg.get("width", 1080), rcfg.get("height", 1920), rcfg.get("fps", 30))
    lufs = rcfg.get("loudness_lufs", -14)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
           "-i", str(raw), "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0?",
           *encoder_args(encoder, kbps), "-pix_fmt", "yuv420p",
           "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample=48000", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", out.name]
    code, log = await run(cmd, cwd=work, timeout=900)
    if code != 0 or not out.exists():
        raise RuntimeError(f"рендер не удался: {log[-500:]}")
    return out


async def render_compilation(ffmpeg: str, encoder: str, parts: list[Path], out: Path, rcfg: dict,
                             total: float) -> Path:
    inputs = []
    for p in parts:
        inputs += ["-i", str(p)]
    n = len(parts)
    fc = "".join(f"[{i}:v]setsar=1[v{i}];[{i}:a]aresample=48000[a{i}];" for i in range(n))
    fc += "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
    kbps = target_kbps(total, rcfg.get("max_file_mb", 45), rcfg.get("max_video_kbps", 8000))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", fc,
           "-map", "[v]", "-map", "[a]", *encoder_args(encoder, kbps), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    code, log = await run(cmd, timeout=1200)
    if code != 0 or not out.exists():
        raise RuntimeError(f"подборка не собралась: {log[-500:]}")
    return out
