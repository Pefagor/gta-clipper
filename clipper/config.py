"""Загрузка config.yaml и секретов из .env."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Secrets:
    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    telegram_token: str = ""
    telegram_owner_id: int | None = None
    youtube_key: str = ""
    routerai_key: str = ""

    @property
    def has_twitch(self) -> bool:
        return bool(self.twitch_client_id and self.twitch_client_secret)


class Config(dict):
    """Словарь настроек + пути + секреты."""

    root: Path
    data_dir: Path
    secrets: Secrets

    def section(self, name: str) -> dict:
        return self.get(name) or {}


class ConfigError(Exception):
    pass


def _read_yaml(path: Path) -> dict:
    """YAML с понятной ошибкой: файл, строка и что не так."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f", строка {mark.line + 1}" if mark else ""
        problem = str(getattr(e, "problem", "") or e).split("\n")[0]
        low = problem.lower()
        if "\t" in problem or "'\\t'" in problem:
            problem = "табуляция вместо пробелов (в YAML отступы — только пробелами)"
        elif "mapping values are not allowed" in low or "could not find expected ':'" in low:
            problem = "пропущено или лишнее двоеточие — проверьте эту и предыдущую строку (нужно «название: значение»)"
        elif "found unexpected end of stream" in low or "while scanning a quoted scalar" in low:
            problem = "не закрыты кавычки"
        elif "expected <block end>" in low or "did not find expected key" in low:
            problem = "сбились отступы — строки одного уровня должны начинаться с одинакового числа пробелов"
        raise ConfigError(f"{path.name}{where}: {problem}") from None


def clean_key(value: str) -> str:
    """Ключ без кавычек, пробелов, переносов и невидимых символов (часто прилипают при копировании)."""
    value = value.strip().strip('"').strip("'").strip()
    return "".join(ch for ch in value if ch.isprintable() and not ch.isspace())


def load_config(path: Path | None = None, env_path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    raw = _read_yaml(path) if path.exists() else {}
    local = ROOT / "config.local.yaml"  # необязательные личные переопределения
    if path == ROOT / "config.yaml" and local.exists():
        raw = _merge(raw, _read_yaml(local) or {})
    cfg = Config(raw)
    cfg.root = ROOT
    data_dir = Path(cfg.get("data_dir", "data"))
    cfg.data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(env_path or ROOT / ".env")
    owner = os.getenv("TELEGRAM_OWNER_ID", "").strip()
    cfg.secrets = Secrets(
        twitch_client_id=os.getenv("TWITCH_CLIENT_ID", "").strip(),
        twitch_client_secret=os.getenv("TWITCH_CLIENT_SECRET", "").strip(),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_owner_id=int(owner) if owner.lstrip("-").isdigit() else None,
        youtube_key=clean_key(os.getenv("YOUTUBE_API_KEY", "")),
        routerai_key=clean_key(os.getenv("ROUTERAI_API_KEY", "")),
    )
    return cfg
