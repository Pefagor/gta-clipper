"""SQLite: каналы, сессии записи, кандидаты, расходы, настройки."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    login TEXT PRIMARY KEY,
    display_name TEXT,
    user_id TEXT,
    state TEXT NOT NULL DEFAULT 'active',      -- active | paused | removing | removed
    added_at REAL,
    removed_at REAL,
    cam_status TEXT NOT NULL DEFAULT 'unknown', -- unknown | pending | ok | none
    cam_mode TEXT,                              -- NULL = как по умолчанию | auto | off | media
    alias TEXT,                                 -- как стримера называть в заголовках (NULL = ник)
    cam_x REAL, cam_y REAL, cam_w REAL, cam_h REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    dir TEXT NOT NULL,
    started_at REAL,
    ended_at REAL
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL DEFAULT 'clip',          -- clip | compilation
    channel TEXT,
    session_id INTEGER,
    status TEXT NOT NULL,
    trigger TEXT,
    peak_ts REAL, win_start REAL, win_end REAL,
    score REAL,
    signals TEXT,
    dir TEXT,
    raw_start_ts REAL, raw_duration REAL,
    screen TEXT, review TEXT,
    clip_start REAL, clip_end REAL,
    edit_start REAL, edit_end REAL,
    titles TEXT, title TEXT, description TEXT,
    category TEXT, tag TEXT, spoiler INTEGER DEFAULT 0, game TEXT,
    overlay_text TEXT,                          -- надпись в начале видео: NULL = как заголовок, '' = без надписи
    render_path TEXT, render_duration REAL,
    parts TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_try_at REAL NOT NULL DEFAULT 0,
    card_msg_id INTEGER, pkg_msg_id INTEGER,
    decided_at REAL, sent_at REAL,
    defer_until REAL NOT NULL DEFAULT 0,
    pub_shorts INTEGER NOT NULL DEFAULT 0,
    pub_tiktok INTEGER NOT NULL DEFAULT 0,
    in_compilation INTEGER NOT NULL DEFAULT 0,
    created_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_cand_status ON candidates(status);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, model TEXT, in_tok INTEGER, out_tok INTEGER, cost_rub REAL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_FIELDS = {"signals", "screen", "review", "titles", "parts"}


class DB:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Новые колонки в уже существующей базе."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(candidates)")}
        if "game" not in cols:
            self.conn.execute("ALTER TABLE candidates ADD COLUMN game TEXT")
        if "overlay_text" not in cols:
            self.conn.execute("ALTER TABLE candidates ADD COLUMN overlay_text TEXT")
        ch_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(channels)")}
        if "cam_mode" not in ch_cols:
            self.conn.execute("ALTER TABLE channels ADD COLUMN cam_mode TEXT")
        if "alias" not in ch_cols:
            self.conn.execute("ALTER TABLE channels ADD COLUMN alias TEXT")

    # --- низкий уровень ---
    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def all(self, sql: str, params: tuple | list = ()) -> list[dict]:
        with self._lock:
            return [_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | list = ()) -> dict | None:
        with self._lock:
            r = self.conn.execute(sql, params).fetchone()
            return _row(r) if r else None

    def scalar(self, sql: str, params: tuple | list = ()) -> Any:
        with self._lock:
            r = self.conn.execute(sql, params).fetchone()
            return r[0] if r else None

    # --- kv-настройки ---
    def kv_get(self, key: str, default: Any = None) -> Any:
        v = self.scalar("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(v) if v is not None else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value, ensure_ascii=False)))

    # --- кандидаты ---
    def add_candidate(self, **fields) -> int:
        now = time.time()
        fields.setdefault("created_at", now)
        fields["updated_at"] = now
        fields = {k: _enc(k, v) for k, v in fields.items()}
        cols = ",".join(fields)
        q = ",".join("?" * len(fields))
        return self.execute(f"INSERT INTO candidates({cols}) VALUES({q})", list(fields.values())).lastrowid

    def update_candidate(self, cid: int, **fields) -> None:
        fields["updated_at"] = time.time()
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE candidates SET {sets} WHERE id=?", [_enc(k, v) for k, v in fields.items()] + [cid])

    def candidate(self, cid: int) -> dict | None:
        return self.one("SELECT * FROM candidates WHERE id=?", (cid,))

    def next_candidate(self, status: str) -> dict | None:
        return self.one("SELECT * FROM candidates WHERE status=? AND next_try_at<=? ORDER BY id LIMIT 1",
                        (status, time.time()))

    def count(self, status: str) -> int:
        return self.scalar("SELECT COUNT(*) FROM candidates WHERE status=?", (status,)) or 0

    # --- каналы ---
    def channel(self, login: str) -> dict | None:
        return self.one("SELECT * FROM channels WHERE login=?", (login.lower(),))

    def channels(self, states: tuple[str, ...] = ("active", "paused", "removing")) -> list[dict]:
        q = ",".join("?" * len(states))
        return self.all(f"SELECT * FROM channels WHERE state IN ({q}) ORDER BY added_at", states)

    def update_channel(self, login: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE channels SET {sets} WHERE login=?", list(fields.values()) + [login.lower()])

    # --- расходы ---
    def add_spend(self, model: str, in_tok: int, out_tok: int, cost: float) -> None:
        self.execute("INSERT INTO spend(ts,model,in_tok,out_tok,cost_rub) VALUES(?,?,?,?,?)",
                     (time.time(), model, in_tok, out_tok, cost))

    def spend_since(self, ts: float) -> float:
        return self.scalar("SELECT COALESCE(SUM(cost_rub),0) FROM spend WHERE ts>=?", (ts,)) or 0.0


def _enc(key: str, value: Any) -> Any:
    if key in JSON_FIELDS and value is not None and not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return value


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in JSON_FIELDS & d.keys():
        if d[k]:
            try:
                d[k] = json.loads(d[k])
            except (TypeError, ValueError):
                pass
    return d


def day_start(now: float | None = None) -> float:
    lt = time.localtime(now or time.time())
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
