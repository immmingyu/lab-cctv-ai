import sqlite3
from pathlib import Path

DB_PATH = Path("data/events.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                event_type TEXT NOT NULL,
                timestamp TEXT,
                created_at TEXT NOT NULL,
                object TEXT,
                location TEXT,
                action TEXT,
                judgment TEXT,
                risk TEXT,
                confidence REAL,
                duration_seconds REAL,
                summary TEXT,
                clip_path TEXT,
                llm_summary TEXT
            )
        """)
        # 예전에 만든 DB에는 llm_summary 칸이 없으므로 있으면 건너뛰고, 없으면 추가
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(events)")]
        if "llm_summary" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN llm_summary TEXT")