import sqlite3
from pathlib import Path

DB_PATH = Path("data/events.db")


def get_conn(db_path=None):
    """db_path를 안 주면 운영 DB(data/events.db) 사용. 평가 때는 평가용 DB 경로를 넘김."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(path) as conn:
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
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(events)")]
        if "llm_summary" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN llm_summary TEXT")