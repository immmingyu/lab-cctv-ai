import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI  # ★ BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.db import get_conn, init_db
from backend.llm import MODEL, PROVIDER, generate_summary
from backend import chat_service, notify

app = FastAPI(title="Lab CCTV Backend")
init_db()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

print(f"[LLM] provider={PROVIDER}, model={MODEL}")  # ★ 어떤 LLM을 쓰는지 서버 시작 시 표시
print(f"[Discord] {'켜짐 (HIGH 이벤트 텍스트 전송)' if notify.is_enabled() else '꺼짐 (.env 의 DISCORD_WEBHOOK_URL 없음)'}")


class EventIn(BaseModel):
    event_id: str
    event_type: Literal["fall_detection", "restricted_area_intrusion", "object_displacement"]
    timestamp: str
    object: str
    location: str
    action: List[str]
    judgment: str
    risk: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float
    duration_seconds: float = 0
    summary: str
    clip_path: Optional[str] = None

class ChatIn(BaseModel):
    question: str


def add_llm_summary(row_id: int, event: dict):  # ★
    """응답을 보낸 뒤 백그라운드에서 실행됨. 실패해도 이벤트 저장에는 영향 없음."""
    text = generate_summary(event)
    if text:
        with get_conn() as conn:
            conn.execute("UPDATE events SET llm_summary = ? WHERE id = ?", (text, row_id))


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": PROVIDER, "llm_model": MODEL}


@app.post("/events")
def create_event(event: EventIn, background_tasks: BackgroundTasks):  # ★
    created_at = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO events
               (event_id, event_type, timestamp, created_at, object, location, action,
                judgment, risk, confidence, duration_seconds, summary, clip_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.event_id, event.event_type, event.timestamp, created_at, event.object,
             event.location, json.dumps(event.action, ensure_ascii=False), event.judgment,
             event.risk, event.confidence, event.duration_seconds, event.summary, event.clip_path),
        )
        row_id = cur.lastrowid

    data = event.model_dump()
    background_tasks.add_task(add_llm_summary, row_id, data)  # ★
    # LLM 요약을 기다리지 않고 따로 보냄 (HIGH일 때만, 텍스트만)
    background_tasks.add_task(notify.notify_high, data)
    return {"id": row_id, "status": "saved"}


@app.get("/events")
def list_events(event_type: Optional[str] = None, limit: int = 50):
    query = "SELECT * FROM events"
    params = []
    if event_type:
        query += " WHERE event_type = ?"
        params.append(event_type)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d["action"] = json.loads(d["action"]) if d["action"] else []
        result.append(d)
    return result

@app.post("/chat")
def chat(req: ChatIn):
    return chat_service.chat(req.question)

    with get_conn() as conn:
        if rel:
            # "최근 1시간" 같은 질문은 자정을 넘을 수 있으므로 날짜가 아니라 시각으로 조회
            start = (now - rel[0]).isoformat(timespec="seconds")
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM events WHERE created_at >= ?
                       ORDER BY id DESC LIMIT 60
                   ) ORDER BY id ASC""",
                (start,),
            ).fetchall()
            scope_label = f"최근 {rel[1]}"
        else:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM events WHERE created_at LIKE ?
                       ORDER BY id DESC LIMIT 60
                   ) ORDER BY id ASC""",
                (day.isoformat() + "%",),
            ).fetchall()
            scope_label = day_label

    events = [dict(r) for r in rows]
    answer = answer_question(req.question, events, day, day_label)
    return {"answer": answer, "event_count": len(events), "scope": scope_label}