from datetime import datetime

from backend.db import get_conn
from backend.llm import answer_question, extract_date, extract_relative

SELECT_LIMIT = 60


def select_events(question, now=None, db_path=None):
    """질문에서 날짜나 상대 시간을 읽고, 그 범위의 이벤트를 시간순으로 반환."""
    now = now or datetime.now()
    day, day_label = extract_date(question, now)
    rel = extract_relative(question)

    with get_conn(db_path) as conn:
        if rel:
            start = (now - rel[0]).isoformat(timespec="seconds")
            end = now.isoformat(timespec="seconds")
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM events WHERE created_at >= ? AND created_at <= ?
                       ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (start, end, SELECT_LIMIT),
            ).fetchall()
            scope_label = f"최근 {rel[1]}"
        else:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM events WHERE created_at LIKE ?
                       ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (day.isoformat() + "%", SELECT_LIMIT),
            ).fetchall()
            scope_label = day_label

    return [dict(r) for r in rows], day, day_label, scope_label


def chat(question, now=None, db_path=None, verify=False, verify_info=None):
    """대시보드 질문창과 평가 스크립트가 함께 쓰는 질문 처리.

    verify=True면 LLM 답변의 시각/건수를 기록과 대조하고, 틀리면 한 번 다시 생성한다.
    (평가용 옵션이라 기본값은 꺼짐)
    """
    now = now or datetime.now()
    events, day, day_label, scope_label = select_events(question, now, db_path)
    answer = answer_question(question, events, day, day_label, now=now,
                             verify=verify, verify_info=verify_info)
    return {"answer": answer, "event_count": len(events), "scope": scope_label}