import json
from datetime import datetime
from pathlib import Path

from backend.db import get_conn, init_db

EVAL_DB = Path("data/llm_eval/eval_events.db")
EVAL_NOW = datetime(2026, 11, 3, 16, 0, 0)   # 평가 기준 "지금" (모든 질문은 이 시각 기준)

# (시각, 종류, 대상, 행동, 판단, 위험도, 지속시간, 요약)
EVENTS = [
    ("2026-11-01T09:05:20", "restricted_area_intrusion", "Person #01", ["entered_restricted_area", "stayed"],
     "intrusion_confirmed", "MEDIUM", 5.0, "제한 구역에 5.0초간 머무름이 확인됨"),
    ("2026-11-02T10:12:40", "restricted_area_intrusion", "Person #02", ["entered_restricted_area", "stayed"],
     "intrusion_confirmed", "MEDIUM", 5.3, "제한 구역에 5.3초간 머무름이 확인됨"),
    ("2026-11-02T13:37:35", "fall_detection", "Person #01", ["lying", "no_movement"],
     "fall_suspected", "MEDIUM", 5.0, "쓰러진 것으로 의심되는 자세가 5.0초간 지속됨 (주의)"),
    ("2026-11-02T13:37:45", "fall_detection", "Person #01", ["lying", "no_movement"],
     "fall_confirmed", "HIGH", 15.1, "이동 중 넘어졌으며 이후 15.1초간 움직임이 없는 상황 (위험)"),
    ("2026-11-02T15:20:10", "object_displacement", "책", ["moved_from_original_position"],
     "displacement_confirmed", "LOW", 2.0, "책이(가) 원래 위치에서 2.0초간 벗어난 상태로 확인됨"),
    ("2026-11-03T09:30:05", "object_displacement", "노트북", ["disappeared_from_original_position"],
     "object_missing", "LOW", 4.0, "노트북이(가) 원래 위치에서 4.0초 이상 보이지 않음 (이동 또는 가려짐)"),
    ("2026-11-03T11:02:30", "restricted_area_intrusion", "Person #02", ["entered_restricted_area", "stayed"],
     "intrusion_confirmed", "MEDIUM", 6.1, "제한 구역에 6.1초간 머무름이 확인됨"),
    ("2026-11-03T13:10:00", "fall_detection", "Person #03", ["lying", "no_movement"],
     "fall_suspected", "MEDIUM", 5.0, "쓰러진 것으로 의심되는 자세가 5.0초간 지속됨 (주의)"),
    ("2026-11-03T13:10:10", "fall_detection", "Person #03", ["lying", "no_movement"],
     "fall_confirmed", "HIGH", 15.0, "이동 중 넘어졌으며 이후 15.0초간 움직임이 없는 상황 (위험)"),
    ("2026-11-03T14:48:20", "restricted_area_intrusion", "Person #01", ["entered_restricted_area", "stayed"],
     "intrusion_confirmed", "MEDIUM", 5.0, "제한 구역에 5.0초간 머무름이 확인됨"),
    ("2026-11-03T15:40:00", "object_displacement", "책", ["moved_from_original_position"],
     "displacement_confirmed", "LOW", 2.1, "책이(가) 원래 위치에서 2.1초간 벗어난 상태로 확인됨"),
]

LOCATIONS = {
    "restricted_area_intrusion": "연구실 제한 구역 (ROI)",
    "fall_detection": "연구실 Camera Area A",
    "object_displacement": "연구실 책상",
}


def main():
    if EVAL_DB.exists():
        EVAL_DB.unlink()          # 항상 같은 상태에서 시작
    init_db(EVAL_DB)
    with get_conn(EVAL_DB) as conn:
        for i, (ts, etype, obj, action, judgment, risk, dur, summary) in enumerate(EVENTS, 1):
            conn.execute(
                """INSERT INTO events
                   (event_id, event_type, timestamp, created_at, object, location, action,
                    judgment, risk, confidence, duration_seconds, summary, clip_path, llm_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (f"EVAL_{i:03d}", etype, ts, ts, obj, LOCATIONS[etype],
                 json.dumps(action, ensure_ascii=False), judgment, risk, 0.9, dur, summary),
            )
    print(f"평가용 DB 생성: {EVAL_DB} (이벤트 {len(EVENTS)}개, 기준 시각 {EVAL_NOW})")


if __name__ == "__main__":
    main()