from datetime import datetime

_event_counter = 0

def _next_event_id():
    global _event_counter
    _event_counter += 1
    return f"EVT_{_event_counter:03d}"

def create_fall_event(track_id, bbox, risk, duration, confidence):
    """
    문서 6.9 Event JSON 형식에 맞춘 낙상 이벤트 생성.
    """
    now = datetime.now().isoformat(timespec="seconds")

    if risk == "HIGH":
        summary = f"이동 중 넘어졌으며 이후 {duration}초간 움직임이 없는 상황 (위험)"
        judgment = "fall_confirmed"
    else:  # MEDIUM
        summary = f"쓰러진 것으로 의심되는 자세가 {duration}초간 지속됨 (주의)"
        judgment = "fall_suspected"

    return {
        "event_id": _next_event_id(),
        "event_type": "fall_detection",
        "timestamp": now,
        "object": f"Person #{track_id:02d}",
        "location": "연구실 Camera Area A",
        "action": ["lying", "no_movement"],
        "judgment": judgment,
        "risk": risk,
        "confidence": confidence,
        "duration_seconds": duration,
        "summary": summary,
    }

def create_intrusion_event(track_id, duration, confidence=0.9):
    """
    문서 6.9 Event JSON 형식에 맞춘 제한구역 접근 이벤트 생성.
    """
    now = datetime.now().isoformat(timespec="seconds")

    return {
        "event_id": _next_event_id(),
        "event_type": "restricted_area_intrusion",
        "timestamp": now,
        "object": f"Person #{track_id:02d}",
        "location": "연구실 제한 구역 (ROI)",
        "action": ["entered_restricted_area", "stayed"],
        "judgment": "intrusion_confirmed",
        "risk": "MEDIUM",
        "confidence": confidence,
        "duration_seconds": duration,
        "summary": f"제한 구역에 {duration}초간 머무름이 확인됨",
    }

def create_displacement_event(object_class, duration, confidence=0.85):
    """
    문서 6.9 Event JSON 형식에 맞춘 물건 이동 이벤트 생성.
    """
    now = datetime.now().isoformat(timespec="seconds")

    object_name_kr = {"laptop": "노트북", "book": "책"}.get(object_class, object_class)

    return {
        "event_id": _next_event_id(),
        "event_type": "object_displacement",
        "timestamp": now,
        "object": object_name_kr,
        "location": "연구실 책상",
        "action": ["moved_from_original_position"],
        "judgment": "displacement_confirmed",
        "risk": "LOW",  # 물건 이동 자체는 낙상보다 긴급도가 낮음
        "confidence": confidence,
        "duration_seconds": duration,
        "summary": f"{object_name_kr}이(가) 원래 위치에서 {duration}초간 벗어난 상태로 확인됨",
    }