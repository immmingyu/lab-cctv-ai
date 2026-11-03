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

def create_vlm_fall_event(track_id, bbox, still_seconds, reason, confidence=0.7):
    """
    조건부 VLM 낙상 이벤트: 규칙(어깨-골반 각도)은 누움이 아니라고 했지만,
    오래 움직임이 없어 VLM에게 물었더니 '누워 있다'고 답한 경우.
    confidence: VLM은 확신도를 주지 않으므로 규칙(0.9)보다 낮은 고정값.
    """
    now = datetime.now().isoformat(timespec="seconds")

    return {
        "event_id": _next_event_id(),
        "event_type": "fall_detection",
        "timestamp": now,
        "object": f"Person #{track_id:02d}",
        "location": "연구실 Camera Area A",
        "action": ["lying", "no_movement"],
        "judgment": "fall_suspected_vlm",
        "risk": "MEDIUM",
        "confidence": confidence,
        "duration_seconds": still_seconds,
        "summary": f"{still_seconds}초간 움직임이 없어 영상 판단 모델(VLM)로 재확인한 결과 누워 있는 것으로 판단됨 (근거: {reason})",
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

def create_displacement_event(object_class, duration, confidence=0.85, reason="moved"):
    """물건 이동 이벤트. reason: 'moved'(다른 위치에서 확인) 또는 'missing'(원래 위치에서 사라짐)."""
    now = datetime.now().isoformat(timespec="seconds")
    object_name_kr = {"laptop": "노트북", "book": "책"}.get(object_class, object_class)

    if reason == "missing":
        action = ["disappeared_from_original_position"]
        judgment = "object_missing"
        summary = f"{object_name_kr}이(가) 원래 위치에서 {duration}초 이상 보이지 않음 (이동 또는 가려짐)"
    else:
        action = ["moved_from_original_position"]
        judgment = "displacement_confirmed"
        summary = f"{object_name_kr}이(가) 원래 위치에서 {duration}초간 벗어난 상태로 확인됨"

    return {
        "event_id": _next_event_id(),
        "event_type": "object_displacement",
        "timestamp": now,
        "object": object_name_kr,
        "location": "연구실 책상",
        "action": action,
        "judgment": judgment,
        "risk": "LOW",
        "confidence": confidence,
        "duration_seconds": duration,
        "summary": summary,
    }