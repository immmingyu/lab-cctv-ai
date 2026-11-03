"""HIGH 이벤트를 Discord로 알리는 선택 기능.

.env 에 DISCORD_WEBHOOK_URL 이 있을 때만 동작한다. 없으면 아무 일도 하지 않는다.
개인정보 보호를 위해 **스냅샷 이미지는 보내지 않고 텍스트만** 보낸다.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
TIMEOUT = 5

TYPE_KR = {
    "fall_detection": "낙상 감지",
    "restricted_area_intrusion": "제한 구역 접근",
    "object_displacement": "물건 이동",
}

# 소리까지 울릴 만큼 급한 위험도
ALERT_RISKS = ("HIGH", "CRITICAL")


def is_enabled():
    return bool(WEBHOOK_URL)


def build_message(event):
    kind = TYPE_KR.get(event.get("event_type"), event.get("event_type"))
    when = str(event.get("timestamp", "")).replace("T", " ")
    lines = [
        f"🚨 **{kind}** ({event.get('risk')})",
        f"시각: {when}",
        f"대상: {event.get('object')} / 위치: {event.get('location')}",
        f"지속: {event.get('duration_seconds')}초 / 신뢰도: {event.get('confidence')}",
        f"내용: {event.get('summary')}",
    ]
    return "\n".join(lines)


def notify_high(event):
    """HIGH 이상일 때만 전송. 실패해도 이벤트 저장에는 영향이 없다."""
    if not is_enabled():
        return False
    if event.get("risk") not in ALERT_RISKS:
        return False
    try:
        res = requests.post(
            WEBHOOK_URL,
            json={"content": build_message(event)},   # 이미지 없이 텍스트만
            timeout=TIMEOUT,
        )
        if res.status_code >= 300:
            print(f"[Discord] 전송 실패 ({res.status_code}): {res.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        print(f"[Discord] 전송 실패: {e}")
        return False
