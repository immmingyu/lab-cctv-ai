"""이벤트를 Discord로 알리는 선택 기능.

.env 에 DISCORD_WEBHOOK_URL 이 있을 때만 동작한다. 없으면 아무 일도 하지 않는다.
개인정보 보호를 위해 **스냅샷 이미지는 보내지 않고 텍스트만** 보낸다.

알림 대상 (선택, 전시회용):
- DISCORD_NOTIFY_TYPES 가 없거나 비어 있으면 예전과 똑같이 HIGH 이상만 보내고, 보내는 간격 제한도 없다.
- 값이 있으면 적힌 종류만 보내고, 같은 종류는 DISCORD_NOTIFY_COOLDOWN_SECONDS(기본 60초)에 한 번만 보낸다.
  예: DISCORD_NOTIFY_TYPES=fall_high,restricted_area_intrusion,object_displacement
이 파일은 알림만 다룬다. 이벤트 판단, 기준값, 이벤트 저장에는 영향이 없다.
"""
import os
import threading
import time

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

# DISCORD_NOTIFY_TYPES 에 쓸 수 있는 이름
NOTIFY_KINDS = {
    "fall_high": "낙상 HIGH 이상",
    "fall_medium": "낙상 MEDIUM (VLM 낙상 의심 포함)",
    "restricted_area_intrusion": "제한 구역 접근",
    "object_displacement": "물건 이동",
}


def _parse_types(raw):
    """빈 값이면 None(예전 동작). 모르는 이름은 경고만 하고 뺀다."""
    names = [s.strip() for s in (raw or "").split(",") if s.strip()]
    if not names:
        return None
    unknown = [n for n in names if n not in NOTIFY_KINDS]
    if unknown:
        print(f"[Discord] DISCORD_NOTIFY_TYPES 에 모르는 이름이 있어 무시합니다: {', '.join(unknown)} "
              f"(사용 가능: {', '.join(NOTIFY_KINDS)})")
    return [n for n in names if n in NOTIFY_KINDS]


def _parse_cooldown(raw, default=60.0):
    try:
        return max(0.0, float(raw)) if raw not in (None, "") else default
    except ValueError:
        print(f"[Discord] DISCORD_NOTIFY_COOLDOWN_SECONDS 값({raw!r})이 숫자가 아니라 {default:g}초를 씁니다")
        return default


NOTIFY_TYPES = _parse_types(os.getenv("DISCORD_NOTIFY_TYPES"))   # None = 예전 동작
COOLDOWN_SECONDS = _parse_cooldown(os.getenv("DISCORD_NOTIFY_COOLDOWN_SECONDS", "").strip())

_last_sent = {}            # 종류 이름 -> 마지막으로 보낸 시각 (time.monotonic)
_lock = threading.Lock()   # 백그라운드 작업이 동시에 돌 수 있으므로


def is_enabled():
    return bool(WEBHOOK_URL)


def describe():
    """서버 시작 때 보여줄 한 줄 설명."""
    if not is_enabled():
        return "꺼짐 (.env 의 DISCORD_WEBHOOK_URL 없음)"
    if NOTIFY_TYPES is None:
        return "켜짐 (HIGH 이벤트 텍스트 전송)"
    if not NOTIFY_TYPES:
        return "켜짐, 하지만 DISCORD_NOTIFY_TYPES 에 쓸 수 있는 이름이 없어 아무것도 보내지 않음"
    kinds = ", ".join(NOTIFY_KINDS[n] for n in NOTIFY_TYPES)
    return f"켜짐 ({kinds} / 같은 종류는 {COOLDOWN_SECONDS:g}초에 한 번)"


def _kind_of(event):
    """이벤트가 DISCORD_NOTIFY_TYPES 의 어느 이름에 해당하는지."""
    etype = event.get("event_type")
    if etype == "fall_detection":
        if event.get("risk") in ALERT_RISKS:
            return "fall_high"
        if event.get("risk") == "MEDIUM":
            return "fall_medium"
        return None
    if etype in ("restricted_area_intrusion", "object_displacement"):
        return etype
    return None


def build_message(event):
    kind = TYPE_KR.get(event.get("event_type"), event.get("event_type"))
    when = str(event.get("timestamp", "")).replace("T", " ")
    icon = "🚨" if event.get("risk") in ALERT_RISKS else ("⚠️" if event.get("risk") == "MEDIUM" else "ℹ️")
    lines = [
        f"{icon} **{kind}** ({event.get('risk')})",
        f"시각: {when}",
        f"대상: {event.get('object')} / 위치: {event.get('location')}",
        f"지속: {event.get('duration_seconds')}초 / 신뢰도: {event.get('confidence')}",
        f"내용: {event.get('summary')}",
    ]
    return "\n".join(lines)


def _post(event):
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


def notify_event(event):
    """설정에 맞는 이벤트만 전송. 실패해도 이벤트 저장에는 영향이 없다."""
    if not is_enabled():
        return False

    # 설정이 없으면 예전과 똑같이: HIGH 이상만, 간격 제한 없음
    if NOTIFY_TYPES is None:
        if event.get("risk") not in ALERT_RISKS:
            return False
        return _post(event)

    kind = _kind_of(event)
    if kind is None or kind not in NOTIFY_TYPES:
        return False

    # 같은 종류는 COOLDOWN_SECONDS 에 한 번만 (전시회에서 도배 방지)
    now = time.monotonic()
    with _lock:
        prev = _last_sent.get(kind)
        if prev is not None and now - prev < COOLDOWN_SECONDS:
            return False
        _last_sent[kind] = now   # 보내기 전에 자리를 잡아 동시에 두 번 가는 것을 막는다

    ok = _post(event)
    if not ok:
        # 전송이 실패했으면 다음 이벤트는 바로 다시 시도할 수 있게 되돌린다
        with _lock:
            if _last_sent.get(kind) == now:
                if prev is None:
                    _last_sent.pop(kind, None)
                else:
                    _last_sent[kind] = prev
    return ok


# 예전 이름 (호환용)
notify_high = notify_event
