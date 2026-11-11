import json
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "exaone3.5:7.8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

TYPE_KR = {
    "fall_detection": "낙상 감지",
    "restricted_area_intrusion": "제한 구역 접근",
    "object_displacement": "물건 이동",
}

SUMMARY_SYSTEM_PROMPT = """너는 연구실 CCTV 관제 시스템의 상황 설명 담당이다.
주어진 이벤트 데이터만 근거로 관리자에게 보낼 한국어 설명을 2~3문장으로 작성한다.
규칙:
- 데이터에 없는 사실(사람의 신원, 의도, 부상 정도 등)은 절대 추측하지 않는다.
- 첫 문장: 언제, 어디서, 무슨 일이 감지되었는지.
- 둘째 문장: 그렇게 판단한 근거(감지된 행동, 지속 시간 등).
- 위험도가 HIGH이면 마지막에 필요한 조치를 짧게 권고한다.
- 인사말, 머리말, 목록 기호 없이 문장만 출력한다."""

CHAT_SYSTEM_PROMPT = """너는 연구실 CCTV 관제 시스템의 질의응답 담당이다.
사용자 메시지에 포함된 이벤트 기록만 근거로 질문에 한국어로 답한다.
규칙:
- 기록에 없는 일은 '해당하는 기록이 없습니다'라고 답하고, 절대 지어내지 않는다.
- [시스템 계산 결과]가 주어지면 그 범위와 건수를 그대로 따르고, 직접 다시 계산하지 않는다.
- 시각과 숫자는 기록에 적힌 값만 사용한다.
- 같은 종류의 이벤트가 여러 건이면 하나씩 나열하지 말고 묶어서 요약한다.
  (예: 13시 37분부터 13시 46분 사이에 낙상 이벤트 8건)
- 사람의 신원이나 의도는 추측하지 않는다.
- 마크다운 기호(**, -, # 등)나 목록을 쓰지 않고 일반 문장으로 3~5문장 이내로 답한다."""

# ★ 기록의 시각을 "시각 11:02:30"처럼 넘길 때 덧붙이는 규칙
#   (예전 "[11시 2분 30초] ... 지속 6.1초" 형식에서 LLM이 30초를 머문 시간으로 읽은 문제)
TIME_FORMAT_RULE = """
- 기록의 '시각'은 24시간 HH:MM:SS 형식이다 (예: 11:02:30 = 11시 2분 30초). 시각의 초를 머문 시간이나 지속 시간으로 읽지 않는다.
- 머문 시간, 지속 시간은 기록의 '머문 시간', '자세 지속 시간', '지속 시간' 값만 사용한다.
- 답할 때 시각은 '11시 2분'처럼 자연스러운 한국어로 쓰고, 같은 분에 여러 건이 있어 구분해야 할 때만 초까지 쓴다."""

# ★ True면 예전 형식 "[11시 2분 30초] ... 지속 6.1초" (2026-11-12부터 실제 시스템 기본값, 전시회용)
#   24시간제 "시각 13:10:00" 형식에서 LLM이 "1시쯤", "3시 20분쯤" 같은 질문을 13시, 15시 기록과
#   맞추지 못하고 "기록 없음"이라고 답하는 실수가 나와서 되돌림 (docs/DEV_NOTES.md '변경된 설정')
LEGACY_TIME_FORMAT = True

DURATION_LABEL = {
    "restricted_area_intrusion": "머문 시간",
    "fall_detection": "자세 지속 시간",
    "object_displacement": "지속 시간",
}


# ---------- 공통: 시각 포맷 ----------
def _parse_dt(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _format_time(ts):
    d = _parse_dt(ts)
    return f"{d.month}월 {d.day}일 {d.hour}시 {d.minute}분 {d.second}초" if d else ts


def _event_time(e):
    """timestamp에 날짜가 없던 예전 데이터는 서버 수신 시각(created_at)으로 대체."""
    d = _parse_dt(e.get("timestamp")) or _parse_dt(e.get("created_at"))
    return f"{d.hour}시 {d.minute}분 {d.second}초" if d else str(e.get("timestamp"))


# ---------- 공통: LLM 호출 ----------
def _call_ollama(system, user):
    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 8192,   # 이벤트 목록이 길어도 잘리지 않게 입력 길이 한도를 늘림
            },
            "keep_alive": "30m",
        },
        timeout=120,
    )
    res.raise_for_status()
    return res.json()["message"]["content"].strip()


def _call_anthropic(system, user, max_tokens):
    import anthropic  # Claude로 전환했을 때만 필요
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def call_llm(system, user, max_tokens=300):
    if PROVIDER == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    return _call_ollama(system, user)


# ---------- 1) 이벤트 요약 ----------
def build_summary_prompt(event):
    facts = {
        "이벤트 종류": TYPE_KR.get(event.get("event_type"), event.get("event_type")),
        "감지 시각": _format_time(event.get("timestamp")),
        "대상": event.get("object"),
        "위치": event.get("location"),
        "감지된 행동": event.get("action"),
        "판단": event.get("judgment"),
        "위험도": event.get("risk"),
        "신뢰도": event.get("confidence"),
        "지속 시간(초)": event.get("duration_seconds"),
        "규칙 기반 요약": event.get("summary"),
    }
    return "다음 이벤트를 설명해줘:\n" + json.dumps(facts, ensure_ascii=False, indent=2)


def generate_summary(event):
    """실패하면 None을 반환. 이 경우 대시보드에는 규칙 기반 summary만 표시됨."""
    try:
        return call_llm(SUMMARY_SYSTEM_PROMPT, build_summary_prompt(event))
    except Exception as e:
        print(f"[LLM] 요약 생성 실패 ({PROVIDER}/{MODEL}): {e}")
        return None


# ---------- 2) 자연어 질문 ----------
TIME_PATTERN = re.compile(r"(오전|오후|아침|저녁|밤|새벽)?\s*(\d{1,2})\s*시(?!간)\s*(?:(\d{1,2})\s*분|(반))?")
CLOCK_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")
RELATIVE_PATTERN = re.compile(r"(\d+)\s*(시간|분)\s*(?:동안|이내|사이|전부터|전)")
MONTH_DAY_PATTERN = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
DAYS_AGO_PATTERN = re.compile(r"(\d+)\s*일\s*전")


def _event_dt(e):
    return _parse_dt(e.get("timestamp")) or _parse_dt(e.get("created_at"))


def _event_dt_text(e):
    """★ 날짜까지 포함한 시각 표시 (계산 결과를 빼는 비교 조건에서 사용)."""
    d = _event_dt(e)
    return f"{d.month}월 {d.day}일 {d.hour}시 {d.minute}분 {d.second}초" if d else str(e.get("timestamp"))


def _kind(e):
    return TYPE_KR.get(e.get("event_type"), e.get("event_type"))


def extract_date(question, now):
    """질문에서 날짜를 찾아 (날짜, 표시용 이름)을 반환. 없으면 오늘."""
    today = now.date()

    m = MONTH_DAY_PATTERN.search(question)
    if m:
        try:
            d = date(today.year, int(m.group(1)), int(m.group(2)))
            if d > today:                      # 미래 날짜면 작년으로 해석
                d = d.replace(year=d.year - 1)
            return d, f"{d.month}월 {d.day}일"
        except ValueError:
            pass                               # "13월 40일" 같은 잘못된 날짜는 무시

    m = DAYS_AGO_PATTERN.search(question)
    if m:
        n = int(m.group(1))
        d = today - timedelta(days=n)
        return d, f"{n}일 전({d.month}월 {d.day}일)"

    if "그저께" in question or "그제" in question:
        d = today - timedelta(days=2)
        return d, f"그저께({d.month}월 {d.day}일)"
    if "어제" in question:
        d = today - timedelta(days=1)
        return d, f"어제({d.month}월 {d.day}일)"

    return today, "오늘"


def extract_relative(question):
    """'1시간 동안' -> (timedelta(hours=1), '1시간'). 없으면 None."""
    m = RELATIVE_PATTERN.search(question)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    delta = timedelta(hours=n) if unit == "시간" else timedelta(minutes=n)
    return delta, f"{n}{unit}"


def extract_time(question, now, day):
    """질문에서 시각을 찾아 (기준 시각, 허용 범위(분))을 반환. 날짜는 day 기준."""
    m = CLOCK_PATTERN.search(question)
    if m:
        period, h, mi, window = None, int(m.group(1)), int(m.group(2)), 15
    else:
        m = TIME_PATTERN.search(question)
        if not m:
            return None
        period, h = m.group(1), int(m.group(2))
        if m.group(3):
            mi, window = int(m.group(3)), 15
        elif m.group(4):
            mi, window = 30, 15
        else:
            mi, window = 0, 30

    if h > 23 or mi > 59:
        return None

    base = datetime.combine(day, datetime.min.time())  # 오늘이 아니라 질문한 날짜 기준

    if period in ("오후", "저녁", "밤") and h < 12:
        h += 12
    elif period in ("오전", "아침", "새벽") and h == 12:
        h = 0
    elif period is None and h < 12:
        am = base.replace(hour=h, minute=mi)
        pm = am + timedelta(hours=12)
        return (pm if pm <= now else am), window

    return base.replace(hour=h, minute=mi), window


def _event_line(i, e, with_date=False, legacy=None):
    """
    ★ with_date=True면 날짜까지 표시.
    ★ 시각은 "시각 11:02:30", 지속 시간은 "머문 시간 6.1초"처럼 이름을 붙여 헷갈리지 않게 함.
      legacy=True(또는 LEGACY_TIME_FORMAT)면 예전 "[11시 2분 30초] ... 지속 6.1초" 형식.
    """
    legacy = LEGACY_TIME_FORMAT if legacy is None else legacy
    desc = e.get("llm_summary") or e.get("summary")
    if legacy:
        when = _event_dt_text(e) if with_date else _event_time(e)
        return (f"{i}. [{when}] {_kind(e)} / 대상: {e.get('object')} / "
                f"위험도: {e.get('risk')} / 지속 {e.get('duration_seconds')}초 / 내용: {desc}")
    d = _event_dt(e)
    clock = d.strftime("%H:%M:%S") if d else str(e.get("timestamp"))
    when = f"{d.month}월 {d.day}일 {clock}" if (with_date and d) else clock
    label = DURATION_LABEL.get(e.get("event_type"), "지속 시간")
    return (f"{i}. 시각 {when} / {_kind(e)} / 대상: {e.get('object')} / "
            f"위험도: {e.get('risk')} / {label} {e.get('duration_seconds')}초 / 내용: {desc}")


def _chat_system_prompt(legacy=None):
    legacy = LEGACY_TIME_FORMAT if legacy is None else legacy
    return CHAT_SYSTEM_PROMPT if legacy else CHAT_SYSTEM_PROMPT + TIME_FORMAT_RULE


def _now_text(now, legacy=None):
    legacy = LEGACY_TIME_FORMAT if legacy is None else legacy
    if legacy:
        return f"{now.month}월 {now.day}일 {now.hour}시 {now.minute}분"
    return f"{now.month}월 {now.day}일 {now:%H:%M}"


def _count_text(events):
    counts = Counter(_kind(e) for e in events)
    return ", ".join(f"{k} {v}건" for k, v in counts.items())


def _fallback_answer(events):
    items = [f"{_event_time(e)} {_kind(e)}({e.get('risk')})" for e in events]
    return (f"AI 응답을 생성하지 못해 기록만 보여드립니다. {len(events)}건의 이벤트가 있었습니다: "
            + ", ".join(items))


def narrow_events(question, events, day, day_label, now):
    """
    질문의 시간 범위를 코드로 좁힌다.
    반환: (대상 이벤트, 범위 설명, LLM 없이 바로 답할 문장 또는 None)
    """
    rel = extract_relative(question)

    if not events:
        if rel:
            return [], f"최근 {rel[1]}", f"최근 {rel[1]} 동안 기록된 이벤트가 없습니다."
        return [], day_label, f"{day_label} 기록된 이벤트가 없습니다."

    if rel:
        delta, text = rel
        start = now - delta
        scope = f"최근 {text}({start.hour}시 {start.minute}분 ~ {now.hour}시 {now.minute}분)"
        target_events = [e for e in events if _event_dt(e) and start <= _event_dt(e) <= now]
        if not target_events:
            return [], scope, f"{scope} 동안 기록된 이벤트는 없습니다."
        return target_events, scope, None

    found = extract_time(question, now, day)
    if found:
        target, window = found
        scope = f"{day_label} {target.hour}시 {target.minute}분 전후 {window}분"
        target_events = [e for e in events
                         if _event_dt(e) and abs((_event_dt(e) - target).total_seconds()) <= window * 60]
        if not target_events:
            dated = [e for e in events if _event_dt(e)]
            if not dated:
                return [], scope, f"{scope} 동안 기록된 이벤트는 없습니다."
            nearest = min(dated, key=lambda e: abs((_event_dt(e) - target).total_seconds()))
            return [], scope, (f"{scope} 동안 기록된 이벤트는 없습니다. "
                               f"{day_label} 기록 중 가장 가까운 것은 "
                               f"{_event_time(nearest)}의 {_kind(nearest)} 이벤트입니다.")
        return target_events, scope, None

    return events, day_label, None


def build_chat_prompt(question, target_events, scope, now):
    lines = "\n".join(_event_line(i, e) for i, e in enumerate(target_events, 1))
    return (f"현재 시각: {_now_text(now)}\n\n"
            f"이벤트 기록 (시간순):\n{lines}\n\n"
            f"[시스템 계산 결과] 질문 범위: {scope}. "
            f"이 범위의 이벤트는 총 {len(target_events)}건 ({_count_text(target_events)}).\n\n"
            f"질문: {question}")


def answer_question(question, events, day=None, day_label="오늘", now=None,
                    verify=False, verify_info=None):
    """
    events: 날짜(또는 최근 N시간) 기준으로 이미 골라서 넘겨준 기록.
    ★ now: 기준 시각. 안 넘기면 현재 시각, 평가 때는 고정된 시각을 넘김.
    ★ verify: True면 답변 속 시각/건수를 기록과 대조하고, 틀리면 한 번 다시 생성한다.
      verify_info: 검증 과정(시도 횟수, 발견된 문제, 대체 여부)을 받아갈 dict.
    """
    now = now or datetime.now()
    day = day or now.date()

    target_events, scope, direct = narrow_events(question, events, day, day_label, now)
    if direct is not None:
        # 코드가 바로 답하는 경우라 검증할 것이 없음
        if verify_info is not None:
            verify_info.update({"attempts": 0, "problems": [], "replaced": False})
        return direct

    user = build_chat_prompt(question, target_events, scope, now)

    def generate():
        return call_llm(_chat_system_prompt(), user, max_tokens=500)

    if verify:
        from backend import verify as verifier   # 순환 import 방지를 위해 여기서 불러옴
        return verifier.verified_answer(generate, question, target_events, scope, now,
                                        info=verify_info)

    try:
        return generate()
    except Exception as e:
        print(f"[LLM] 질문 응답 실패 ({PROVIDER}/{MODEL}): {e}")
        return _fallback_answer(target_events)


# ---------- 3) 평가용 비교 조건 ----------
def answer_question_ungrounded(question, events, now=None):
    """
    ★ 평가용 비교 조건: 코드가 날짜, 시간 범위, 건수를 계산하지 않고
    전체 기록을 날짜와 함께 LLM에게 그대로 넘겨서 LLM이 알아서 답하게 함.
    (실제 시스템에서는 사용하지 않음)
    """
    now = now or datetime.now()
    if not events:
        return "기록된 이벤트가 없습니다."
    # 지난 실험(계산 결과 넣음 vs 뺌)을 그대로 재현할 수 있게 이 비교 조건은 예전 형식 유지
    lines = "\n".join(_event_line(i, e, with_date=True, legacy=True) for i, e in enumerate(events, 1))
    user = (f"현재 시각: {now.year}년 {now.month}월 {now.day}일 {now.hour}시 {now.minute}분\n\n"
            f"이벤트 기록 (시간순):\n{lines}\n\n"
            f"질문: {question}")
    try:
        return call_llm(CHAT_SYSTEM_PROMPT, user, max_tokens=500)
    except Exception as e:
        print(f"[LLM] 질문 응답 실패 ({PROVIDER}/{MODEL}): {e}")
        return None