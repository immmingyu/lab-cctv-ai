"""LLM 답변 검증 장치.

LLM이 만든 문장에서 **시각**과 **건수**를 뽑아, 실제로 LLM에게 넘겨준 이벤트 기록과
맞는지 코드로 확인한다. 맞지 않으면 한 번 다시 생성하고, 그래도 맞지 않으면
코드가 기록만으로 만든 목록 문장으로 대체한다.

(검증은 "지어낸 시각/건수"를 걸러내는 용도이고, 문장이 자연스러운지는 보지 않는다.)
"""
import re
from collections import Counter

# "13시 37분", "13시" (단, "3시간"은 제외)
HM_PATTERN = re.compile(r"(\d{1,2})\s*시(?!간)\s*(?:(\d{1,2})\s*분)?")
# "13:37"
CLOCK_PATTERN = re.compile(r"(\d{1,2})\s*:\s*(\d{2})")
# "3건", "2번", "1개"
COUNT_PATTERN = re.compile(r"(\d+)\s*(?:건|번|개)(?!월|의 달)")
# "한 건", "두 번" 처럼 숫자 대신 쓰는 말
WORD_NUMBERS = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6,
                "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
WORD_COUNT_PATTERN = re.compile(r"(" + "|".join(WORD_NUMBERS) + r")\s*(?:건|번|개)")


def extract_times(text):
    """문장에서 (시, 분)과 (시, None)을 모두 뽑는다. 분이 없으면 None."""
    found = set()
    for h, m in CLOCK_PATTERN.findall(text or ""):
        found.add((int(h), int(m)))
    for h, m in HM_PATTERN.findall(text or ""):
        h = int(h)
        if h > 23:
            continue
        if m and int(m) <= 59:
            found.add((h, int(m)))
        elif not m:
            found.add((h, None))
    return found


def extract_counts(text):
    """문장에서 '~건 / ~번 / ~개' 앞의 숫자를 뽑는다."""
    found = {int(n) for n in COUNT_PATTERN.findall(text or "")}
    found |= {WORD_NUMBERS[w] for w in WORD_COUNT_PATTERN.findall(text or "")}
    return found


def _event_dt(e):
    from backend.llm import _event_dt as f   # 시각 읽는 방법을 한 곳에서만 관리
    return f(e)


def allowed_times(events, question, scope, now):
    """답변에 나와도 되는 시각 = 이벤트 시각 + 질문에 적힌 시각 + 범위 설명의 시각 + 현재 시각."""
    times = set()
    for e in events:
        d = _event_dt(e)
        if d:
            times.add((d.hour, d.minute))
            times.add((d.hour, None))
    times |= extract_times(question or "")
    times |= extract_times(scope or "")
    if now:
        times.add((now.hour, now.minute))
        times.add((now.hour, None))
    hours = {h for h, _ in times}
    return times, hours


def allowed_counts(events):
    """답변에 나와도 되는 건수 = 전체 건수와, 종류별/위험도별/대상별 건수."""
    if not events:
        return {0}
    counts = {len(events)}
    for key in ("event_type", "risk", "object"):
        counts |= set(Counter(e.get(key) for e in events).values())
    return counts


def find_problems(answer, events, question="", scope="", now=None):
    """답변에서 기록과 맞지 않는 시각/건수를 찾아 한국어 설명 목록으로 돌려준다."""
    problems = []
    if not answer:
        return ["답변이 비어 있음"]

    ok_times, ok_hours = allowed_times(events, question, scope, now)
    for h, m in sorted(extract_times(answer), key=lambda x: (x[0], x[1] is None, x[1])):
        if m is None:
            if h not in ok_hours:
                problems.append(f"기록에 없는 시각 {h}시")
        elif (h, m) not in ok_times:
            problems.append(f"기록에 없는 시각 {h}시 {m}분")

    ok_counts = allowed_counts(events)
    for n in sorted(extract_counts(answer)):
        if n not in ok_counts:
            problems.append(f"기록과 다른 건수 {n}건 (실제 {len(events)}건)")

    return problems


def code_answer(events, scope=""):
    """LLM 없이 코드가 기록만으로 만드는 답변 (검증을 두 번 통과하지 못했을 때 사용)."""
    from backend.llm import TYPE_KR
    if not events:
        return f"{scope} 기록된 이벤트가 없습니다.".strip()
    items = []
    for e in events:
        d = _event_dt(e)
        when = f"{d.hour}시 {d.minute}분 {d.second}초" if d else str(e.get("timestamp"))
        kind = TYPE_KR.get(e.get("event_type"), e.get("event_type"))
        items.append(f"{when} {kind}({e.get('risk')})")
    head = f"{scope} 이벤트 {len(events)}건" if scope else f"이벤트 {len(events)}건"
    return f"{head}입니다: " + ", ".join(items) + "."


def verified_answer(generate, question, events, scope="", now=None, info=None, max_tries=2):
    """
    generate: LLM을 호출해 답변 문자열을 돌려주는 함수 (인자 없음)

    1) 생성 -> 검증 통과하면 그대로 사용
    2) 통과 못 하면 한 번 더 생성
    3) 두 번째도 통과 못 하면 코드가 만든 기록 목록으로 대체
    """
    record = {"attempts": 0, "problems": [], "replaced": False}
    last_problems = []

    for _ in range(max_tries):
        record["attempts"] += 1
        try:
            answer = generate()
        except Exception as e:
            print(f"[검증] 답변 생성 실패: {e}")
            last_problems = [f"생성 실패: {e}"]
            continue
        problems = find_problems(answer, events, question, scope, now)
        if not problems:
            record["problems"] = []
            if info is not None:
                info.update(record)
            return answer
        last_problems = problems
        print(f"[검증] 시도 {record['attempts']}: " + "; ".join(problems))

    record["problems"] = last_problems
    record["replaced"] = True
    if info is not None:
        info.update(record)
    return code_answer(events, scope)
