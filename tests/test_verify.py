"""검증 장치(backend/verify.py) 확인. LLM 없이 가짜 답변으로 테스트한다.

실행: python -m tests.test_verify   (또는 pytest tests/test_verify.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import verify

EVENTS = [
    {"timestamp": "2026-11-03T13:10:00", "event_type": "fall_detection",
     "object": "Person #03", "risk": "MEDIUM", "duration_seconds": 5.0,
     "summary": "쓰러진 것으로 의심되는 자세가 5.0초간 지속됨"},
    {"timestamp": "2026-11-03T13:10:10", "event_type": "fall_detection",
     "object": "Person #03", "risk": "HIGH", "duration_seconds": 15.0,
     "summary": "15.0초간 움직임이 없는 상황"},
    {"timestamp": "2026-11-03T14:48:20", "event_type": "restricted_area_intrusion",
     "object": "Person #01", "risk": "MEDIUM", "duration_seconds": 5.0,
     "summary": "제한 구역에 5.0초간 머무름"},
]
QUESTION = "오늘 무슨 일 있었어?"
SCOPE = "오늘"


def test_뽑아내기():
    assert (13, 10) in verify.extract_times("13시 10분에 낙상이 있었습니다.")
    assert (13, None) in verify.extract_times("13시쯤 낙상이 있었습니다.")
    assert (15, 40) in verify.extract_times("15:40 에 책이 움직였습니다.")
    # "3시간"은 시각이 아니다
    assert verify.extract_times("최근 3시간 동안") == set()
    assert verify.extract_counts("모두 3건입니다") == {3}
    assert verify.extract_counts("두 번 있었습니다") == {2}


def test_맞는_답변은_통과():
    good = "오늘 13시 10분에 낙상이 감지됐고, 14시 48분에 제한 구역 접근이 있었습니다. 모두 3건입니다."
    assert verify.find_problems(good, EVENTS, QUESTION, SCOPE) == []


def test_지어낸_시각을_잡는다():
    bad = "오늘 9시 25분에 낙상이 감지됐습니다."
    problems = verify.find_problems(bad, EVENTS, QUESTION, SCOPE)
    assert any("9시 25분" in p for p in problems), problems


def test_틀린_건수를_잡는다():
    bad = "오늘 13시 10분 낙상 등 모두 8건이 있었습니다."
    problems = verify.find_problems(bad, EVENTS, QUESTION, SCOPE)
    assert any("8건" in p for p in problems), problems


def test_종류별_건수는_통과():
    # 낙상 2건 / 제한 구역 1건 처럼 종류별로 센 숫자도 맞는 것으로 본다
    ok = "오늘 낙상 2건과 제한 구역 접근 1건이 있었습니다."
    assert verify.find_problems(ok, EVENTS, QUESTION, SCOPE) == []


def test_질문에_나온_시각은_통과():
    q = "오늘 오전 7시쯤 무슨 일 있었어?"
    ok = "7시 전후로 기록된 이벤트는 없습니다."
    assert verify.find_problems(ok, EVENTS, q, SCOPE) == []


def test_한_번_다시_생성하고_통과하면_그_답을_쓴다():
    answers = iter([
        "오늘 9시 25분에 낙상이 있었습니다.",            # 지어낸 시각 -> 실패
        "오늘 13시 10분에 낙상이 있었습니다.",           # 통과
    ])
    info = {}
    out = verify.verified_answer(lambda: next(answers), QUESTION, EVENTS, SCOPE, info=info)
    assert out == "오늘 13시 10분에 낙상이 있었습니다."
    assert info["attempts"] == 2 and info["replaced"] is False


def test_두_번_다_틀리면_기록_목록으로_대체():
    info = {}
    out = verify.verified_answer(lambda: "오늘 9시 25분에 낙상 8건이 있었습니다.",
                                 QUESTION, EVENTS, SCOPE, info=info)
    assert info["replaced"] is True and info["attempts"] == 2
    assert "3건" in out and "13시 10분 0초" in out
    assert verify.find_problems(out, EVENTS, QUESTION, SCOPE) == []   # 대체 답변 자체는 항상 맞아야 함


def test_생성_실패해도_기록_목록으로_대체():
    def boom():
        raise RuntimeError("Ollama 꺼짐")
    info = {}
    out = verify.verified_answer(boom, QUESTION, EVENTS, SCOPE, info=info)
    assert info["replaced"] is True and "3건" in out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  통과: {t.__name__}")
    print(f"\n{len(tests)}개 모두 통과")
