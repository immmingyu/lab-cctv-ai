import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import backend.llm as llm
from backend import chat_service
from backend.db import get_conn
from src.experiments.llm_eval.build_eval_db import EVAL_DB, EVAL_NOW

QUESTIONS = Path(__file__).with_name("questions.json")
OUT_ROOT = Path("data/llm_eval")
CONDITIONS = ("grounded", "ungrounded")   # 계산 결과 넣음 / 뺌

# LLM이 실제로 호출됐는지 세기 위한 장치
# (계산 결과를 넣는 조건은 "기록 없음"이면 LLM 없이 바로 답하기 때문)
_calls = {"n": 0}
_original_call_llm = llm.call_llm


def _counting_call_llm(*args, **kwargs):
    _calls["n"] += 1
    return _original_call_llm(*args, **kwargs)


llm.call_llm = _counting_call_llm


def all_events_until(now):
    """계산 결과를 빼는 조건용: 기준 시각까지의 전체 기록 (날짜 구분 없이)."""
    with get_conn(EVAL_DB) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE created_at <= ? ORDER BY created_at ASC",
            (now.isoformat(timespec="seconds"),),
        ).fetchall()
    return [dict(r) for r in rows]


def run_condition(cond, question, all_events):
    _calls["n"] = 0
    start = time.perf_counter()
    if cond == "grounded":
        answer = chat_service.chat(question, now=EVAL_NOW, db_path=EVAL_DB)["answer"]
    else:
        answer = llm.answer_question_ungrounded(question, all_events, now=EVAL_NOW)
    return answer or "(응답 실패)", round(time.perf_counter() - start, 2), _calls["n"] > 0


def write_csv(path, header, rows):
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않게
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="LLM 질의응답 정량 평가")
    parser.add_argument("--seed", type=int, default=42, help="A/B 순서 섞기용 시드")
    args = parser.parse_args()

    if not EVAL_DB.exists():
        raise SystemExit("평가 DB가 없습니다. 먼저 build_eval_db를 실행하세요.")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    all_events = all_events_until(EVAL_NOW)

    model_tag = f"{llm.PROVIDER}_{llm.MODEL}".replace(":", "-").replace("/", "-")
    out_dir = OUT_ROOT / f"run_{datetime.now():%Y%m%d_%H%M%S}_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[평가 시작] 모델 {llm.PROVIDER}/{llm.MODEL}, 질문 {len(questions)}개 x 조건 2개")
    print("  모델 로딩 중 (첫 응답 시간이 측정에 섞이지 않게 미리 한 번 호출)...")
    _original_call_llm("짧게 답해.", "안녕")

    rng = random.Random(args.seed)
    raw, sheet, key = [], [], []

    for q in questions:
        answers = {}
        for cond in CONDITIONS:
            ans, sec, called = run_condition(cond, q["question"], all_events)
            answers[cond] = ans
            raw.append([q["id"], q["category"], q["question"], q["expected"], cond, ans, sec,
                        "Y" if called else "N"])
            print(f"  {q['id']} {cond:<10} {sec:5.1f}s  LLM {'호출' if called else '생략'}")

        order = list(CONDITIONS)
        rng.shuffle(order)
        sheet.append([q["id"], q["category"], q["question"], q["expected"],
                      answers[order[0]], "", "", answers[order[1]], "", "", ""])
        key.append([q["id"], order[0], order[1]])

    write_csv(out_dir / "raw.csv",
              ["번호", "유형", "질문", "정답", "조건", "답변", "응답시간(초)", "LLM호출"], raw)
    write_csv(out_dir / "grading_sheet.csv",
              ["번호", "유형", "질문", "정답", "답변 A", "A 정확(O/△/X)", "A 지어냄(Y/N)",
               "답변 B", "B 정확(O/△/X)", "B 지어냄(Y/N)", "메모"], sheet)
    write_csv(out_dir / "answer_key.csv", ["번호", "A 조건", "B 조건"], key)

    print(f"\n[완료] {out_dir}")
    print("  채점자에게는 grading_sheet.csv만 전달하세요. (answer_key.csv, raw.csv는 채점 후 공개)")


if __name__ == "__main__":
    main()