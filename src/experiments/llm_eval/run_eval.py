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
CONDITIONS = ("grounded", "ungrounded")           # 계산 결과 넣음 / 뺌
VERIFY_CONDITIONS = ("plain", "verified")         # --verify: 검증 없음 / 검증 있음
TIME_CONDITIONS = ("time_old", "time_new")        # --time-format: 시각 표시 예전 / 개선

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
    """조건 하나를 실행하고 (답변, 걸린 시간, LLM 호출 여부, 검증 기록)을 돌려준다."""
    _calls["n"] = 0
    info = {"attempts": 0, "problems": [], "replaced": False}
    start = time.perf_counter()
    if cond == "ungrounded":
        answer = llm.answer_question_ungrounded(question, all_events, now=EVAL_NOW)
    elif cond in TIME_CONDITIONS:
        # 실제 시스템과 같은 경로(계산 결과 넣음, 검증 없음)에서 시각 표시 형식만 바꿔 비교
        llm.LEGACY_TIME_FORMAT = (cond == "time_old")
        try:
            answer = chat_service.chat(question, now=EVAL_NOW, db_path=EVAL_DB)["answer"]
        finally:
            llm.LEGACY_TIME_FORMAT = False
    else:
        # grounded / plain: 검증 없음, verified: 검증 있음
        answer = chat_service.chat(question, now=EVAL_NOW, db_path=EVAL_DB,
                                   verify=(cond == "verified"), verify_info=info)["answer"]
    return answer or "(응답 실패)", round(time.perf_counter() - start, 2), _calls["n"] > 0, info


def write_csv(path, header, rows):
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않게
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="LLM 질의응답 정량 평가")
    parser.add_argument("--seed", type=int, default=42, help="A/B 순서 섞기용 시드")
    parser.add_argument("--questions", nargs="+", default=[str(QUESTIONS)],
                        help="질문 파일 경로, 여러 개면 한 시트로 합침 (기본: questions.json)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true",
                       help="'검증 없음 vs 검증 있음'을 비교 (기본은 '계산 결과 넣음 vs 뺌')")
    group.add_argument("--time-format", action="store_true",
                       help="'시각 표시 예전 vs 개선'을 비교 (예: [11시 2분 30초] vs 시각 11:02:30)")
    args = parser.parse_args()

    if args.verify:
        conditions, tag = VERIFY_CONDITIONS, "verify"
    elif args.time_format:
        conditions, tag = TIME_CONDITIONS, "timefmt"
    else:
        conditions, tag = CONDITIONS, "grounding"

    if not EVAL_DB.exists():
        raise SystemExit("평가 DB가 없습니다. 먼저 build_eval_db를 실행하세요.")

    q_paths = [Path(p) for p in args.questions]
    questions = []
    for q_path in q_paths:
        if not q_path.exists():
            raise SystemExit(f"질문 파일이 없습니다: {q_path}")
        questions += json.loads(q_path.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit("질문 파일들 사이에 같은 번호가 있습니다.")
    all_events = all_events_until(EVAL_NOW)

    model_tag = f"{llm.PROVIDER}_{llm.MODEL}".replace(":", "-").replace("/", "-")
    out_dir = OUT_ROOT / f"run_{datetime.now():%Y%m%d_%H%M%S}_{tag}_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[평가 시작] 모델 {llm.PROVIDER}/{llm.MODEL}, 질문 파일 {', '.join(p.name for p in q_paths)}, "
          f"질문 {len(questions)}개 x 조건 {len(conditions)}개 ({' vs '.join(conditions)})")
    print("  모델 로딩 중 (첫 응답 시간이 측정에 섞이지 않게 미리 한 번 호출)...")
    _original_call_llm("짧게 답해.", "안녕")

    rng = random.Random(args.seed)
    raw, sheet, key = [], [], []

    for q in questions:
        answers = {}
        for cond in conditions:
            ans, sec, called, info = run_condition(cond, q["question"], all_events)
            answers[cond] = ans
            raw.append([q["id"], q["category"], q["question"], q["expected"], cond, ans, sec,
                        "Y" if called else "N",
                        info["attempts"], "; ".join(info["problems"]),
                        "Y" if info["replaced"] else "N"])
            mark = ""
            if cond == "verified":
                mark = f"  검증 {info['attempts']}회" + ("  -> 기록 목록으로 대체" if info["replaced"]
                                                     else ("  -> 재생성 후 통과" if info["attempts"] > 1 else ""))
            print(f"  {q['id']} {cond:<10} {sec:5.1f}s  LLM {'호출' if called else '생략'}{mark}")

        order = list(conditions)
        rng.shuffle(order)
        sheet.append([q["id"], q["category"], q["question"], q["expected"],
                      answers[order[0]], "", "", answers[order[1]], "", "", ""])
        key.append([q["id"], order[0], order[1]])

    write_csv(out_dir / "raw.csv",
              ["번호", "유형", "질문", "정답", "조건", "답변", "응답시간(초)", "LLM호출",
               "검증 시도", "검증이 찾은 문제", "기록 목록으로 대체"], raw)
    write_csv(out_dir / "grading_sheet.csv",
              ["번호", "유형", "질문", "정답", "답변 A", "A 정확(O/△/X)", "A 지어냄(Y/N)",
               "답변 B", "B 정확(O/△/X)", "B 지어냄(Y/N)", "메모"], sheet)
    write_csv(out_dir / "answer_key.csv", ["번호", "A 조건", "B 조건"], key)

    print(f"\n[완료] {out_dir}")
    print("  채점자에게는 grading_sheet.csv만 전달하세요. (answer_key.csv, raw.csv는 채점 후 공개)")


if __name__ == "__main__":
    main()