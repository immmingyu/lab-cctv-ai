import argparse
import csv
from collections import defaultdict
from pathlib import Path

SCORE = {"O": 1.0, "△": 0.5, "X": 0.0}
NAMES = {"grounded": "계산 결과 넣음 (현재 설계)", "ungrounded": "계산 결과 뺌 (LLM이 계산)"}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="LLM 평가 채점 결과 집계")
    parser.add_argument("run_dir", help="run_eval이 만든 폴더 경로")
    parser.add_argument("--sheet", default="grading_sheet.csv", help="채점이 끝난 시트 파일 이름")
    args = parser.parse_args()

    d = Path(args.run_dir)
    sheet = read(d / args.sheet)
    key = {r["번호"]: r for r in read(d / "answer_key.csv")}
    raw = read(d / "raw.csv")

    total = defaultdict(lambda: {"n": 0, "O": 0, "△": 0, "X": 0, "Y": 0, "score": 0.0})
    by_cat = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0]))  # [문항 수, 점수, 지어냄]
    problems = []

    for r in sheet:
        k = key[r["번호"]]
        for side in ("A", "B"):
            cond = k[f"{side} 조건"]
            acc = r[f"{side} 정확(O/△/X)"].strip()
            hal = r[f"{side} 지어냄(Y/N)"].strip().upper()
            if acc not in SCORE or hal not in ("Y", "N"):
                problems.append(f"{r['번호']} {side}: 정확='{acc}', 지어냄='{hal}'")
                continue
            t = total[cond]
            t["n"] += 1
            t[acc] += 1
            t["score"] += SCORE[acc]
            t["Y"] += hal == "Y"
            c = by_cat[cond][r["유형"]]
            c[0] += 1
            c[1] += SCORE[acc]
            c[2] += hal == "Y"

    latency, calls = defaultdict(list), defaultdict(int)
    for r in raw:
        latency[r["조건"]].append(float(r["응답시간(초)"]))
        calls[r["조건"]] += r["LLM호출"] == "Y"

    if problems:
        print("[경고] 채점이 비었거나 형식이 다른 칸:")
        for p in problems:
            print("  ", p)

    print("\n===== 조건별 결과 =====")
    rows = []
    for cond in ("grounded", "ungrounded"):
        t = total[cond]
        n = t["n"] or 1
        avg_lat = sum(latency[cond]) / max(len(latency[cond]), 1)
        print(f"\n{NAMES[cond]}")
        print(f"  정답 O {t['O']} / △ {t['△']} / X {t['X']}  (총 {t['n']}문항)")
        print(f"  점수 {t['score']:.1f}/{t['n']}  ({t['score'] / n * 100:.1f}%, O=1, △=0.5, X=0)")
        print(f"  지어냄 {t['Y']}/{t['n']}  ({t['Y'] / n * 100:.1f}%)")
        print(f"  평균 응답 시간 {avg_lat:.1f}초, LLM 호출 {calls[cond]}/{len(latency[cond])}문항")
        rows.append([NAMES[cond], t["n"], t["O"], t["△"], t["X"],
                     round(t["score"] / n * 100, 1), t["Y"], round(t["Y"] / n * 100, 1),
                     round(avg_lat, 1), calls[cond]])

    print("\n===== 유형별 점수 (%) =====")
    cats = list(dict.fromkeys(r["유형"] for r in sheet))
    print(f"{'유형':<10}{'계산 넣음':>10}{'계산 뺌':>10}")
    for c in cats:
        g, u = by_cat["grounded"][c], by_cat["ungrounded"][c]
        gp = g[1] / g[0] * 100 if g[0] else 0
        up = u[1] / u[0] * 100 if u[0] else 0
        print(f"{c:<10}{gp:>9.0f}%{up:>9.0f}%")

    with open(d / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["조건", "문항", "O", "△", "X", "점수(%)", "지어냄", "지어냄(%)", "평균응답(초)", "LLM호출"])
        w.writerows(rows)
    print(f"\n저장: {d / 'summary.csv'}")


if __name__ == "__main__":
    main()