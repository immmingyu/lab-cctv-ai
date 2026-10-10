import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path("data/experiments/smoothing_log.csv")
PNG_PATH = Path("data/experiments/smoothing_compare.png")

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 (Mac에서는 "AppleGothic")
plt.rcParams["axes.unicode_minus"] = False

LEVEL = {"standing": 0, "lying": 1}  # unknown은 값 없음(None)으로 처리


def main():
    t, raw, smooth, base_ev, smooth_ev = [], [], [], [], []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ti = float(row["t"])
            t.append(ti)
            raw.append(LEVEL.get(row["raw"]))
            smooth.append(LEVEL.get(row["smoothed"], 0))
            if row["baseline_event"]:
                base_ev.append((ti, row["baseline_event"]))
            if row["smoothed_event"]:
                smooth_ev.append((ti, row["smoothed_event"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # 위: 스무딩 전 (프레임별 원본 판단)
    raw_plot = [v if v is not None else float("nan") for v in raw]
    ax1.step(t, raw_plot, where="post", color="#ef4444", linewidth=1)
    unknown_t = [ti for ti, v in zip(t, raw) if v is None]
    if unknown_t:
        ax1.scatter(unknown_t, [0.5] * len(unknown_t), s=6, color="gray", label="관절 인식 실패")
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"스무딩 전 (프레임 단위 판단): 이벤트 {len(base_ev)}건", fontsize=11)

    # 아래: 스무딩 후
    ax2.fill_between(t, smooth, step="post", color="#3b82f6", alpha=0.35)
    ax2.step(t, smooth, where="post", color="#1d4ed8", linewidth=1.5)
    ax2.set_title(f"스무딩 후 (최근 15프레임 다수결 + 회복 3초): 이벤트 {len(smooth_ev)}건", fontsize=11)

    # 이벤트 발생 지점 표시 (M = MEDIUM, H = HIGH)
    for ax, events in ((ax1, base_ev), (ax2, smooth_ev)):
        for ti, level in events:
            ax.axvline(ti, color="black", linestyle="--", linewidth=0.7)
            ax.text(ti, 1.18, level[0], ha="center", fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["서 있음", "누움"])
        ax.set_ylim(-0.2, 1.35)
        ax.grid(axis="x", alpha=0.3)

    ax2.set_xlabel("시간 (초)")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300)  # 포스터용 고해상도
    print(f"저장 완료: {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()