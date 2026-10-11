import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("data/experiments/posture_log.csv")
OUT_DIR = Path("data/experiments")

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac에서는 "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROWS = [("stand", "서기"), ("sit", "앉기"), ("bend", "숙이기"),
        ("lie_side", "눕기 (옆)"), ("lie_front", "눕기 (카메라 방향)")]
COLS = [("standing", "서 있음"), ("lying", "누움"), ("other", "판단 불가")]
METHODS = [("bbox_pred", "박스 비율 (기존)"), ("pose_pred", "Pose 각도"),
           ("smooth_pred", "Pose + 스무딩 (현재)")]


def load():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col_of(pred):
    return pred if pred in ("standing", "lying") else "other"


def binary_metrics(gt, pred):
    """누움(lying)을 양성으로 본 정확도, 정밀도, 재현율, F1.
    '판단 불가'는 시스템이 경보를 내지 않는 것과 같으므로 '누움 아님'으로 처리."""
    gt, pred = np.array(gt), np.array(pred)
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt)
    acc = np.mean(pred == gt)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1


def plot_confusion(rows):
    present = [(k, n) for k, n in ROWS if any(r["label"] == k for r in rows)]
    gt = [r["label"].startswith("lie") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    print("\n방식별 성능 (누움 = 양성)")
    for ax, (key, name) in zip(axes, METHODS):
        mat = np.zeros((len(present), len(COLS)))
        for i, (lk, _) in enumerate(present):
            sub = [r for r in rows if r["label"] == lk]
            for j, (ck, _) in enumerate(COLS):
                mat[i, j] = sum(col_of(r[key]) == ck for r in sub) / len(sub) * 100

        ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=11,
                        color="white" if mat[i, j] > 60 else "black")

        acc, prec, rec, f1 = binary_metrics(gt, [r[key] == "lying" for r in rows])
        print(f"  {name:<18} 정확도 {acc*100:5.1f}%  정밀도 {prec*100:5.1f}%  "
              f"재현율 {rec*100:5.1f}%  F1 {f1:.3f}")
        ax.set_title(f"{name}\n정확도 {acc*100:.1f}%  ·  F1 {f1:.2f}", fontsize=11)
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels([n for _, n in COLS])
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([n for _, n in present])
        ax.set_xlabel("판단 결과")
    axes[0].set_ylabel("실제 자세")
    fig.suptitle(f"자세 판단 방식별 혼동행렬 (총 {len(rows)} 프레임)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=300)
    print(f"저장: {out}")


def plot_threshold(rows):
    valid = [r for r in rows if r["angle"]]
    angles = np.array([float(r["angle"]) for r in valid])
    gt = [r["label"].startswith("lie") for r in valid]

    ths = np.arange(10, 81, 5)
    accs, f1s = [], []
    for th in ths:
        acc, _, _, f1 = binary_metrics(gt, list(angles < th))
        accs.append(acc * 100)
        f1s.append(f1 * 100)
    best = int(ths[int(np.argmax(f1s))])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ths, accs, marker="o", color="#3b82f6", label="정확도")
    ax.plot(ths, f1s, marker="s", color="#f59e0b", label="F1 점수 (누움)")
    ax.axvline(45, color="#ef4444", linestyle="--", linewidth=1.5, label="현재 기준 45°")
    ax.scatter([best], [max(f1s)], s=120, facecolors="none", edgecolors="black", zorder=5)
    ax.annotate(f"최고 F1: {best}°", (best, max(f1s)), textcoords="offset points",
                xytext=(10, -18), fontsize=10)
    ax.set_xlabel("누움 판단 기준 각도 (어깨-골반 선, 도)")
    ax.set_ylabel("점수 (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"기준 각도에 따른 판단 성능 (관절 인식 성공 {len(valid)} 프레임)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "threshold_curve.png"
    fig.savefig(out, dpi=300)
    print(f"\n최고 F1 기준 각도: {best}° (현재 45°)")
    print(f"저장: {out}")


def main():
    rows = load()
    print(f"총 {len(rows)} 프레임")
    plot_confusion(rows)
    plot_threshold(rows)
    plt.show()


if __name__ == "__main__":
    main()