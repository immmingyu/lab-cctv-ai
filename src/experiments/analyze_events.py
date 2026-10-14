import csv
import sys
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = Path("data/experiments/posture_log_v1.csv")
OUT_DIR = Path("data/experiments")
MEDIUM_SECONDS = 5.0     # 실제 시스템의 MEDIUM 경보 기준
RECOVERY_SECONDS = 3.0   # 실제 시스템의 회복 시간

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac에서는 "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ACTIVITIES = [("stand", "서기"), ("sit", "앉기"), ("bend", "숙이기"),
              ("lie_side", "눕기 (옆)"), ("lie_front", "눕기\n(카메라와 일직선)")]
METHODS = [("frame", "프레임 단위 판단", "#ef4444"),
           ("improved", "시간 흐름 판단 (현재)", "#3b82f6")]


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["t"]))
    return rows


def segments(rows):
    """같은 바퀴, 같은 동작으로 연속된 프레임을 하나의 구간으로 묶음."""
    return [(key[1], list(g)) for key, g in groupby(rows, key=lambda r: (r["round"], r["label"]))]


def alarms_frame(frames):
    """프레임 단위: 누움이 새로 시작될 때마다 경보 1회."""
    count, prev = 0, False
    for r in frames:
        now = r["pose_pred"] == "lying"
        if now and not prev:
            count += 1
        prev = now
    return count


def alarms_duration(frames, key, recovery):
    """누움이 MEDIUM_SECONDS 이상 이어지면 경보. recovery=0이면 끊기는 즉시 초기화."""
    lying_since = standing_since = None
    notified, count = False, 0
    for r in frames:
        t, lying = float(r["t"]), r[key] == "lying"
        if lying:
            standing_since = None
            if lying_since is None:
                lying_since = t
            if not notified and t - lying_since >= MEDIUM_SECONDS:
                count += 1
                notified = True
        elif lying_since is not None:
            if recovery == 0:
                lying_since, notified = None, False
            else:
                if standing_since is None:
                    standing_since = t
                if t - standing_since >= recovery:
                    lying_since = standing_since = None
                    notified = False
    return count

def alarms_improved(frames, key="smooth_pred"):
    """개선안: 연속 누움만 시간으로 셈 + 경보 후에는 회복 시간이 지나야 새 사건."""
    lying_since = standing_since = None
    notified, count = False, 0
    for r in frames:
        t, lying = float(r["t"]), r[key] == "lying"
        if lying:
            standing_since = None
            if lying_since is None:
                lying_since = t
            if not notified and t - lying_since >= MEDIUM_SECONDS:
                count += 1
                notified = True
        else:
            lying_since = None
            if notified:
                if standing_since is None:
                    standing_since = t
                if t - standing_since >= RECOVERY_SECONDS:
                    notified, standing_since = False, None
    return count


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    rows = load(path)
    segs = segments(rows)
    print(f"분석 파일: {path}  (총 {len(rows)} 프레임, {len(segs)}개 구간)\n")

    result = {}
    for label, frames in segs:
        r = result.setdefault(label, {"segments": 0, "frames": 0, "lying_frames": 0,
                                      "frame": 0, "duration": 0, "current": 0, "improved": 0})
        r["segments"] += 1
        r["frames"] += len(frames)
        r["lying_frames"] += sum(f["pose_pred"] == "lying" for f in frames)
        r["frame"] += alarms_frame(frames)
        r["duration"] += alarms_duration(frames, "pose_pred", 0)
        r["current"] += alarms_duration(frames, "smooth_pred", RECOVERY_SECONDS)
        r["improved"] += alarms_improved(frames)

    print(f"{'동작':<14}{'구간':>4}{'누움 판단 프레임':>16}{'즉시':>8}{'지속시간':>10}{'기존':>8}{'개선':>8}{'이상적':>8}")
    for key, name in ACTIVITIES:
        if key not in result:
            continue
        r = result[key]
        ideal = r["segments"] if key.startswith("lie") else 0
        pct = r["lying_frames"] / r["frames"] * 100
        print(f"{name.replace(chr(10), ' '):<14}{r['segments']:>4}{pct:>15.0f}%"
              f"{r['frame']:>8}{r['duration']:>10}{r['current']:>8}{r['improved']:>8}{ideal:>8}")

    # 그래프: 동작별 경보 횟수 비교
    present = [(k, n) for k, n in ACTIVITIES if k in result]
    x = np.arange(len(present))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (mkey, mname, color) in enumerate(METHODS):
        values = [result[k][mkey] for k, _ in present]
        bars = ax.bar(x + (i - 1) * width, values, width, label=mname, color=color)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", fontsize=9)

    for xi, (k, _) in zip(x, present):  # 이상적인 경보 수 표시
        ideal = result[k]["segments"] if k.startswith("lie") else 0
        ax.hlines(ideal, xi - 1.5 * width, xi + 1.5 * width, colors="black",
                  linestyles="--", linewidth=1.5)
    ax.plot([], [], "k--", label="이상적인 경보 수")

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in present])
    ax.set_ylabel("경보 횟수")
    ax.set_title("판단 방식별 경보 횟수 (누움이 아닌 동작은 0회, 눕기는 구간당 1회가 정답)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"event_level_{path.stem}.png"
    fig.savefig(out, dpi=300)
    print(f"\n저장: {out}")
    plt.show()


if __name__ == "__main__":
    main()