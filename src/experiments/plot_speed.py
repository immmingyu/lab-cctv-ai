import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path("data/experiments/speed_benchmark.csv")
PNG_PATH = Path("data/experiments/speed_compare.png")

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac에서는 "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    data, cam_ms = {}, None
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["장치"] == "-":
                cam_ms = float(row["평균 ms"])
            else:
                data[(row["단계"], row["장치"])] = float(row["평균 ms"])

    devices = ["GPU", "CPU"]                       # 아래부터 GPU, 위에 CPU
    labels = ["GPU (RTX 3060)", "CPU"]
    pose = [data[("Pose (yolov8n-pose)", d)] for d in devices]
    obj = [data[("물건 검출 (yolov8s)", d)] for d in devices]
    totals = [p + o for p, o in zip(pose, obj)]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = list(range(len(devices)))
    ax.barh(y, pose, color="#3b82f6", height=0.5, label="Pose 추정 (yolov8n-pose)")
    ax.barh(y, obj, left=pose, color="#f59e0b", height=0.5, label="물건 검출 (yolov8s)")

    for i, total in enumerate(totals):
        ax.text(total + 2, i, f"{total:.1f} ms (≈ {1000 / total:.0f} FPS)", va="center", fontsize=10)

    if cam_ms:
        ax.axvline(cam_ms, color="#ef4444", linestyle="--", linewidth=1.5)
        ax.text(cam_ms + 2, 1.55, f"카메라 프레임 간격 {cam_ms:.0f} ms (이보다 짧아야 실시간)",
                color="#ef4444", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, 1.9)
    ax.set_xlim(0, max(totals) * 1.4)
    ax.set_xlabel("프레임 1장당 AI 처리 시간 (ms, 640x480)")
    ax.set_title("CPU vs GPU 처리 시간 비교", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300)
    print(f"저장 완료: {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()