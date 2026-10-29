import csv
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from src.capture.camera import open_source, read_frame

N_FRAMES = 100
OUT_CSV = Path("data/experiments/speed_benchmark.csv")
MODELS = [("Pose (yolov8n-pose)", "yolov8n-pose.pt"),
          ("물건 검출 (yolov8s)", "yolov8s.pt")]


def measure(model, frames, device):
    for f in frames[:10]:                      # 워밍업 (첫 추론은 느리므로 제외)
        model(f, device=device, verbose=False)
    times = []
    for f in frames:
        t = time.perf_counter()
        model(f, device=device, verbose=False)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.mean(times)


def main():
    print("GPU 사용 가능:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    # 1) 카메라가 실제로 주는 FPS
    cap = open_source()
    frames = []
    start = time.perf_counter()
    while len(frames) < N_FRAMES:
        ret, frame = read_frame(cap, mirror=True)
        if ret:
            frames.append(frame)
    cam_fps = N_FRAMES / (time.perf_counter() - start)
    cap.release()
    h, w = frames[0].shape[:2]
    print(f"\n카메라: {w}x{h}, 실제 {cam_fps:.1f} FPS\n")

    # 2) 같은 프레임으로 모델별 CPU / GPU 추론 시간
    rows = [["단계", "장치", "평균 ms", "환산 FPS"]]
    rows.append(["카메라 프레임 수신", "-", round(1000 / cam_fps, 1), round(cam_fps, 1)])
    for name, path in MODELS:
        model = YOLO(path)
        for device, label in (("cpu", "CPU"), (0, "GPU")):
            if device == 0 and not torch.cuda.is_available():
                continue
            ms = measure(model, frames, device)
            rows.append([name, label, round(ms, 1), round(1000 / ms, 1)])
            print(f"{name:<22} {label}: {ms:6.1f} ms  (≈ {1000 / ms:5.1f} FPS)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()