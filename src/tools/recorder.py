import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.capture.camera import open_source, read_frame

# 숫자 키로 시도 종류 표시, e로 시도 끝
# 9 = 정리 시간(reset): 시도 사이에 물건을 되돌리는 등 채점하지 않을 구간
LABEL_KEYS = {ord("1"): "fall", ord("2"): "intrusion", ord("3"): "object_move", ord("0"): "normal",
              ord("9"): "reset"}
LABEL_TEXT = {"fall": "FALL", "intrusion": "INTRUSION", "object_move": "OBJECT MOVE", "normal": "NORMAL (no event)",
              "reset": "RESET (tidy up)"}


def parse_args():
    p = argparse.ArgumentParser(description="평가용 원본 영상 녹화 + 시도 구간 표시")
    p.add_argument("--name", required=True, help="녹화 이름 (예: eval_object_room2)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path("data/recordings") / f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_source()

    # 1) 카메라가 실제로 주는 FPS 측정 (영상 파일의 재생 속도를 실제와 맞추기 위함)
    n, t0 = 0, time.perf_counter()
    while n < 30:
        ret, _ = read_frame(cap)
        if ret:
            n += 1
    fps = n / (time.perf_counter() - t0)

    ret, frame = read_frame(cap)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames_file = open(out_dir / "frames.csv", "w", newline="", encoding="utf-8")
    frames_csv = csv.writer(frames_file)
    frames_csv.writerow(["frame", "t"])

    marks = []        # [label, start_t, end_t, start_frame, end_frame]
    current = None
    idx = 0
    start = time.perf_counter()

    print(f"[녹화 시작] {out_dir}  ({w}x{h}, 약 {fps:.1f} FPS)")
    print("  1=낙상  2=제한 구역  3=물건 이동  0=정상 동작  9=정리 시간  →  e=끝  q=종료")

    while True:
        ret, frame = read_frame(cap)            # 원본 그대로 (좌우 반전 없음)
        if not ret:
            break
        t = time.perf_counter() - start
        writer.write(frame)
        frames_csv.writerow([idx, round(t, 4)])

        # 화면 표시용 (거울처럼 보이게만 하고, 저장은 원본)
        preview = cv2.flip(frame, 1)
        cv2.circle(preview, (20, 20), 8, (0, 0, 255), -1)
        cv2.putText(preview, f"REC {int(t // 60):02d}:{int(t % 60):02d}", (35, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        counts = {k: sum(m[0] == k for m in marks) for k in LABEL_TEXT}
        cv2.putText(preview, f"fall {counts['fall']}  intr {counts['intrusion']}  "
                             f"obj {counts['object_move']}  normal {counts['normal']}  reset {counts['reset']}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if current:
            cv2.putText(preview, f"TRIAL: {LABEL_TEXT[current[0]]} ({t - current[1]:.1f}s)  [e]=end",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Recorder", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in LABEL_KEYS:
            if current:                          # 끝 표시를 잊었으면 자동으로 닫음
                current[2], current[4] = t, idx
                marks.append(current)
            current = [LABEL_KEYS[key], t, None, idx, None]
            print(f"  시도 시작: {LABEL_TEXT[current[0]]}  ({t:.1f}s)")
        elif key == ord("e") and current:
            current[2], current[4] = t, idx
            marks.append(current)
            print(f"  시도 끝:   {LABEL_TEXT[current[0]]}  ({t - current[1]:.1f}초)")
            current = None
        elif key == ord("q"):
            break
        idx += 1

    if current:
        current[2], current[4] = time.perf_counter() - start, idx
        marks.append(current)

    cap.release()
    writer.release()
    frames_file.close()
    cv2.destroyAllWindows()

    with open(out_dir / "marks.csv", "w", newline="", encoding="utf-8") as f:
        mw = csv.writer(f)
        mw.writerow(["trial", "label", "start_t", "end_t", "start_frame", "end_frame"])
        # 시도 번호는 진짜 시도만 1, 2, 3... (--groups 번호가 밀리지 않게), 정리 시간은 R1, R2...
        n_trial = n_reset = 0
        for label, st, et, sf, ef in marks:
            if label == "reset":
                n_reset += 1
                num = f"R{n_reset}"
            else:
                n_trial += 1
                num = n_trial
            mw.writerow([num, label, round(st, 3), round(et, 3), sf, ef])

    print(f"\n[녹화 완료] {idx}프레임, {time.perf_counter() - start:.1f}초")
    for k, v in LABEL_TEXT.items():
        print(f"  {v:<20} {sum(m[0] == k for m in marks)}회")
    print(f"  저장 위치: {out_dir}")


if __name__ == "__main__":
    main()