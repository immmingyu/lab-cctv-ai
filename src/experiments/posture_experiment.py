import csv
import math
import time
from pathlib import Path

import cv2

from src.capture.camera import open_source, read_frame
from src.detection.pose_estimator import PoseEstimator
from src.events.fall_detector import FallDetector

OUT_CSV = Path("data/experiments/posture_log.csv")
PREP_SECONDS = 6            # 자세를 바꾸는 준비 시간 (기록 안 함)
RECORD_SECONDS = 15         # 자세를 유지하며 기록하는 시간
ROUNDS = 2
BBOX_RATIO_THRESHOLD = 1.3  # 처음에 썼던 박스 비율 방식의 기준값

# (정답 라벨, 화면 안내 문구)
SCHEDULE = [
    ("stand", "STAND"),
    ("sit", "SIT on chair"),
    ("bend", "BEND (pick up, repeat)"),
    ("lie_side", "LIE (sideways)"),
    ("lie_front", "LIE (head to camera)"),
]


def torso_angle(kp):
    """어깨 중심-골반 중심 선의 각도 (수직 90도, 수평 0도). 관절 인식 실패 시 None."""
    pts = [kp.get(i, (0, 0)) for i in (5, 6, 11, 12)]
    if (0, 0) in pts:
        return None
    ls, rs, lh, rh = pts
    sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
    dx, dy = abs(hx - sx), abs(hy - sy)
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def bbox_area(p):
    x1, y1, x2, y2 = p["bbox"]
    return (x2 - x1) * (y2 - y1)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cap = open_source(0)
    pose_estimator = PoseEstimator()
    fall_detector = FallDetector()

    steps = [(r + 1, key, text) for r in range(ROUNDS) for key, text in SCHEDULE]
    step_len = PREP_SECONDS + RECORD_SECONDS
    counts = {}

    with open(OUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "label", "t", "person", "track_id", "bbox_ratio",
                         "angle", "bbox_pred", "pose_pred", "smooth_pred"])
        start = time.time()

        while True:
            ret, frame = read_frame(cap, mirror=True)
            if not ret:
                break
            elapsed = time.time() - start
            idx = int(elapsed // step_len)
            if idx >= len(steps):
                break
            rnd, label, text = steps[idx]
            in_step = elapsed - idx * step_len
            recording = in_step >= PREP_SECONDS

            # 준비 시간에도 판단은 계속 돌려서 스무딩 상태를 최신으로 유지
            people = pose_estimator.estimate(frame)
            if people:
                p = max(people, key=bbox_area)
                x1, y1, x2, y2 = p["bbox"]
                ratio = (x2 - x1) / (y2 - y1) if y2 > y1 else 0
                bbox_pred = "lying" if ratio >= BBOX_RATIO_THRESHOLD else "standing"
                angle = torso_angle(p["keypoints"])
                pose_pred = fall_detector.get_posture(p["keypoints"])
                smooth_pred = fall_detector.update(p["track_id"], pose_pred)["posture"]
                track_id = p["track_id"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"bbox:{bbox_pred} pose:{pose_pred} smooth:{smooth_pred}",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                ratio = angle = track_id = None
                bbox_pred = pose_pred = smooth_pred = "none"

            if recording:
                writer.writerow([rnd, label, round(elapsed, 3), int(bool(people)), track_id,
                                 round(ratio, 3) if ratio is not None else "",
                                 round(angle, 1) if angle is not None else "",
                                 bbox_pred, pose_pred, smooth_pred])
                counts[label] = counts.get(label, 0) + 1
                msg = f"REC  {text}  {RECORD_SECONDS - (in_step - PREP_SECONDS):.0f}s"
                color = (0, 0, 255)
            else:
                msg = f"NEXT: {text}  in {PREP_SECONDS - in_step:.0f}s"
                color = (0, 255, 255)

            cv2.putText(frame, f"Round {rnd}/{ROUNDS}  Step {idx % len(SCHEDULE) + 1}/{len(SCHEDULE)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, msg, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Posture Experiment", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n===== 기록 완료 =====")
    for key, _ in SCHEDULE:
        print(f"{key:<10} {counts.get(key, 0)} 프레임")
    print(f"저장: {OUT_CSV}")


if __name__ == "__main__":
    main()