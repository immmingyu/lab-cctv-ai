import csv
import time
from pathlib import Path

import cv2

from src.capture.camera import open_source, read_frame
from src.detection.pose_estimator import PoseEstimator
from src.events.fall_detector import FallDetector

OUT_DIR = Path("data/experiments")
CSV_PATH = OUT_DIR / "smoothing_log.csv"


class BaselineFallDetector:
    """비교용: 스무딩 이전의 예전 방식. lying이 아닌 프레임이 하나라도 나오면 즉시 초기화."""

    def __init__(self, high_risk_seconds=15, medium_risk_seconds=5):
        self.high_risk_seconds = high_risk_seconds
        self.medium_risk_seconds = medium_risk_seconds
        self.states = {}

    def update(self, track_id, posture):
        now = time.time()
        state = self.states.setdefault(
            track_id, {"lying_since": None, "notified_medium": False, "notified_high": False})

        if posture == "lying":
            if state["lying_since"] is None:
                state["lying_since"] = now
            duration = now - state["lying_since"]
        else:
            state.update(lying_since=None, notified_medium=False, notified_high=False)
            duration = 0

        if duration >= self.high_risk_seconds and not state["notified_high"]:
            state["notified_high"] = True
            return "HIGH"
        if duration >= self.medium_risk_seconds and not state["notified_medium"]:
            state["notified_medium"] = True
            return "MEDIUM"
        return None


def bbox_area(p):
    x1, y1, x2, y2 = p["bbox"]
    return (x2 - x1) * (y2 - y1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = open_source()
    pose_estimator = PoseEstimator()
    smoothed_detector = FallDetector()        # 현재 방식
    baseline_detector = BaselineFallDetector()  # 예전 방식

    counts = {"baseline": 0, "smoothed": 0}
    start = time.time()
    frames = 0

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "track_id", "raw", "smoothed", "baseline_event", "smoothed_event"])

        while True:
            ret, frame = read_frame(cap, mirror=True)
            if not ret:
                break
            frames += 1
            t = time.time() - start

            people = pose_estimator.estimate(frame)
            if people:
                p = max(people, key=bbox_area)  # 화면에서 가장 큰 사람(실험 대상)만 기록
                raw = smoothed_detector.get_posture(p["keypoints"])
                status = smoothed_detector.update(p["track_id"], raw)
                base_event = baseline_detector.update(p["track_id"], raw)

                if base_event:
                    counts["baseline"] += 1
                if status["new_event"]:
                    counts["smoothed"] += 1

                writer.writerow([round(t, 3), p["track_id"], raw, status["posture"],
                                 base_event or "", status["new_event"] or ""])

                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {p['track_id']}  raw: {raw}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"smoothed: {status['posture']} ({status['duration']}s)", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.putText(frame, f"events  baseline={counts['baseline']}  smoothed={counts['smoothed']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Smoothing Experiment", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start
    print("\n===== 실험 결과 =====")
    print(f"측정 시간: {elapsed:.1f}초, 프레임: {frames}, 평균 FPS: {frames / elapsed:.1f}")
    print(f"예전 방식(스무딩 없음) 이벤트: {counts['baseline']}건")
    print(f"현재 방식(스무딩)      이벤트: {counts['smoothed']}건")
    print(f"기록 파일: {CSV_PATH}")


if __name__ == "__main__":
    main()