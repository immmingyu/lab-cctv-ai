import cv2
import csv
import json
from datetime import datetime
from pathlib import Path
from src.capture.camera import open_source, read_frame
from src.detection.pose_estimator import PoseEstimator
from src.detection.object_detector import ObjectDetector
from src.events.fall_detector import FallDetector
from src.events.intrusion_detector import IntrusionDetector
from src.events.object_displacement_detector import ObjectDisplacementDetector
from src.events.event_sender import EventSender
from src.schemas.event_schema import (
    create_fall_event,
    create_intrusion_event,
    create_displacement_event,
)

LOG_PATH = "data/tracking_log.csv"
EVENT_LOG_PATH = "data/events.jsonl"
SNAPSHOT_DIR = Path("data/snapshots")  # ★

SKELETON_LINES = [(5, 6), (5, 11), (6, 12), (11, 12)]
RISK_COLORS = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
ROI_POINTS = [(417, 340), (346, 407), (510, 442), (584, 354)]

# 나중에 5개로 늘릴 땐 여기에 클래스만 추가: ["laptop", "book", "cup", "cell phone", "backpack"]
TARGET_OBJECTS = ["laptop", "book"]

sender = EventSender()

def init_logs():
    with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "x1", "y1", "x2", "y2", "posture", "duration", "risk"])
    open(EVENT_LOG_PATH, mode="w", encoding="utf-8").close()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)  # ★ (스냅샷은 DB가 참조하므로 삭제하지 않음)

def log_track(track_id, bbox, posture, duration, risk):
    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = bbox
        writer.writerow([timestamp, track_id, x1, y1, x2, y2, posture, duration, risk])

def save_snapshot(frame, event):  # ★
    """프레임을 JPG로 저장하고, 서버가 제공하는 경로(snapshots/파일명)를 반환."""
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event['event_id']}.jpg"
    ok = cv2.imwrite(str(SNAPSHOT_DIR / filename), frame)
    return f"snapshots/{filename}" if ok else None

def save_event(event, frame=None):  # ★ frame 인자 추가
    if frame is not None:
        event["clip_path"] = save_snapshot(frame, event)
    with open(EVENT_LOG_PATH, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    sender.send(event)
    print("\n[EVENT 발생]", json.dumps(event, ensure_ascii=False, indent=2))

def get_foot_point(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def draw_objects(frame, detections, displacement_result):
    if displacement_result["calibrating"]:
        cv2.putText(frame, "CALIBRATING... (don't move objects)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for cls, bbox in detections.items():
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox

        status = "calibrating"
        if not displacement_result["calibrating"]:
            status = displacement_result["results"].get(cls, {}).get("status", "?")

        color = (0, 0, 255) if status == "moved" else (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} [{status}]", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    cap = open_source(0)
    pose_estimator = PoseEstimator()
    fall_detector = FallDetector()
    intrusion_detector = IntrusionDetector(ROI_POINTS, stay_seconds_threshold=5)
    object_detector = ObjectDetector(TARGET_OBJECTS, conf_thresholds={"book": 0.15})
    displacement_detector = ObjectDisplacementDetector(TARGET_OBJECTS)
    init_logs()

    while True:
        ret, frame = read_frame(cap, mirror=True)
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        pending_events = []  # ★ 이번 프레임에서 발생한 이벤트를 모아둠

        intrusion_detector.draw(frame)

        # --- 물건 이동 감지 ---
        obj_detections = object_detector.detect(frame)
        displacement_result = displacement_detector.update(obj_detections)
        draw_objects(frame, obj_detections, displacement_result)

        if not displacement_result["calibrating"]:
            for cls, r in displacement_result["results"].items():
                if r.get("new_event"):
                    pending_events.append(create_displacement_event(cls, r["duration"]))  # ★

        # --- 사람 (낙상 / 제한 구역) ---
        people = pose_estimator.estimate(frame)

        for p in people:
            track_id = p["track_id"]
            x1, y1, x2, y2 = p["bbox"]
            keypoints = p["keypoints"]

            posture = fall_detector.get_posture(keypoints)
            fall_status = fall_detector.update(track_id, posture)
            posture = fall_status["posture"]  # ★ 스무딩된 자세로 표시/기록
            risk = fall_status["risk"]
            duration = fall_status["duration"]

            color = RISK_COLORS[risk]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id} [{posture}] risk={risk}"
            if posture == "lying":
                label += f" ({duration}s)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            foot_point = get_foot_point((x1, y1, x2, y2))
            is_inside_now = intrusion_detector.is_inside(foot_point)
            intrusion_status = intrusion_detector.update(track_id, is_inside_now)

            dot_color = (0, 0, 255) if intrusion_status["inside"] else (0, 255, 0)
            cv2.circle(frame, foot_point, 6, dot_color, -1)
            if intrusion_status["inside"]:
                cv2.putText(frame, f"IN AREA ({intrusion_status['duration']}s)",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if intrusion_status["new_event"]:
                pending_events.append(create_intrusion_event(track_id, intrusion_status["duration"]))  # ★

            for idx, (x, y) in keypoints.items():
                if (x, y) != (0, 0):
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            for (a, b) in SKELETON_LINES:
                pa, pb = keypoints.get(a, (0, 0)), keypoints.get(b, (0, 0))
                if pa != (0, 0) and pb != (0, 0):
                    cv2.line(frame, pa, pb, (255, 0, 0), 2)

            log_track(track_id, (x1, y1, x2, y2), posture, duration, risk)

            if fall_status["new_event"] is not None:
                pending_events.append(create_fall_event(  # ★
                    track_id=track_id, bbox=(x1, y1, x2, y2),
                    risk=fall_status["new_event"], duration=duration, confidence=0.9,
                ))

        # ★ 모든 그리기가 끝난 프레임으로 스냅샷 저장 + 이벤트 전송
        for ev in pending_events:
            save_event(ev, frame)

        cv2.imshow("Lab CCTV - Full Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sender.flush()

if __name__ == "__main__":
    main()