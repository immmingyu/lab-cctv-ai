import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from src.capture.camera import open_source, read_frame
from src.detection.object_detector import ObjectDetector
from src.detection.pose_estimator import PoseEstimator
from src.detection.vlm_client import VlmClient, crop_person
from src.events.event_sender import EventSender
from src.events.fall_detector import FallDetector, touches_edge
from src.events.intrusion_detector import IntrusionDetector
from src.events.object_displacement_detector import ObjectDisplacementDetector
from src.events.stillness_trigger import StillnessTrigger
from src.schemas.event_schema import (
    create_displacement_event,
    create_fall_event,
    create_intrusion_event,
    create_vlm_fall_event,
)

LOG_PATH = "data/tracking_log.csv"  
EVENT_LOG_PATH = "data/events.jsonl"    
SNAPSHOT_DIR = Path("data/snapshots")
CAPTURE_DIR = Path("data/captures")

SKELETON_LINES = [(5, 6), (5, 11), (6, 12), (11, 12)]
RISK_COLORS = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
ROI_POINTS = [(178, 395), (185, 475), (311, 478), (318, 407)]

# 나중에 5개로 늘릴 땐 여기에 클래스만 추가: ["laptop", "book", "cup", "cell phone", "backpack"]
TARGET_OBJECTS = ["laptop", "book"]

sender = EventSender()


def parse_args():
    parser = argparse.ArgumentParser(description="연구실 지능형 CCTV")
    parser.add_argument("--only", choices=["all", "fall", "intrusion", "object"], default="all",
                        help="한 기능만 화면에 표시하고 감지 (사진 촬영용)")
    parser.add_argument("--no-vlm", action="store_true", help="조건부 VLM 재확인 끄기 (규칙만)")
    return parser.parse_args()


def init_logs():
    with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "x1", "y1", "x2", "y2", "posture", "duration", "risk"])
    open(EVENT_LOG_PATH, mode="w", encoding="utf-8").close()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def log_track(track_id, bbox, posture, duration, risk):
    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = bbox
        writer.writerow([timestamp, track_id, x1, y1, x2, y2, posture, duration, risk])


def save_snapshot(frame, event):
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event['event_id']}.jpg"
    ok = cv2.imwrite(str(SNAPSHOT_DIR / filename), frame)
    return f"snapshots/{filename}" if ok else None


def save_event(event, frame=None):
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

    # ★ 안 보이는 물건은 박스가 없으니 화면 왼쪽 위에 따로 표시
    if not displacement_result["calibrating"]:
        y = 60
        for cls, r in displacement_result["results"].items():
            if r.get("status") == "missing":
                cv2.putText(frame, f"{cls}: MISSING ({r.get('duration', 0)}s)", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25
            elif r.get("status") == "occluded":
                cv2.putText(frame, f"{cls}: HIDDEN BY PERSON", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                y += 25


def main():
    args = parse_args()
    mode = args.only
    show_fall = mode in ("all", "fall")
    show_intrusion = mode in ("all", "intrusion")
    show_object = mode in ("all", "object")
    print(f"[모드] {mode}  ('s' 캡처 저장, 'q' 종료)")

    cap = open_source()
    pose_estimator = PoseEstimator() if (show_fall or show_intrusion) else None
    fall_detector = FallDetector()
    # 조건부 VLM: 오래 안 움직이는데 규칙이 누움이 아니라고 할 때만 물어봄 (별도 스레드, 영상 처리는 계속)
    use_vlm = show_fall and not args.no_vlm
    stillness = StillnessTrigger()
    vlm = VlmClient() if use_vlm else None
    vlm_pool = ThreadPoolExecutor(max_workers=1) if use_vlm else None
    vlm_jobs = []   # [(future, track_id, bbox, 정지 시간)]
    intrusion_detector = IntrusionDetector(ROI_POINTS, stay_seconds_threshold=5)
    object_detector = ObjectDetector(TARGET_OBJECTS, conf_thresholds={"book": 0.15}) if show_object else None
    displacement_detector = ObjectDisplacementDetector(TARGET_OBJECTS) if show_object else None
    init_logs()

    while True:
        ret, frame = read_frame(cap, mirror=True)
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        pending_events = []  # 이번 프레임의 이벤트 (그리기가 끝난 뒤 스냅샷과 함께 저장)
        raw_frame = frame.copy() if use_vlm else None   # VLM에 보낼 원본 (그림 없는 상태)

        # --- 제한 구역 표시 ---
        if show_intrusion:
            intrusion_detector.draw(frame)

        # --- 물건 이동 ---
        if show_object:
            # 같은 YOLO 결과의 사람 박스도 넘겨서, 사람이 가리는 동안은 사라짐으로 세지 않게 함
            obj_detections, person_boxes = object_detector.detect_all(frame)
            displacement_result = displacement_detector.update(obj_detections, person_boxes=person_boxes)
            draw_objects(frame, obj_detections, displacement_result)
            if not displacement_result["calibrating"]:
                for cls, r in displacement_result["results"].items():
                    if r.get("new_event"):
                        pending_events.append(create_displacement_event(cls, r["duration"], reason=r["status"]))

        # --- 사람 (낙상 / 제한 구역) ---
        people = pose_estimator.estimate(frame) if pose_estimator else []
        for p in people:
            track_id = p["track_id"]
            x1, y1, x2, y2 = p["bbox"]
            keypoints = p["keypoints"]
            # 몸이 화면 밖으로 잘림 -> 누움 쪽 판단과 VLM 보류 (화면에 EDGE로 표시해 촬영 구도 확인)
            edge = touches_edge((x1, y1, x2, y2), frame.shape)

            if show_fall:
                posture = fall_detector.get_posture(keypoints)
                fall_status = fall_detector.update(track_id, posture, hold=edge)
                posture = fall_status["posture"]
                risk = fall_status["risk"]
                duration = fall_status["duration"]

                color = RISK_COLORS[risk]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track_id} [{posture}] risk={risk}"
                if posture == "lying":
                    label += f" ({duration}s)"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for idx, (x, y) in keypoints.items():
                    if (x, y) != (0, 0):
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                for (a, b) in SKELETON_LINES:
                    pa, pb = keypoints.get(a, (0, 0)), keypoints.get(b, (0, 0))
                    if pa != (0, 0) and pb != (0, 0):
                        cv2.line(frame, pa, pb, (255, 0, 0), 2)

                log_track(track_id, (x1, y1, x2, y2), posture, duration, risk)

                if fall_status["new_event"] is not None:
                    pending_events.append(create_fall_event(
                        track_id=track_id, bbox=(x1, y1, x2, y2),
                        risk=fall_status["new_event"], duration=duration, confidence=0.9,
                    ))

                if use_vlm:
                    st = stillness.update(track_id, keypoints, (x1, y1, x2, y2), posture, hold=edge)
                    if st["ask"]:
                        crop = crop_person(raw_frame, (x1, y1, x2, y2))   # 박스, 뼈대가 그려지기 전 원본에서 자름
                        vlm_jobs.append((vlm_pool.submit(vlm.ask_lying, crop), track_id, (x1, y1, x2, y2), st["still"]))
                        print(f"[VLM] ID {track_id} {st['still']}초 정지, 규칙={posture} -> 재확인 요청")
            else:
                # 낙상 표시 없이 사람 박스와 ID만
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if edge:   # 박스 위쪽이 잘릴 수 있어 박스 안쪽에 표시
                cv2.putText(frame, "EDGE", (x1 + 5, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            if show_intrusion:
                foot_point = get_foot_point((x1, y1, x2, y2))
                intrusion_status = intrusion_detector.update(track_id, intrusion_detector.is_inside(foot_point))

                dot_color = (0, 0, 255) if intrusion_status["inside"] else (0, 255, 0)
                cv2.circle(frame, foot_point, 6, dot_color, -1)
                if intrusion_status["inside"]:
                    cv2.putText(frame, f"IN AREA ({intrusion_status['duration']}s)",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if intrusion_status["new_event"]:
                    pending_events.append(create_intrusion_event(track_id, intrusion_status["duration"]))

        # --- VLM 답이 온 것만 처리 (안 끝난 건 다음 프레임에서 다시 확인) ---
        for job in [j for j in vlm_jobs if j[0].done()]:
            vlm_jobs.remove(job)
            future, track_id, bbox, still = job
            ans = future.result()
            print(f"[VLM] ID {track_id} 답: {ans}")
            if ans["lying"]:
                pending_events.append(create_vlm_fall_event(track_id, bbox, still, ans["reason"]))

        for ev in pending_events:
            save_event(ev, frame)

        cv2.imshow(f"Lab CCTV - {mode}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # 포스터용 캡처
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            path = CAPTURE_DIR / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), frame)
            print(f"[캡처 저장] {path}")

    cap.release()
    cv2.destroyAllWindows()
    if vlm_pool:
        vlm_pool.shutdown(wait=False, cancel_futures=True)
    sender.flush()


if __name__ == "__main__":
    main()