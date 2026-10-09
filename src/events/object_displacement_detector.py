import math
import time

class ObjectDisplacementDetector:
    def __init__(self, target_classes, calibration_seconds=3, move_distance_threshold=60, confirm_seconds=2):
        """
        calibration_seconds: 시작 후 이 시간(초) 동안 "원래 위치" 측정
        move_distance_threshold: 중심점이 이 거리(픽셀) 이상 벗어나면 "이동 후보"
        confirm_seconds: 이동 후보 상태가 이 시간(초) 이상 유지되면 진짜 "이동"으로 확정
        """
        self.target_classes = target_classes
        self.calibration_seconds = calibration_seconds
        self.move_distance_threshold = move_distance_threshold
        self.confirm_seconds = confirm_seconds

        self.start_time = time.time()
        self.calibrating = True

        self.baseline = {cls: [] for cls in target_classes}       # calibration 중 수집
        self.initial_position = {cls: None for cls in target_classes}  # 확정된 초기 위치
        self.moved_since = {cls: None for cls in target_classes}   # 이동 후보 시작 시각
        self.notified = {cls: False for cls in target_classes}     # 이벤트 중복 방지

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections):
        """
        detections: ObjectDetector.detect()의 결과 {"laptop": bbox 또는 None, ...}

        반환값: {
            "calibrating": bool,
            "results": {
                "laptop": {"status": "ok"/"moved"/"missing"/"calibrating", "new_event": bool},
                ...
            }
        }
        """
        now = time.time()

        # --- 초기 위치 측정(calibration) 단계 ---
        if self.calibrating:
            for cls in self.target_classes:
                bbox = detections.get(cls)
                if bbox is not None:
                    self.baseline[cls].append(self._center(bbox))

            if now - self.start_time >= self.calibration_seconds:
                # calibration 종료 -> 평균 위치를 초기 위치로 확정
                for cls in self.target_classes:
                    points = self.baseline[cls]
                    if points:
                        avg_x = sum(p[0] for p in points) / len(points)
                        avg_y = sum(p[1] for p in points) / len(points)
                        self.initial_position[cls] = (avg_x, avg_y)
                self.calibrating = False
            else:
                return {"calibrating": True, "results": {}}

        # --- 이동 감지 단계 ---
        results = {}
        for cls in self.target_classes:
            bbox = detections.get(cls)

            if self.initial_position[cls] is None:
                results[cls] = {"status": "no_baseline", "new_event": False}
                continue

            if bbox is None:
                results[cls] = {"status": "missing", "new_event": False}
                continue

            current_center = self._center(bbox)
            dist = self._distance(current_center, self.initial_position[cls])

            if dist >= self.move_distance_threshold:
                if self.moved_since[cls] is None:
                    self.moved_since[cls] = now
                moved_duration = now - self.moved_since[cls]

                new_event = False
                if moved_duration >= self.confirm_seconds and not self.notified[cls]:
                    new_event = True
                    self.notified[cls] = True

                results[cls] = {"status": "moved", "duration": round(moved_duration, 1), "new_event": new_event}
            else:
                self.moved_since[cls] = None
                self.notified[cls] = False
                results[cls] = {"status": "ok", "duration": 0, "new_event": False}

        return {"calibrating": False, "results": results}