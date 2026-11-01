import math
import time


class ObjectDisplacementDetector:
    def __init__(self, target_classes, calibration_seconds=3, move_distance_threshold=60,
                 confirm_seconds=2, missing_confirm_seconds=4):
        """
        calibration_seconds: 시작 후 이 시간(초) 동안 "원래 위치" 측정
        move_distance_threshold: 중심점이 이 거리(픽셀) 이상 벗어나면 "이동 후보"
        confirm_seconds: 이동 후보 상태가 이 시간(초) 이상 유지되면 "이동" 확정
        missing_confirm_seconds: 원래 있던 물건이 이 시간(초) 이상 안 보이면 "사라짐" 확정
        """
        self.target_classes = target_classes
        self.calibration_seconds = calibration_seconds
        self.move_distance_threshold = move_distance_threshold
        self.confirm_seconds = confirm_seconds
        self.missing_confirm_seconds = missing_confirm_seconds

        self.start_time = None          # 첫 update 때 정해짐 (영상 재생 시각에도 맞추기 위함)
        self.calibrating = True

        self.baseline = {cls: [] for cls in target_classes}
        self.initial_position = {cls: None for cls in target_classes}
        self.moved_since = {cls: None for cls in target_classes}
        self.missing_since = {cls: None for cls in target_classes}
        self.notified = {cls: False for cls in target_classes}

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections, now=None):
        """
        detections: {"laptop": bbox 또는 None, ...}
        now: 현재 시각(초). 안 넘기면 실제 시계 사용. 영상 재생 때는 영상 속 시각을 넘김.
        """
        if now is None:
            now = time.time()
        if self.start_time is None:
            self.start_time = now

        # --- 원래 위치 측정 (calibration) ---
        if self.calibrating:
            for cls in self.target_classes:
                bbox = detections.get(cls)
                if bbox is not None:
                    self.baseline[cls].append(self._center(bbox))

            if now - self.start_time >= self.calibration_seconds:
                for cls in self.target_classes:
                    points = self.baseline[cls]
                    if points:
                        avg_x = sum(p[0] for p in points) / len(points)
                        avg_y = sum(p[1] for p in points) / len(points)
                        self.initial_position[cls] = (avg_x, avg_y)
                self.calibrating = False
            else:
                return {"calibrating": True, "results": {}}

        # --- 이동 / 사라짐 감지 ---
        results = {}
        for cls in self.target_classes:
            bbox = detections.get(cls)

            if self.initial_position[cls] is None:
                results[cls] = {"status": "no_baseline", "duration": 0, "new_event": False}
                continue

            if bbox is None:
                # ★ 원래 있던 물건이 일정 시간 이상 안 보이면 이동(사라짐)으로 판단
                self.moved_since[cls] = None
                if self.missing_since[cls] is None:
                    self.missing_since[cls] = now
                missing_duration = now - self.missing_since[cls]
                new_event = False
                if missing_duration >= self.missing_confirm_seconds and not self.notified[cls]:
                    new_event = True
                    self.notified[cls] = True
                results[cls] = {"status": "missing", "duration": round(missing_duration, 1),
                                "new_event": new_event}
                continue

            self.missing_since[cls] = None  # 다시 보이면 초기화

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

                results[cls] = {"status": "moved", "duration": round(moved_duration, 1),
                                "new_event": new_event}
            else:
                # 원래 자리로 돌아오면 초기화 (다음 이동을 새로 감지할 수 있게)
                self.moved_since[cls] = None
                self.notified[cls] = False
                results[cls] = {"status": "ok", "duration": 0, "new_event": False}

        return {"calibrating": False, "results": results}