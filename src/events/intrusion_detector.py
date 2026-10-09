import cv2
import numpy as np
import time

class IntrusionDetector:
    def __init__(self, roi_points, stay_seconds_threshold=5):
        """
        roi_points: [(x1,y1), (x2,y2), ...] ROI 다각형 꼭짓점
        stay_seconds_threshold: 이 시간(초) 이상 머무르면 접근 이벤트 발생
        """
        self.roi_points = roi_points
        self.stay_seconds_threshold = stay_seconds_threshold
        self.states = {}  # track_id별 상태 기억

    def is_inside(self, point):
        result = cv2.pointPolygonTest(self._as_contour(), point, measureDist=False)
        return result >= 0

    def _as_contour(self):
        return np.array(self.roi_points, dtype=np.int32)

    def draw(self, frame, color=(0, 255, 255), thickness=2):
        pts = np.array(self.roi_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    def update(self, track_id, is_inside_now):
        """
        매 프레임마다 호출. track_id가 ROI 안에 머문 시간을 추적.

        반환값: {
            "inside": bool,
            "duration": float,
            "new_event": bool  <- 방금 막 '접근 확정'(임계시간 초과) 시점일 때만 True
        }
        """
        now = time.time()

        if track_id not in self.states:
            self.states[track_id] = {"entered_at": None, "notified": False}

        state = self.states[track_id]
        new_event = False

        if is_inside_now:
            if state["entered_at"] is None:
                state["entered_at"] = now  # 방금 들어옴
            duration = now - state["entered_at"]

            if duration >= self.stay_seconds_threshold and not state["notified"]:
                new_event = True
                state["notified"] = True
        else:
            # 구역 밖으로 나가면 리셋 (다음 번 접근 때도 이벤트가 다시 발생하도록)
            state["entered_at"] = None
            state["notified"] = False
            duration = 0

        return {"inside": is_inside_now, "duration": round(duration, 1), "new_event": new_event}