import math
import time
from collections import deque


class FallDetector:
    """
    어깨-골반 각도로 자세를 판단하고, 사람(track_id)별로 낙상 지속 시간을 추적.
    - 다수결 스무딩: 최근 여러 프레임을 보고 자세를 결정 (한두 프레임 오류 무시)
    - unknown 무시: 관절 인식 실패 프레임은 판단에서 제외
    - 회복 시간: 일정 시간 계속 서 있어야 '사건 종료'로 인정 (중복 이벤트 방지)
    """

    def __init__(self, lying_angle_threshold=45, high_risk_seconds=15, medium_risk_seconds=5,
                 window_frames=15, enter_ratio=0.7, recovery_seconds=3):
        self.lying_angle_threshold = lying_angle_threshold
        self.high_risk_seconds = high_risk_seconds
        self.medium_risk_seconds = medium_risk_seconds
        self.window_frames = window_frames      # 다수결에 쓰는 최근 프레임 수
        self.enter_ratio = enter_ratio          # 이 비율 이상이면 상태 전환
        self.recovery_seconds = recovery_seconds
        self.states = {}

    def get_posture(self, keypoints):
        """한 프레임의 '원래' 자세 판단 (기존과 동일)."""
        left_shoulder = keypoints.get(5, (0, 0))
        right_shoulder = keypoints.get(6, (0, 0))
        left_hip = keypoints.get(11, (0, 0))
        right_hip = keypoints.get(12, (0, 0))

        if (0, 0) in [left_shoulder, right_shoulder, left_hip, right_hip]:
            return "unknown"

        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                           (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2,
                      (left_hip[1] + right_hip[1]) / 2)

        dx = abs(hip_center[0] - shoulder_center[0])
        dy = abs(hip_center[1] - shoulder_center[1])
        if dx == 0 and dy == 0:
            return "unknown"

        angle = math.degrees(math.atan2(dy, dx))
        return "lying" if angle < self.lying_angle_threshold else "standing"

    def _new_state(self):
        return {
            "history": deque(maxlen=self.window_frames),
            "smoothed": "standing",
            "lying_since": None,
            "standing_since": None,
            "notified_medium": False,
            "notified_high": False,
        }

    def _smooth(self, state, posture):
        """최근 프레임 다수결. 애매하면 이전 상태를 유지(히스테리시스)."""
        if posture != "unknown":          # 인식 실패 프레임은 투표에서 제외
            state["history"].append(posture)

        hist = state["history"]
        if hist:
            lying_ratio = hist.count("lying") / len(hist)
            if lying_ratio >= self.enter_ratio:
                state["smoothed"] = "lying"
            elif lying_ratio <= 1 - self.enter_ratio:
                state["smoothed"] = "standing"
            # 그 사이 값이면 바꾸지 않음
        return state["smoothed"]

    def update(self, track_id, posture):
        """
        반환값: {
            "posture": 스무딩된 자세 (화면 표시용),
            "risk": "LOW"/"MEDIUM"/"HIGH",
            "duration": 누워 있는 시간(초),
            "new_event": None / "MEDIUM" / "HIGH"
        }
        """
        now = time.time()
        if track_id not in self.states:
            self.states[track_id] = self._new_state()
        state = self.states[track_id]

        smoothed = self._smooth(state, posture)
        new_event = None
        duration = 0

        if smoothed == "lying":
            state["standing_since"] = None
            if state["lying_since"] is None:
                state["lying_since"] = now
            duration = now - state["lying_since"]
        elif state["lying_since"] is not None:
            # 일어났다고 바로 사건을 끝내지 않고, recovery_seconds 동안 계속 서 있는지 확인
            if state["standing_since"] is None:
                state["standing_since"] = now
            if now - state["standing_since"] >= self.recovery_seconds:
                state["lying_since"] = None
                state["standing_since"] = None
                state["notified_medium"] = False
                state["notified_high"] = False

        if duration >= self.high_risk_seconds:
            risk = "HIGH"
            if not state["notified_high"]:
                new_event = "HIGH"
                state["notified_high"] = True
        elif duration >= self.medium_risk_seconds:
            risk = "MEDIUM"
            if not state["notified_medium"]:
                new_event = "MEDIUM"
                state["notified_medium"] = True
        else:
            risk = "LOW"

        return {"posture": smoothed, "risk": risk,
                "duration": round(duration, 1), "new_event": new_event}