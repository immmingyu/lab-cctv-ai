import math
import time


class StillnessTrigger:
    """
    조건부 VLM 호출 시점 판단.
    사람의 관절이 still_seconds 동안 거의 움직이지 않고, 규칙이 '누움'으로 판단하지 않을 때
    -> 정지 구간마다 한 번만 True (VLM에게 물어볼 때).
    """

    def __init__(self, still_seconds=8, motion_ratio=0.05, min_points=3, reset_frames=3):
        """
        still_seconds: 이 시간(초) 이상 거의 안 움직이면 정지로 봄
        motion_ratio: 정지 시작 때 관절 위치에서 평균 이동 거리가 사람 박스 긴 변의 이 비율보다 작으면 '거의 안 움직임'
        min_points: 비교할 수 있는 관절(두 시점 모두 인식된 것)이 이보다 적으면 그 프레임은 '움직임'으로 봄
        reset_frames: '움직임'이 이 프레임 수 이상 연속일 때만 정지 시간을 초기화 (한두 프레임 관절 튐 무시)
        (값은 README '평가 설정'에 확정, 새 영상 검증 전 고정)
        """
        self.still_seconds = still_seconds
        self.motion_ratio = motion_ratio
        self.min_points = min_points
        self.reset_frames = reset_frames
        self.states = {}

    def _avg_move(self, anchor, keypoints):
        """두 시점 모두 인식된 관절들의 평균 이동 거리. 비교할 관절이 부족하면 None."""
        dists = [math.hypot(keypoints[i][0] - a[0], keypoints[i][1] - a[1])
                 for i, a in anchor.items()
                 if a != (0, 0) and keypoints.get(i, (0, 0)) != (0, 0)]
        if len(dists) < self.min_points:
            return None
        return sum(dists) / len(dists)

    def update(self, track_id, keypoints, bbox, rule_posture, now=None, hold=False):
        """
        rule_posture: 규칙(FallDetector)이 판단한 스무딩 자세
        hold: True면 이 프레임은 판단 보류 (예: 사람 박스가 화면 가장자리에 닿음) -> 묻지 않음
        반환값: {"still": 정지 시간(초), "ask": 지금 VLM에게 물어볼지}
        """
        now = time.time() if now is None else now
        x1, y1, x2, y2 = bbox
        limit = max(x2 - x1, y2 - y1) * self.motion_ratio

        state = self.states.get(track_id)
        if state is None:
            state = {"anchor": dict(keypoints), "since": now, "asked": False, "moving": 0}
            self.states[track_id] = state
        else:
            move = self._avg_move(state["anchor"], keypoints)
            if move is None or move > limit:
                state["moving"] += 1
                if state["moving"] >= self.reset_frames:
                    # 연속으로 움직였음 -> 지금 위치를 새 기준으로, 정지 시간 다시 셈
                    state.update(anchor=dict(keypoints), since=now, asked=False, moving=0)
            else:
                state["moving"] = 0

        still = now - state["since"]
        ask = False
        if (still >= self.still_seconds and not state["asked"] and rule_posture != "lying"
                and state["moving"] == 0 and not hold):   # 튀는 프레임, 보류 프레임에서는 묻지 않음
            ask = True
            state["asked"] = True   # 같은 정지 구간에서는 한 번만
        return {"still": round(still, 1), "ask": ask}
