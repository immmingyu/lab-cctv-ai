from ultralytics import YOLO
from src.config import YOLO_DEVICE

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", conf_threshold=0.5):
        """
        model_path: Pose 전용 모델. 처음 실행 시 자동 다운로드됨.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def estimate(self, frame):
        """
        반환값: [
            {
                "track_id": 1,
                "bbox": (x1, y1, x2, y2),
                "keypoints": {0: (x, y), 1: (x, y), ...}  # 17개 관절
            },
            ...
        ]
        """
        results = self.model.track(frame, persist=True, verbose=False, device=YOLO_DEVICE)[0]

        people = []
        if results.boxes.id is None:
            return people

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            keypoints = {}
            kpts = results.keypoints[i].xy[0]  # (17, 2) 형태의 좌표
            for idx, (x, y) in enumerate(kpts):
                keypoints[idx] = (int(x), int(y))

            people.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "keypoints": keypoints
            })

        return people