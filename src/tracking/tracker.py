from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        # (기존 코드 그대로 유지)
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            if class_name != "person":
                continue
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": round(conf, 2)
            })
        return detections

    def track(self, frame):
        """
        detect()와 달리, 각 사람에게 고유 track_id를 부여해서 추적.
        반환값: [{"track_id": 1, "bbox": (x1,y1,x2,y2), "confidence": 0.87}, ...]
        """
        results = self.model.track(frame, persist=True, verbose=False)[0]

        tracks = []
        if results.boxes.id is None:
            return tracks  # 아직 추적 중인 대상 없음

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            if class_name != "person":
                continue

            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracks.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": round(conf, 2)
            })
        return tracks