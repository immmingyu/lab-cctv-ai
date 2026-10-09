from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, target_classes, model_path="yolov8s.pt",
                 conf_thresholds=None, default_conf=0.4):
        """
        conf_thresholds: 물건별 확신도 기준. 예: {"book": 0.15}
        지정 안 한 물건은 default_conf 사용.
        model_path: 물건 검출은 yolov8s(small)로 변경 - 작은 물건 인식이 n보다 낫고,
                    RTX 3060이면 속도 부담 거의 없음.
        """
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.conf_thresholds = conf_thresholds or {}
        self.default_conf = default_conf

    def detect(self, frame):
        # ※ 중요: YOLO 자체 기본 기준이 0.25라서, 그보다 낮은 기준을 쓰려면
        #   모델 호출 시 conf를 가장 낮은 값으로 내려놔야 함
        min_conf = min([self.default_conf] + list(self.conf_thresholds.values()))
        results = self.model(frame, conf=min_conf, verbose=False)[0]

        found = {cls: None for cls in self.target_classes}
        best_conf = {cls: 0.0 for cls in self.target_classes}

        for box in results.boxes:
            class_name = self.model.names[int(box.cls[0])]
            if class_name not in self.target_classes:
                continue

            conf = float(box.conf[0])
            threshold = self.conf_thresholds.get(class_name, self.default_conf)
            if conf < threshold:
                continue

            if conf > best_conf[class_name]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                found[class_name] = (x1, y1, x2, y2)
                best_conf[class_name] = conf

        return found