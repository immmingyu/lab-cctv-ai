import json
from pathlib import Path

from ultralytics import YOLO
from src.config import OBJECT_MODEL, YOLO_DEVICE

# 기본 모델(yolov8s)의 확신도 기준. 모델 파일 옆에 thresholds.json이 있으면 그 값을 씀
DEFAULT_CONF_THRESHOLDS = {"book": 0.15}


def conf_thresholds_for(model_path):
    """
    모델에 맞는 물건별 확신도 기준.
    파인튜닝 모델은 weights 폴더에 thresholds.json (예: {"book": 0.42})을 두면 그 값을 씀
    -> .env의 OBJECT_MODEL 한 줄만 바꿔도 기준값이 함께 따라감.
    """
    f = Path(model_path).with_name("thresholds.json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return dict(DEFAULT_CONF_THRESHOLDS)


class ObjectDetector:
    def __init__(self, target_classes, model_path=None,
                 conf_thresholds=None, default_conf=0.4, person_conf=0.4):
        """
        model_path: 안 넘기면 .env의 OBJECT_MODEL (기본 yolov8s.pt).
                    yolov8s(small) - 작은 물건 인식이 n보다 낫고, RTX 3060이면 속도 부담 거의 없음.
        conf_thresholds: 물건별 확신도 기준. 안 넘기면 conf_thresholds_for(model_path).
                         지정 안 한 물건은 default_conf 사용.
        person_conf: detect_all()에서 사람 박스로 인정할 확신도 기준
                     (파인튜닝 모델처럼 person 클래스가 없으면 사람 박스는 항상 빈 목록)
        """
        self.model_path = model_path or OBJECT_MODEL
        self.model = YOLO(self.model_path)
        names = set(self.model.names.values())
        missing = [c for c in target_classes if c not in names]
        if missing:
            raise SystemExit(f"물건 검출 모델({self.model_path})에 없는 클래스: {missing}")
        self.has_person = "person" in names
        self.target_classes = target_classes
        self.conf_thresholds = (conf_thresholds if conf_thresholds is not None
                                else conf_thresholds_for(self.model_path))
        self.default_conf = default_conf
        self.person_conf = person_conf

    def detect(self, frame):
        """물건만 필요할 때. 반환값: {"book": bbox 또는 None, ...}"""
        return self.detect_all(frame)[0]

    def detect_all(self, frame):
        """
        같은 YOLO 결과에서 물건과 사람 박스를 함께 꺼냄 (모델 추가 없음).
        ※ 물건 이동의 '사람 가림' 판단에는 Pose 모델의 사람 박스를 씀 (README 파인튜닝 절).
          여기 사람 박스는 11/4 결과와 비교하는 참고용.
        반환값: ({"book": bbox 또는 None, ...}, [사람 bbox, ...])
        """
        # ※ 중요: YOLO 자체 기본 기준이 0.25라서, 그보다 낮은 기준을 쓰려면
        #   모델 호출 시 conf를 가장 낮은 값으로 내려놔야 함
        min_conf = min([self.default_conf] + list(self.conf_thresholds.values()))
        results = self.model(frame, conf=min_conf, verbose=False, device=YOLO_DEVICE)[0]

        found = {cls: None for cls in self.target_classes}
        best_conf = {cls: 0.0 for cls in self.target_classes}
        persons = []

        for box in results.boxes:
            class_name = self.model.names[int(box.cls[0])]
            if class_name == "person":
                if float(box.conf[0]) >= self.person_conf:
                    persons.append(tuple(map(int, box.xyxy[0])))
                continue
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

        return found, persons