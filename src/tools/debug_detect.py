import cv2
from ultralytics import YOLO
from src.capture.camera import open_source, read_frame

def main():
    model = YOLO("yolov8n.pt")
    cap = open_source()

    while True:
        ret, frame = read_frame(cap, mirror=True)
        if not ret:
            break

        # conf=0.1: 확신도가 낮은 검출도 전부 보여줌 (디버깅용)
        results = model(frame, conf=0.1, verbose=False)[0]

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("Debug Detect", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()