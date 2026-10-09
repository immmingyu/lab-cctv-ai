import cv2
from src.capture.camera import open_source, read_frame

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"클릭한 좌표: ({x}, {y})")

def main():
    cap = open_source(0)
    cv2.namedWindow("ROI Picker")
    cv2.setMouseCallback("ROI Picker", on_mouse_click)

    while True:
        ret, frame = read_frame(cap, mirror=True)
        if not ret:
            break

        cv2.imshow("ROI Picker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()