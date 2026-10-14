import cv2

# 0~4번 카메라를 차례로 열어보고, 화면을 띄워서 어떤 카메라인지 눈으로 확인
for idx in range(5):
    cap = cv2.VideoCapture(idx)
    ret, frame = cap.read() if cap.isOpened() else (False, None)
    if not ret:
        print(f"{idx}번: 없음")
        cap.release()
        continue
    h, w = frame.shape[:2]
    print(f"{idx}번: 열림 ({w}x{h}) - 화면 확인 후 아무 키나 누르세요")
    cv2.imshow(f"Camera {idx}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()