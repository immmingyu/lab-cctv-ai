import cv2

def open_source(source=0):
    """
    source=0 이면 웹캠(C920E) 사용.
    source="data/sample_videos/test.mp4" 처럼 경로를 주면 저장된 영상 파일 사용.
    -> 카메라가 없거나 문제 있을 때도 같은 코드로 테스트 가능하게 하기 위한 함수.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    return cap

def read_frame(cap, mirror=False):
    """
    mirror=True로 주면 좌우 반전된 프레임을 반환.
    """
    ret, frame = cap.read()
    if ret and mirror:
        frame = cv2.flip(frame, 1)  # 1 = 좌우 반전
    return ret, frame

if __name__ == "__main__":
    cap = open_source(0)  # C920E가 연결되어 있어야 함

    while True:
        ret, frame = read_frame(cap, mirror=True)  # 여기 변경됨

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        cv2.imshow("Lab CCTV - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()