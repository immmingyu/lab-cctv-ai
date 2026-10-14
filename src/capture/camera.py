import cv2

from src.config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


def open_source(source=None):
    """
    source를 안 주면 .env의 CAMERA_INDEX 카메라를 엶 (기본 0번).
    source="data/sample_videos/test.mp4"처럼 경로를 주면 저장된 영상 파일 사용.
    카메라일 때는 해상도를 CAMERA_WIDTH x CAMERA_HEIGHT(기본 640x480)로 고정.
    """
    if source is None:
        source = CAMERA_INDEX
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    if isinstance(source, int):  # 영상 파일이 아니라 카메라일 때만 해상도 고정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


def read_frame(cap, mirror=False):
    """mirror=True로 주면 좌우 반전된 프레임을 반환."""
    ret, frame = cap.read()
    if ret and mirror:
        frame = cv2.flip(frame, 1)  # 1 = 좌우 반전
    return ret, frame


if __name__ == "__main__":
    cap = open_source()
    ret, frame = read_frame(cap)
    if ret:
        h, w = frame.shape[:2]
        print(f"카메라 해상도: {w}x{h}")

    while True:
        ret, frame = read_frame(cap, mirror=True)
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        cv2.imshow("Lab CCTV - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()