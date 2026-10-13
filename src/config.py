import os
from dotenv import load_dotenv

load_dotenv()

# 컴퓨터마다 다른 값은 .env에서 읽음 (없으면 기본값)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE") or None   # None이면 자동 선택 (Windows는 CUDA)