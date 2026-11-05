import os
from dotenv import load_dotenv

load_dotenv()

# 컴퓨터마다 다른 값은 .env에서 읽음 (없으면 기본값)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE") or None   # None이면 자동 선택 (Windows는 CUDA)

# 조건부 VLM (일직선 눕기 재확인). 사진이 외부로 나가지 않게 로컬 Ollama 사용
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl:4b-instruct")

# 물건 검출 모델 (기본 yolov8s.pt, 파인튜닝: runs/finetune/<이름>/weights/best.pt)
OBJECT_MODEL = os.getenv("OBJECT_MODEL") or "yolov8s.pt"
