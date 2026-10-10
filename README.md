# 연구실 지능형 CCTV

멀티모달 AI 기반 실시간 영상 상황 이해 및 이상행동 추론 시스템 (캡스톤 디자인)

## 핵심 기능
- 낙상 감지 (Pose 기반, 지속 시간에 따른 위험도)
- 제한 구역 접근 감지 (ROI)
- 물건 이동 감지 (노트북, 책)
- 이벤트 스냅샷 + LLM 상황 설명 + 자연어 질문 대시보드

## 실행 방법

### 1. 환경 만들기 (Python 3.12)
- `conda create -n lab-cctv python=3.12 -y`
- `conda activate lab-cctv`
- **PyTorch GPU 버전으로 설치** (PyTorch 공식 홈페이지에서 본인 CUDA 버전 확인)
  - 예: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- `python -m pip install -r requirements.txt`
- CUDA 인식 확인: `python -c "import torch; print(torch.cuda.is_available())"` -> `True` 출력 확인

### 2. 환경 변수 설정
- `.env.example`을 복사하여 `.env`를 만들고 값 설정

### 3. 로컬 LLM 사용 시
- Ollama 설치 후: `ollama pull exaone3.5:7.8b`

### 4. 프로젝트 루트 폴더에서 실행
- **서버:** `uvicorn backend.app:app --reload`
- **AI:** `python -m src.main`
- **대시보드:** `http://127.0.0.1:8000/`