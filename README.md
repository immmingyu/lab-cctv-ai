# 연구실 지능형 CCTV

멀티모달 AI 기반 실시간 영상 상황 이해 및 이상행동 추론 시스템 (캡스톤 디자인)

## 핵심 기능
- 낙상 감지 (Pose 기반, 지속 시간에 따른 위험도)
- 제한 구역 접근 감지 (ROI)
- 물건 이동 감지 (노트북, 책)
- 이벤트 스냅샷 + LLM 상황 설명 + 자연어 질문 대시보드

## 실행 방법
1. 패키지 설치: `pip install -r requirements.txt` (PyTorch는 GPU 환경에 맞게 별도 설치)
2. `.env.example`을 복사해 `.env`를 만들고 값 설정
3. 로컬 LLM 사용 시 Ollama 설치 후 `ollama pull exaone3.5:7.8b`
4. 프로젝트 최상위 폴더에서 실행
   - 서버: `uvicorn backend.app:app --reload`
   - AI: `python -m src.main`
   - 대시보드: http://127.0.0.1:8000/