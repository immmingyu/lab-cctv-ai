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

### 5. 한 번에 실행 (데모용)
위 3개를 순서대로 띄우고 브라우저까지 여는 스크립트. 중간에 실패하면 어디서 막혔는지 알려주고, 창을 닫으면 같이 띄운 프로세스를 정리한다.
- Windows: `powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1`
- Mac/Linux: `bash scripts/start_demo.sh`
- 화면이 없는 환경(SSH 등)에서는 `-NoBrowser` / `--no-browser` 사용

## 평가 설정 (확정: 2026-11-03, 새 영상 검증 전 고정)
2026-11-03 녹화 영상은 개발용으로만 쓰고, 아래 값은 새 영상 결과를 보고 다시 바꾸지 않는다.

| 항목 | 값 | 위치 |
|---|---|---|
| VLM 질문 문구 | 이 사람이 누워 있거나 쓰러진 자세입니까? 장소(바닥, 매트 등)와 상관없이 자세만 보고 예/아니오로 답하고 근거를 짧게 적어주세요. | `src/detection/vlm_client.py` `QUESTION` |
| VLM 호출: 정지 시간 | 8초 | `StillnessTrigger(still_seconds=8)` |
| VLM 호출: 정지 기준 | 관절 평균 이동이 사람 박스 긴 변의 5% 이하 | `StillnessTrigger(motion_ratio=0.05)` |
| VLM 호출: 튐 무시 | 기준을 넘는 움직임이 연속 3프레임 이상일 때만 정지 시간 초기화 | `StillnessTrigger(reset_frames=3)` |
| 화면 가장자리 보류 | 사람 박스가 상하좌우 5픽셀 이내에 닿은 프레임은 **누움 쪽 판단만 보류**: '누움' 투표와 새 경보 발생을 막고, '서 있음' 투표와 회복(서 있음 3초 → 사건 종료)은 인정. VLM은 호출하지 않음 (규칙만 / 규칙+VLM 공통). 화면에 EDGE 표시 | `src/events/fall_detector.py` `EDGE_MARGIN`, `FallDetector.update(hold=)` |
| VLM 모델 | 로컬 Ollama `qwen3-vl:4b-instruct` (`.env`의 `VLM_MODEL`) | `src/config.py` |

가장자리 처리 변경 (2026-11-03, B안): 처음에는 가장자리 프레임의 모든 자세 판단을 보류했으나, 서 있는 사람은 화면 위아래 끝에 닿기 쉬워 일어선 프레임까지 보류되면서 자세 상태가 '누움'으로 멈추고 회복이 인정되지 않았다(개발용 영상에서 옆 눕기 여러 시도가 한 사건으로 묶임). 잘린 몸 때문에 생기는 낙상 오경보를 막는다는 목적은 누움 쪽만 보류해도 지켜지므로 B안으로 확정. 5픽셀 값은 그대로 유지.

재생 평가 (사람):
`python -m src.experiments.replay_eval <녹화 폴더> --task people --vlm --roi "..." --groups "..."`
결과는 `data/replay_results/`에 저장 (`data/recordings`는 읽기만).

## LLM 답변 검증 (평가용)
LLM 답변에서 시각과 건수를 뽑아 실제로 넘겨준 이벤트 기록과 대조한다. 맞지 않으면 한 번 다시 생성하고, 그래도 맞지 않으면 코드가 기록만으로 만든 목록 문장으로 대체한다. (`backend/verify.py`, 평가용 옵션이라 대시보드 기본값은 꺼짐)

```
python -m src.experiments.llm_eval.build_eval_db
python -m src.experiments.llm_eval.run_eval --questions src/experiments/llm_eval/questions_test.json --verify
python -m src.experiments.llm_eval.score_eval <run 폴더>   # 블라인드 채점 후
python -m tests.test_verify                                 # 검증 장치 자체 테스트 (LLM 불필요)
```
`questions.json`(23개)은 문제를 찾아낸 개발용, `questions_test.json`(12개)은 검증 장치 비교용.
