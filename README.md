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

## 변경된 설정 (feature/finetune 브랜치, 2026-11-05~)
위 "평가 설정 (확정)"은 2026-11-04 시험 영상 결과의 기록이라 그대로 두고, 이후 바뀐 설정만 여기에 적는다.

| 날짜 | 항목 | 이전 | 변경 | 이유 | 위치 |
|---|---|---|---|---|---|
| 2026-11-05 | LLM에 넘기는 이벤트 시각 표시 | `[11시 2분 30초] ... / 지속 6.1초` | `시각 11:02:30 / ... / 머문 시간 6.1초` (종류별로 머문 시간 / 자세 지속 시간 / 지속 시간), 현재 시각 `16:00`, 시스템 프롬프트에 "시각의 초를 지속 시간으로 읽지 말고, 답할 때는 '11시 2분'처럼 한국어로" 규칙 추가 | 새 질문 T12에서 LLM이 11시 2분 30초의 30초를 머문 시간으로 읽음 | `backend/llm.py` `_event_line`, `TIME_FORMAT_RULE` (예전 형식은 `LEGACY_TIME_FORMAT=True`로 비교 평가 때만) |

LLM 시각 표시 비교 평가: `python -m src.experiments.llm_eval.run_eval --time-format --questions src/experiments/llm_eval/questions.json src/experiments/llm_eval/questions_test.json`

## 물건 검출 파인튜닝 (feature/finetune)
목적: 학교 시험 영상에서 제자리의 책을 거의 검출하지 못한 문제(#23 가만히 3분 동안 0%) 해결.
학습 데이터는 집에서만 찍고, 학교 시험 영상(`eval2_object_school_20261104_135118`)은 평가에만 쓴다 (학습, 기준값 정하기에 쓰지 않음).

### 정해진 방식 (2026-11-05)
| 항목 | 내용 |
|---|---|
| 사람 박스 출처 | Pose 모델(yolov8n-pose)의 사람 박스. 기본 모델과 파인튜닝 모델 모두 같음 (차이가 물건 검출에서만 나오게). 11/4 결과와 잇기 위해 "기본 + yolov8s 사람 박스"는 참고 설정으로만 같이 냄 |
| 클래스 | `book`, `laptop` 2개 (이름 그대로, 이 순서) |
| 책 확신도 기준 | 기본 모델: 0.15 유지. 파인튜닝 모델: 집 검증 데이터에서 F1이 가장 높은 값 → 학교 영상 평가 **전에** 아래 칸에 기록 |
| 파인튜닝 모델 책 확신도 (확정값) | (학습 후 기록) |
| 모델 교체 | `.env`의 `OBJECT_MODEL` 한 줄 (기본 `yolov8s.pt`) |
| 라벨링 도구 | X-AnyLabeling (설치형, 로컬에서만 동작) |

### 촬영 목록 (집)
- 카메라는 학교와 비슷한 높이, 각도 (기준 화면: 학교 물건 세션 `reference_view.jpg`)
- 학교에서 쓴 그 책 **반드시 포함** + 색, 크기가 다른 책 여러 권
- 책 놓인 모양: 눕힘, 세움, 펼침 / 책상 위 여러 위치 / 손에 일부 가려진 장면
- 화면에서 **작게** 보이는 책 (카메라에서 먼 곳)
- 노트북 (열림, 닫힘)
- 책이 없는 장면: 빈 책상 + **비슷한 물건만 있는 장면** (노트, 종이, 태블릿, 휴대폰) — 이런 장면은 라벨 없이 그대로 둠 (틀리게 찾지 않도록 배우는 용도)
- 촬영은 **2~3개 세션**으로 나눔 (예: 낮, 밤, 다른 방). 학습/검증은 **세션 단위로** 나눔 (같은 세션의 프레임이 양쪽에 섞이지 않게)

### 라벨링 규칙 (학습 데이터와 학교 평가 데이터에 똑같이 적용)
1. 책(또는 노트북)이 **30% 이상 보이면** 라벨을 단다. 30% 미만으로 보이면 달지 않는다.
2. 박스는 **보이는 부분만** 감싼다 (가려진 부분을 짐작해서 넓히지 않음).
3. 책 여러 권이 겹쳐 쌓여 있으면 보이는 책마다 따로 단다.
4. 노트, 종이, 태블릿, 휴대폰은 책이 아니므로 라벨을 달지 않는다.
5. 애매하면 메모(`labels/notes.txt`)에 프레임 이름과 이유를 적고, 같은 판단을 끝까지 유지한다.

### 폴더 구조 (`data/finetune/`, git에 올리지 않음 — 사람 얼굴 포함)
```
data/finetune/
  frames/<녹화이름>/          # extract_frames가 뽑은 프레임 (+ X-AnyLabeling 라벨 .txt, classes.txt)
  dataset/                     # build_dataset가 만든 학습용 (YOLO 형식)
    images/train, images/val
    labels/train, labels/val
    dataset.yaml
  school_test/<녹화이름>/      # 학교 평가용 프레임과 라벨 (학습에 쓰지 않음)
runs/finetune/<이름>/weights/best.pt   # 학습 결과
```

### 순서
1. 집에서 세션별로 녹화: `python -m src.tools.recorder --name ft_home_s1_day` (세션마다 이름을 다르게)
2. 프레임 뽑기: `python -m src.tools.extract_frames data/recordings/<녹화이름> --every 1.5`
3. X-AnyLabeling으로 `data/finetune/frames/<녹화이름>/` 라벨링 (YOLO 형식, 클래스 순서 book, laptop)
4. 데이터셋 만들기: `python -m src.tools.build_dataset --train <세션1> <세션2> --val <세션3>`
5. 학습: `yolo detect train model=yolov8s.pt data=data/finetune/dataset/dataset.yaml epochs=100 imgsz=640 batch=16 device=0 patience=20 project=runs/finetune name=book_laptop_v1`
