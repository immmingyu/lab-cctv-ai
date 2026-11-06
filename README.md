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
| 2026-11-05 | 물건 이동 '사람 가림'의 사람 박스 출처 | 물건 검출 모델(yolov8s)의 person | Pose 모델(yolov8n-pose)의 사람 박스. 실시간 `--only object`에서도 Pose를 띄우되 화면에는 사람을 그리지 않음 | 책/노트북만 파인튜닝한 모델은 person을 못 찾음. 기본/파인튜닝 비교에서 차이가 물건 검출에서만 나오게 통일 | `src/main.py`, `replay_eval.py` `OBJECT_VARIANTS` (11/4 방식은 `[참고] ... yolov8s 사람 박스` 설정으로 남김) |
| 2026-11-05 | 물건 검출 모델 교체 | 코드에 `yolov8s.pt`, 책 0.15 고정 | `.env`의 `OBJECT_MODEL` 한 줄. 확신도 기준은 모델 weights 폴더의 `thresholds.json` (없으면 book 0.15) | 기본/파인튜닝 모델을 한 줄로 바꿔 끼우기 | `src/config.py`, `src/detection/object_detector.py` `conf_thresholds_for` |
| 2026-11-05 | 재생 평가 `--groups` | 범위(`a-b`)만 | 번호 하나(`idle=23`)도 됨, 영상 처리 전에 먼저 검사 | 11/4 평가가 영상 처리 후 채점 단계에서 멈춤 | `replay_eval.py` `parse_groups` |

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
| 파인튜닝 모델 책 확신도 (확정값) | **0.78** (`book_laptop_v1`, 세션 2 F1 0.7758 / 정밀도 0.860 / 재현율 0.706). 2026-11-06 학교 평가 전에 확정. weights 폴더 `thresholds.json`에 저장 |
| 기본 모델 책 확신도, 다시 고른 값 (확정값, 비교용) | **0.14** (yolov8s, 세션 2 F1 0.6075 / 정밀도 0.533 / 재현율 0.706. 참고로 0.15의 F1은 0.6004). 2026-11-06 학교 평가 전에 확정. 시스템 기본값은 0.15 유지 |
| 모델 교체 | `.env`의 `OBJECT_MODEL` 한 줄 (기본 `yolov8s.pt`) |
| 라벨링 도구 | X-AnyLabeling (설치형, 로컬에서만 동작) |

### 세션 계획 (2026-11-05 확정 → 2026-11-06 변경: 세션 2, 3 촬영 결과를 보기 전에 정함. 학습/검증은 세션 단위)
| 세션 | 장소 | 녹화 | 내용 | 용도 |
|---|---|---|---|---|
| 1 | 집 | `eval_object_home_20261103_144117` (기존), 프레임 폴더 `eval_object_home_20261103_144117_v3` | 책 이동, 손으로 가리기 장면이 많음 | 학습 |
| 2 | 거실 | `ft_home_s2_living_20261106_185234` | 감시 대상 책(학교에서 쓴 책) 포함. Mac북으로 녹화해서 **노트북은 화면에 고정된 상태로만** 나옴 | **검증** (책 확신도 F1 기준도 여기서) |
| 3 | 방 | `ft_home_s3_room_20261106_191008` | 감시 대상 책(학교에서 쓴 책) 포함. **노트북을 다양하게** (열림, 닫힘, 위치, 각도) | 학습 |

- 학습: 세션 1, 3 / 검증: 세션 2 (변경 없음)
- 감시 대상 책 = 학교 시험 영상에서 쓴 책. 세션 2, 3 모두에 넣는다.
- 세션 2의 노트북은 한 자리에 고정이라, 검증 데이터의 노트북 결과는 다양한 모습을 대표하지 못한다 (참고로만 본다). 확신도 기준을 정하는 대상은 책뿐이라 영향이 없다.

### 공정 비교 계획 (2026-11-06 추가, 학교 결과를 보기 전에 정함)
"기준값을 다시 고른 효과"와 "파인튜닝 효과"를 나눠 보려고, 기본 모델도 파인튜닝 모델과 **같은 방법**으로 책 확신도를 다시 고른다.

| 비교 | 모델 | 책 확신도 | 무엇을 보나 |
|---|---|---|---|
| A. 기준선 | yolov8s | 0.15 (지금 시스템, 이미 측정) | 출발점 |
| B. 기본 모델 + 다시 고른 기준 | yolov8s | 세션 2에서 F1이 가장 높은 값 | A→B = 기준값을 다시 고른 효과 |
| C. 파인튜닝 모델 | 파인튜닝 | 세션 2에서 F1이 가장 높은 값 | B→C = 파인튜닝 효과 |

- B와 C 모두 세션 2(집 거실, 검증)만 보고 고른다. 학교 영상은 기준값 고르기에 쓰지 않는다.
- 기본 모델 값은 **세션 2 라벨링이 끝나면 바로**, 학교 결과를 보기 전에 위 "정해진 방식" 표에 확정 기록한다.
- B, C 모두 학교 `detect_eval`과 학교 물건 재생 평가(`replay_eval`)를 한다.
- 시스템 기본값은 **0.15 그대로** 둔다. B의 값은 비교용으로 명령에서 직접 넘긴다. `yolov8s.pt` 옆에 `thresholds.json`을 두면 실시간 시스템 값까지 바뀌므로 두지 않는다.

- `eval_people_home_p1_20261103_150718`은 **제외**: 책이 화면에 없음 (물건 영상의 책 자리가 비어 있고, 기본 모델이 화면 오른쪽 끝에서 '책'으로 찾은 6.5%는 책이 아닌 다른 물건)
- 집 영상이 학습에 들어가므로, 파인튜닝 모델로 집 영상을 재생 평가한 결과는 **참고용(학습에 쓴 장소)** 으로 자동 표시 (결과 폴더 이름 `_ref_trainplace`, notes.txt). 성능 비교는 학교 시험 영상으로만 한다.
- 11/3 집 영상으로 낸 기본 모델 결과(개발용 확인)는 그대로 개발용 기록으로 둔다.

### 촬영 목록 (집)
- 카메라는 학교와 비슷한 높이, 각도 (기준 화면: 학교 물건 세션 `reference_view.jpg`)
- 학교에서 쓴 그 책 **반드시 포함** + 색, 크기가 다른 책 여러 권
- 책 놓인 모양: 눕힘, 세움, 펼침 / 책상 위 여러 위치 / 손에 일부 가려진 장면
- 화면에서 **작게** 보이는 책 (카메라에서 먼 곳)
- 노트북 (열림, 닫힘)
- 책이 없는 장면: 빈 책상 + **비슷한 물건만 있는 장면** (노트, 종이, 태블릿, 휴대폰) — 이런 장면은 박스 없이 **passed로 확인 표시** (틀리게 찾지 않도록 배우는 용도, 확인 표시가 없으면 학습에서 빠짐)
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
  frames/<세션 폴더>/          # extract_frames가 뽑은 프레임 + X-AnyLabeling .json (확인한 이미지에만 생김)
  labels/<세션 폴더>/          # X-AnyLabeling YOLO 내보내기 .txt (안 본 이미지도 빈 .txt로 생김)
  dataset/                     # build_dataset가 만든 학습용 (YOLO 형식)
    images/train, images/val
    labels/train, labels/val
    dataset.yaml
  test_school/                 # 학교 평가용 (학습에 절대 쓰지 않음, build_dataset가 이 경로를 거부)
    frames/<녹화이름>/         #   5초 간격, 건너뛰기 없음 (가만히 둔 구간 포함) + classes.txt
    labels/<녹화이름>/         #   X-AnyLabeling YOLO 내보내기 위치
runs/finetune/<이름>/weights/best.pt   # 학습 결과
```

### 순서
1. 집에서 세션별로 녹화: `python -m src.tools.recorder --name ft_home_s2_living` / `--name ft_home_s3_room` (폴더 이름 뒤에 날짜_시각이 붙음)
2. 프레임 뽑기: `python -m src.tools.extract_frames data/recordings/<녹화이름> --every <간격> --min-change 3` (비슷한 장면 건너뛰기 켬. 간격은 세션 길이에 맞춰 조정)

   | 세션 | 녹화 길이 | 간격 | 장수 |
   |---|---|---|---|
   | 1 `eval_object_home_20261103_144117_v3` | 1273.6초 | 4초 (프레임 간격으로 추정) | 305 |
   | 2 `ft_home_s2_living_20261106_185234` | 544.1초 | 3초 | 118 |
   | 3 `ft_home_s3_room_20261106_191008` | 662.6초 | 1.75초 | 165 |
3. X-AnyLabeling으로 `data/finetune/frames/<세션 폴더>/` 라벨링 (클래스 이름 book, laptop)
   - 책/노트북이 있으면 사각형으로 라벨, 없으면 박스 없이 **passed** — 둘 다 그 이미지에 .json이 생겨 "확인한 이미지"가 됨
   - **.json이 없는 이미지는 안 본 것으로 보고 학습에서 뺀다** (내보내기가 만든 빈 .txt가 있어도)
   - 라벨링이 **다 끝나면 마지막에 한 번** YOLO 형식으로 내보내기 → `data/finetune/labels/<세션 폴더>/`
     (중간에 내보낸 뒤 라벨을 고쳤으면 다시 내보내기. build_dataset이 .json 박스와 .txt 줄이 다르면 그 이미지를 빼고 알려줌)
4. 데이터셋 만들기: `python -m src.tools.build_dataset --train eval_object_home_20261103_144117_v3 ft_home_s3_room_20261106_191008 --val ft_home_s2_living_20261106_185234`
   - 세션별로 확인한 장수, 책 있는 장수, 책 없는 장수, 빠진 장수(안 봄, 라벨 오류)를 출력
5. 학습: `yolo detect train model=yolov8s.pt data=data/finetune/dataset/dataset.yaml epochs=100 imgsz=640 batch=16 device=0 patience=20 project=runs/finetune name=book_laptop_v1`
   - ultralytics 8.4는 결과를 `runs/detect/runs/finetune/<이름>`에 저장한다. 학습 뒤 `runs/finetune/<이름>`으로 옮긴다.
6. 책 확신도 고르기 (기본 모델, 파인튜닝 모델 같은 방법, 세션 2만 사용):
   `python -m src.experiments.pick_threshold --model yolov8s.pt`,
   `python -m src.experiments.pick_threshold --model runs/finetune/<이름>/weights/best.pt --write`
   - 반전해서 검출, IoU 0.5, 기준 0.01~0.95 중 F1 최고 (같으면 낮은 기준). 결과는 `data/threshold_select/`
   - `--write`는 파인튜닝 모델에만 동작한다 (`thresholds.json`). 기본 모델은 거부한다.

### 학습 기록: `book_laptop_v1` (2026-11-06)
| 항목 | 내용 |
|---|---|
| 데이터 | 학습: 세션 1(305장) + 세션 3(165장) = 470장 / 검증: 세션 2(118장, 책 박스 218개, 노트북 118개) |
| 학습 | 36에폭에서 조기 종료 (patience 20), best.pt = 16에폭 |
| ultralytics 검증 (반전 없음) | book 정밀도 0.913, 재현율 0.862, mAP50 0.901 / laptop 재현율 0 (세션 2 노트북은 화면에 고정된 Mac북 하나라 참고만) |
| 세션 2, 시스템 조건(반전) 0.78 | 맞힘 154, 잘못 찾음 25, 놓침 64. 같은 기준에서 반전하지 않으면 잘못 찾음 5 → **반전한 화면에서 정답과 안 겹치는 박스가 늘어남** (규칙은 시스템과 같은 반전 조건으로 고정했으므로 그대로 씀) |
| 검증 세션 특이점 | 세션 2에는 책 없는 이미지가 0장. 잘못 찾은 박스는 책이 있는 이미지 안에서만 셈 |

비교 B, C의 명령 (학교):
```
python -m src.experiments.detect_eval --conf 0.14 --tag base_f1                                          # B
python -m src.experiments.detect_eval --model runs/finetune/book_laptop_v1/weights/best.pt --tag ft_v1      # C (thresholds.json 0.78)
python -m src.experiments.replay_eval data/recordings/eval2_object_school_20261104_135118 --task object --book-conf 0.14 --tag base_f1 --groups "desk=1-5 out=6-10 carry=11-15 touch=16-19 cover=20-22 idle=23 pass=24-26"
python -m src.experiments.replay_eval data/recordings/eval2_object_school_20261104_135118 --task object --object-model runs/finetune/book_laptop_v1/weights/best.pt --tag ft_v1 --groups "desk=1-5 out=6-10 carry=11-15 touch=16-19 cover=20-22 idle=23 pass=24-26"
```

### 학교 검출 평가 (`detect_eval`, 책만)
손으로 단 학교 정답 라벨(`test_school`, 169장 중 책 있는 이미지 158장)로 모델의 책 검출을 채점한다. 노트북은 학교 영상에 없어서 채점하지 않는다.
```
python -m src.experiments.detect_eval --tag baseline                                          # 기본 모델 (기준선)
python -m src.experiments.detect_eval --model runs/finetune/<이름>/weights/best.pt --tag <이름>   # 파인튜닝 모델
```
- 시스템과 같은 조건:
  - 화면을 좌우 반전해서 검출하고, 박스를 원본 좌표로 되돌려 정답과 비교한다.
  - 책 확신도는 모델 옆 `thresholds.json` 값을 쓰고, 없으면 0.15를 쓴다.
- 맞음 기준: 정답과 IoU 0.5 이상. 정답 하나에는 박스 하나만 짝짓고, 남은 박스는 "잘못 찾은 박스"로 센다.
- 두 가지로 센다:
  - **시스템**: 프레임마다 확신도가 가장 높은 책 1개만 본다. 물건 이동 판단이 실제로 쓰는 박스다.
  - **모든 박스**: 기준을 넘은 책 박스를 모두 본다.
- 단계 구분 (marks.csv 기준):
  - 이동: 첫 정상 시도(460.1초) 전, 92장
  - 정상 동작: 그 뒤 #23 제외, 40장
  - 가만히: #23, 37장
- 결과는 `data/detect_eval/<녹화>_<모델>_<tag>/`에 저장한다(summary.csv, per_image.csv, info.json). 같은 이름이 있으면 덮어쓰지 않는다.

**기준선(기본 모델)**: yolov8s, 책 0.15, IoU 0.5, 반전 켬. 커밋 시점의 결과.

| 단계 | 이미지 | 정답 책 | 재현율 (시스템 = 모든 박스) | 잘못 찾은 박스 |
|---|---|---|---|---|
| 이동 | 92 | 82 | 42/82 = 0.512 | 0 |
| 정상 동작 | 40 | 39 | 1/39 = 0.026 | 0 |
| 가만히 3분 | 37 | 37 | **0/37 = 0.0** | 0 |
| 전체 | 169 | 158 | 43/158 = 0.272 | 0 |
