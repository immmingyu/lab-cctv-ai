#!/usr/bin/env bash
# 연구실 지능형 CCTV 데모 한 번에 실행 (Mac / Linux)
#
# 순서: Ollama 확인 -> 백엔드(uvicorn) 실행 -> AI(main.py) 실행 -> 브라우저 열기
# Ctrl+C 를 누르거나 AI가 끝나면, 이 스크립트가 띄운 프로세스만 같이 정리한다.
#
# 사용법 (conda 환경을 먼저 켜고):
#   conda activate lab-cctv
#   bash scripts/start_demo.sh
# 옵션:
#   --port 8000    백엔드 포트
#   --no-browser   브라우저를 열지 않음 (SSH 접속처럼 화면이 없을 때)
#   --no-vlm       AI를 규칙만으로 실행 (python -m src.main --no-vlm)

set -u

PORT=8000
OPEN_BROWSER=1
VLM_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-browser) OPEN_BROWSER=0; shift ;;
    --no-vlm) VLM_ARG="--no-vlm"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "모르는 옵션입니다: $1"; exit 1 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
BASE_URL="http://127.0.0.1:${PORT}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# 이 스크립트가 직접 띄운 프로세스만 담는다. (원래 켜져 있던 건 건드리지 않음)
STARTED_PIDS=()
STARTED_NAMES=()

step() { printf '\n[단계] %s\n' "$1"; }
ok()   { printf '  OK  %s\n' "$1"; }
info() { printf '      %s\n' "$1"; }

cleanup() {
  if [ ${#STARTED_PIDS[@]} -eq 0 ]; then return; fi
  printf '\n[정리] 이 스크립트가 띄운 프로세스를 종료합니다.\n'
  local i
  for i in "${!STARTED_PIDS[@]}"; do
    local pid="${STARTED_PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      info "${STARTED_NAMES[$i]} 종료 (PID $pid)"
    fi
  done
  sleep 1
  for i in "${!STARTED_PIDS[@]}"; do
    local pid="${STARTED_PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null; fi
  done
  STARTED_PIDS=()
}
trap cleanup EXIT INT TERM

fail() {  # fail <어디서> <원인> <해결>
  printf '\n[실패] %s 에서 막혔습니다.\n' "$1"
  printf '  원인: %s\n' "$2"
  [ -n "${3:-}" ] && printf '  해결: %s\n' "$3"
  exit 1
}

remember() { STARTED_PIDS+=("$1"); STARTED_NAMES+=("$2"); }

url_ok() { curl -fsS -m "${2:-3}" "$1" >/dev/null 2>&1; }

# wait_url <주소> <최대 초> [지켜볼 PID]
wait_url() {
  local url="$1" secs="$2" watch="${3:-}" waited=0
  while [ "$waited" -lt "$secs" ]; do
    if [ -n "$watch" ] && ! kill -0 "$watch" 2>/dev/null; then return 1; fi
    if url_ok "$url" 2; then return 0; fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

printf '연구실 지능형 CCTV 데모 시작\n'
info "작업 폴더: $PROJECT_ROOT"

# ---------- 0) 파이썬 환경 ----------
step "파이썬 환경 확인"
command -v python >/dev/null 2>&1 || fail "파이썬 확인" "python 명령을 찾을 수 없습니다." "conda activate lab-cctv 를 먼저 실행한 뒤 다시 시도하세요."
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "lab-cctv" ]; then
  info "주의: 현재 conda 환경이 '$CONDA_DEFAULT_ENV' 입니다 (예상: lab-cctv)"
fi
python -c "import fastapi, uvicorn, cv2, ultralytics" >/dev/null 2>&1 \
  || fail "파이썬 환경 확인" "필요한 패키지(fastapi/uvicorn/opencv/ultralytics)를 불러오지 못했습니다." \
          "conda activate lab-cctv 후 python -m pip install -r requirements.txt 를 실행하세요."
ok "python 과 필요한 패키지 확인됨"

# ---------- 1) Ollama ----------
step "Ollama 확인 (LLM 요약/질문용)"
if url_ok "$OLLAMA_URL/api/tags"; then
  ok "Ollama 가 이미 켜져 있습니다 ($OLLAMA_URL)"
else
  command -v ollama >/dev/null 2>&1 \
    || fail "Ollama 확인" "$OLLAMA_URL 에 연결되지 않고, ollama 명령도 없습니다." \
            "https://ollama.com 에서 Ollama 를 설치하거나, 이미 설치했다면 'ollama serve' 를 먼저 실행하세요."
  info "Ollama 가 꺼져 있어 'ollama serve' 로 켭니다."
  ollama serve >/dev/null 2>&1 &
  remember "$!" "Ollama"
  wait_url "$OLLAMA_URL/api/tags" 20 "$!" \
    || fail "Ollama 시작" "20초 안에 $OLLAMA_URL 가 응답하지 않았습니다." \
            "터미널을 따로 열어 'ollama serve' 를 실행하고 오류 메시지를 확인하세요."
  ok "Ollama 시작됨"
fi

# 모델이 받아져 있는지 확인 (없으면 멈추지 않고 알려만 줌)
TAGS="$(curl -fsS -m 5 "$OLLAMA_URL/api/tags" 2>/dev/null || true)"
for NEED in "exaone3.5:7.8b" "qwen3-vl:4b-instruct"; do
  case "$TAGS" in
    *"$NEED"*) ;;
    *) info "주의: 모델 '$NEED' 이(가) 없습니다. 필요하면 'ollama pull $NEED' 를 실행하세요." ;;
  esac
done

# ---------- 2) 백엔드 ----------
step "백엔드(uvicorn) 실행"
if url_ok "$BASE_URL/health"; then
  ok "백엔드가 이미 $BASE_URL 에서 돌고 있습니다 (그대로 사용)"
else
  python -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT" &
  BACKEND_PID=$!
  remember "$BACKEND_PID" "백엔드(uvicorn)"
  wait_url "$BASE_URL/health" 40 "$BACKEND_PID" \
    || fail "백엔드 실행" "40초 안에 $BASE_URL/health 가 응답하지 않았습니다 (포트 $PORT 가 이미 쓰이는 중이거나 서버가 바로 종료됨)." \
            "위에 찍힌 uvicorn 오류 메시지를 확인하거나, --port 9000 처럼 다른 포트로 다시 실행하세요."
  ok "백엔드 준비됨 ($BASE_URL)"
fi

# ---------- 3) AI ----------
step "AI(main.py) 실행"
if [ -n "$VLM_ARG" ]; then
  python -m src.main "$VLM_ARG" &
else
  python -m src.main &
fi
AI_PID=$!
remember "$AI_PID" "AI(main.py)"
sleep 5
kill -0 "$AI_PID" 2>/dev/null \
  || fail "AI 실행" "main.py 가 시작하자마자 종료됐습니다." \
          "카메라가 연결되어 있는지, .env 의 CAMERA_INDEX 값이 맞는지 확인하세요. 화면 창을 띄울 수 없는 환경(SSH 등)에서도 종료될 수 있습니다."
ok "AI 실행 중 (PID $AI_PID)"

# ---------- 4) 대시보드 ----------
step "대시보드 열기"
if [ "$OPEN_BROWSER" -eq 0 ]; then
  info "--no-browser 옵션이라 열지 않습니다. 주소: $BASE_URL/"
elif command -v open >/dev/null 2>&1; then
  open "$BASE_URL/" && ok "브라우저에서 $BASE_URL/ 를 열었습니다"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$BASE_URL/" >/dev/null 2>&1 && ok "브라우저에서 $BASE_URL/ 를 열었습니다"
else
  info "브라우저를 열 명령이 없습니다. 직접 $BASE_URL/ 로 접속하세요."
fi

printf '\n준비 끝. 이 창에서 Ctrl+C 를 누르면 전부 종료됩니다.\n'
wait "$AI_PID"
printf '\nAI(main.py)가 종료됐습니다.\n'
