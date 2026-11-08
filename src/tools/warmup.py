"""
VLM과 LLM을 한 번씩 미리 불러두기 (전시회용 준비 호출).

Ollama는 모델을 처음 쓸 때 GPU에 올리느라 오래 걸려서 첫 질문이 시간 초과(VLM 60초)될 수 있다.
시작할 때 한 번 불러두면 keep_alive(30분) 동안 바로 답한다. 답은 버리고, 판단 설정은 바꾸지 않는다.

  python -m src.tools.warmup            # VLM + LLM
  python -m src.tools.warmup --no-vlm   # LLM만
main.py는 시작할 때 별도 스레드에서 같은 함수를 부른다 (영상 처리는 기다리지 않음).
"""
import argparse
import sys
import time

import numpy as np
import requests

WARMUP_TIMEOUT = 180   # 처음 올릴 때는 평소(60초)보다 오래 걸릴 수 있어 넉넉히


def warm_vlm():
    """실제 질문과 같은 경로(사진 + 질문)로 한 번 호출. 반환: (성공 여부, 걸린 초, 메모)"""
    from src.detection.vlm_client import VlmClient
    vlm = VlmClient(timeout=WARMUP_TIMEOUT)
    ans = vlm.ask_lying(np.full((64, 64, 3), 128, np.uint8))
    return ans["lying"] is not None, ans["latency"], vlm.model if ans["lying"] is not None else ans["reason"]


def warm_llm():
    """LLM(요약, 질문 답변용)을 Ollama에 올려둠. 프롬프트 없이 보내면 모델만 올리고 바로 끝남."""
    from backend.llm import MODEL, OLLAMA_URL, PROVIDER
    if PROVIDER != "ollama":
        return True, 0.0, f"{PROVIDER}는 미리 불러올 필요 없음"
    start = time.perf_counter()
    try:
        res = requests.post(f"{OLLAMA_URL}/api/generate",
                            json={"model": MODEL, "keep_alive": "30m"}, timeout=WARMUP_TIMEOUT)
        res.raise_for_status()
        return True, round(time.perf_counter() - start, 2), MODEL
    except Exception as e:
        return False, round(time.perf_counter() - start, 2), f"LLM 불러오기 실패: {e}"


def warm_all(use_vlm=True, log=print):
    ok_all = True
    for name, fn in (("VLM", warm_vlm), ("LLM", warm_llm)):
        if name == "VLM" and not use_vlm:
            continue
        ok, sec, memo = fn()
        ok_all &= ok
        log(f"[준비 호출] {name} {'OK' if ok else '실패'} ({sec}초) {memo}")
    return ok_all


def main():
    p = argparse.ArgumentParser(description="VLM, LLM 미리 불러두기 (첫 호출 시간 초과 방지)")
    p.add_argument("--no-vlm", action="store_true", help="VLM은 건너뜀 (LLM만)")
    args = p.parse_args()
    sys.exit(0 if warm_all(not args.no_vlm) else 1)


if __name__ == "__main__":
    main()
