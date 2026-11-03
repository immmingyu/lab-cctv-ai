import base64
import json
import time

import cv2
import requests

from src.config import OLLAMA_URL, VLM_MODEL

# 질문 문구 (확정, 새 영상 검증 전 고정 - README '평가 설정' 참고)
# 장소 조건(바닥 등)을 넣으면 매트 위에 누운 사람을 '침대라서 아니오'로 답해서, 자세만 묻도록 함
QUESTION = ("이 사람이 누워 있거나 쓰러진 자세입니까? "
            "장소(바닥, 매트 등)와 상관없이 자세만 보고 예/아니오로 답하고 근거를 짧게 적어주세요.")

# 답을 항상 같은 모양으로 받기 위한 JSON 형식 (Ollama structured output)
ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["예", "아니오"]},
        "reason": {"type": "string"},
    },
    "required": ["answer", "reason"],
}


def crop_person(frame, bbox, margin=0.3):
    """사람 박스를 사방으로 margin 비율만큼 넓혀서 잘라냄 (바닥, 주변이 보이게)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)].copy()


class VlmClient:
    def __init__(self, model=VLM_MODEL, url=OLLAMA_URL, timeout=60):
        self.model = model
        self.url = url
        self.timeout = timeout

    def ask_lying(self, image):
        """
        image: 잘라낸 사람 사진 (BGR)
        반환값: {"lying": True/False, "reason": "...", "latency": 초}
                호출 실패 시 {"lying": None, "reason": 오류 내용, "latency": 초}
        """
        ok, jpg = cv2.imencode(".jpg", image)
        start = time.perf_counter()
        try:
            res = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": QUESTION,
                                  "images": [base64.b64encode(jpg.tobytes()).decode()]}],
                    "format": ANSWER_FORMAT,
                    "stream": False,
                    "options": {"temperature": 0},   # 같은 사진엔 같은 답이 나오게
                    "keep_alive": "30m",
                },
                timeout=self.timeout,
            )
            res.raise_for_status()
            ans = json.loads(res.json()["message"]["content"])
            return {"lying": ans["answer"] == "예", "reason": ans.get("reason", ""),
                    "latency": round(time.perf_counter() - start, 2)}
        except Exception as e:   # Ollama가 꺼져 있어도 영상 처리는 계속되게
            return {"lying": None, "reason": f"VLM 호출 실패: {e}",
                    "latency": round(time.perf_counter() - start, 2)}
