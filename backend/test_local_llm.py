import time
import requests

MODEL = "exaone3.5:7.8b"

prompt = ("연구실 CCTV에서 14시 32분에 사람이 넘어진 뒤 15초간 움직이지 않았습니다. "
          "관리자에게 보낼 상황 설명을 두 문장으로 써주세요.")

start = time.time()
res = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    },
    timeout=120,
)
res.raise_for_status()
data = res.json()

print("모델:", MODEL)
print("응답:", data["message"]["content"])
print(f"걸린 시간: {time.time() - start:.1f}초")