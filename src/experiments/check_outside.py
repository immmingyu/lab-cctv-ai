import csv
import sys
from pathlib import Path

d = Path(sys.argv[1])
times = [float(x) for x in sys.argv[2].split(",")]
with open(d / "marks.csv", encoding="utf-8") as f:
    marks = list(csv.DictReader(f))

for t in times:
    prev = [m for m in marks if float(m["end_t"]) <= t]
    nxt = [m for m in marks if float(m["start_t"]) > t]
    p = prev[-1] if prev else None
    n = nxt[0] if nxt else None
    before = f"직전 #{p['trial']}({p['label']}) 끝난 뒤 {t - float(p['end_t']):5.1f}초" if p else "첫 시도 전"
    after = f"다음 #{n['trial']} 시작 {float(n['start_t']) - t:5.1f}초 전" if n else "마지막 시도 이후"
    print(f"{t:7.1f}s  {before}  |  {after}")