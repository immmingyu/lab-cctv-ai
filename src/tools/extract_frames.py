import argparse
import csv
from pathlib import Path

import cv2

# 파인튜닝 클래스 (이 순서가 YOLO 라벨 번호: book=0, laptop=1)
CLASSES = ["book", "laptop"]
OUT_ROOT = Path("data/finetune/frames")
RECORDINGS = Path("data/recordings").resolve()


def parse_args():
    p = argparse.ArgumentParser(description="녹화 영상에서 라벨링용 프레임 뽑기 (원본 그대로, 반전 없음)")
    p.add_argument("rec_dir", help="녹화 폴더 (video.mp4, frames.csv)")
    p.add_argument("--every", type=float, default=1.5, help="몇 초마다 한 장 (기본 1.5)")
    p.add_argument("--out", default="", help="저장 폴더 (기본: data/finetune/frames/<녹화이름>)")
    p.add_argument("--start", type=float, default=0, help="이 시각(초)부터")
    p.add_argument("--end", type=float, default=float("inf"), help="이 시각(초)까지")
    p.add_argument("--min-change", type=float, default=0,
                   help="바로 앞에 저장한 프레임과 평균 밝기 차이가 이보다 작으면 건너뜀 (0=끄기, 예: 3)")
    return p.parse_args()


def small_gray(frame):
    """장면 변화 비교용으로 작게 줄인 흑백 이미지."""
    return cv2.cvtColor(cv2.resize(frame, (64, 48)), cv2.COLOR_BGR2GRAY).astype("float32")


def main():
    args = parse_args()
    rec = Path(args.rec_dir)
    out = Path(args.out) if args.out else OUT_ROOT / rec.name
    if RECORDINGS in out.resolve().parents or out.resolve() == RECORDINGS:
        raise SystemExit("녹화 폴더(data/recordings) 안에는 저장하지 않습니다. --out을 다른 곳으로 지정하세요.")
    out.mkdir(parents=True, exist_ok=True)

    with open(rec / "frames.csv", encoding="utf-8") as f:
        times = [float(r["t"]) for r in csv.DictReader(f)]
    cap = cv2.VideoCapture(str(rec / "video.mp4"))

    rows, next_t, prev, skipped = [], max(args.start, 0), None, 0
    for idx, t in enumerate(times):
        ret, frame = cap.read()                  # 순서대로 읽기 (mp4 위치 이동보다 정확)
        if not ret or t > args.end:
            break
        if t < next_t:
            continue
        next_t = t + args.every
        if args.min_change > 0:
            g = small_gray(frame)
            if prev is not None and abs(g - prev).mean() < args.min_change:
                skipped += 1
                continue
            prev = g
        name = f"{rec.name}_f{idx:06d}_t{t:07.1f}.jpg"
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([name, idx, round(t, 3)])
    cap.release()

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "frame", "t"])
        w.writerows(rows)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")   # X-AnyLabeling YOLO 형식용

    print(f"[프레임 추출] {rec.name}: {len(rows)}장 저장 ({args.every}초 간격"
          + (f", 비슷해서 건너뜀 {skipped}장" if args.min_change > 0 else "") + f") -> {out}")


if __name__ == "__main__":
    main()
