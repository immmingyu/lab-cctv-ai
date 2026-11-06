"""
집 검증 세션(세션 2) 라벨로 책 확신도 기준 고르기 — F1이 가장 높은 값.
기본 모델과 파인튜닝 모델에 똑같이 씀 (README '공정 비교 계획').

  python -m src.experiments.pick_threshold --model yolov8s.pt
  python -m src.experiments.pick_threshold --model runs/finetune/book_laptop_v1/weights/best.pt --write

정한 방법 (학교 결과를 보기 전에 고정):
- 검출 조건은 시스템, detect_eval과 같음: 좌우 반전해서 검출 -> 원본 좌표로 되돌려 비교
- 기준을 넘은 책 박스 전부를 정답과 IoU 0.5 이상으로 짝지음 (detect_eval의 match와 같은 함수)
- 기준 0.01 ~ 0.95 (0.01 간격) 중 F1이 가장 높은 값. F1이 같으면 더 낮은 기준 (놓치는 게 문제라서)
- --write: 파인튜닝 모델의 weights 폴더에 thresholds.json 저장.
  기본 모델(yolov8s.pt)은 저장하지 않음 — 옆에 두면 실시간 시스템 기본값(0.15)이 바뀌기 때문.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2

from src.config import YOLO_DEVICE
from src.experiments.detect_eval import TARGET, match, model_label, read_truth
from src.tools.build_dataset import is_test_path
from src.tools.extract_frames import OUT_ROOT as FRAMES_ROOT

LABELS_ROOT = Path("data/finetune/labels")
RESULT_ROOT = Path("data/threshold_select")
VAL_SESSION = "ft_home_s2_living_20261106_185234"
GRID = [round(0.01 * i, 2) for i in range(1, 96)]


def parse_args():
    p = argparse.ArgumentParser(description="집 검증 세션으로 책 확신도 기준 고르기 (F1 최고)")
    p.add_argument("--model", required=True)
    p.add_argument("--session", default=VAL_SESSION, help="검증 세션 폴더 이름 (기본: 세션 2)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--write", action="store_true", help="파인튜닝 모델 weights 폴더에 thresholds.json 저장")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    frames_dir, labels_dir = FRAMES_ROOT / args.session, LABELS_ROOT / args.session
    if "school" in args.session.lower() or is_test_path(frames_dir) or is_test_path(labels_dir):
        raise SystemExit("학교 영상은 기준값 고르기에 쓰지 않습니다 (평가 전용).")
    is_finetuned = Path(args.model).parent.name == "weights"
    if args.write and not is_finetuned:
        raise SystemExit("기본 모델은 thresholds.json을 저장하지 않습니다 (실시간 시스템 기본값 0.15 유지). "
                         "값은 README에 기록하고 평가 명령에서 --conf / --book-conf로 넘기세요.")

    model = YOLO(args.model)
    if TARGET not in model.names.values():
        raise SystemExit(f"모델에 {TARGET} 클래스가 없습니다: {args.model}")

    images = []                                   # (정답 박스, [(박스, 확신도)...]) — 낮은 기준으로 한 번만 검출
    for img_path in sorted(frames_dir.glob("*.jpg")):
        if not img_path.with_suffix(".json").exists():
            continue                              # 확인 안 한 이미지는 제외 (build_dataset과 같은 기준)
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        truth = read_truth(labels_dir / f"{img_path.stem}.txt", w, h)
        src = frame if args.no_mirror else cv2.flip(frame, 1)
        res = model(src, conf=GRID[0], verbose=False, device=YOLO_DEVICE)[0]
        preds = []
        for b in res.boxes:
            if model.names[int(b.cls[0])] != TARGET:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            if not args.no_mirror:
                x1, x2 = w - x2, w - x1
            preds.append(((x1, y1, x2, y2), float(b.conf[0])))
        images.append((truth, preds))
    if not images:
        raise SystemExit(f"확인한 이미지가 없습니다: {frames_dir}")

    n_truth = sum(len(t) for t, _ in images)
    rows = []
    for thr in GRID:
        tp = fp = 0
        for truth, preds in images:
            a, b, _ = match(truth, [p for p in preds if p[1] >= thr], args.iou)
            tp, fp = tp + a, fp + b
        fn = n_truth - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n_truth if n_truth else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"기준": thr, "맞힘": tp, "잘못 찾음": fp, "놓침": fn,
                     "정밀도": round(prec, 4), "재현율": round(rec, 4), "F1": round(f1, 4)})
    best = max(rows, key=lambda r: (r["F1"], -r["기준"]))   # F1 최고, 같으면 낮은 기준

    out = RESULT_ROOT / f"{args.session}_{model_label(args.model)}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    info = {"model": args.model, "session": args.session, "images": len(images), "book_boxes": n_truth,
            "iou": args.iou, "mirror": not args.no_mirror, "best": best,
            "rule": "F1 최고, 같으면 낮은 기준 (0.01~0.95, 0.01 간격)",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (out / "best.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[책 확신도 고르기] 모델 {args.model} | 세션 {args.session} ({len(images)}장, 정답 책 {n_truth}개)")
    for r in rows:
        if r["기준"] in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) or r is best:
            print(f"  {r['기준']:.2f}  F1 {r['F1']:.4f}  정밀도 {r['정밀도']:.4f}  재현율 {r['재현율']:.4f}  "
                  f"(맞힘 {r['맞힘']}, 잘못 찾음 {r['잘못 찾음']}, 놓침 {r['놓침']}){'  <- 최고' if r is best else ''}")
    print(f"  -> 고른 책 확신도: {best['기준']}  ({out})")
    if args.write:
        th = Path(args.model).with_name("thresholds.json")
        th.write_text(json.dumps({"book": best["기준"]}, indent=2) + "\n", encoding="utf-8")
        print(f"  thresholds.json 저장: {th}")


if __name__ == "__main__":
    main()
