"""
학교 평가 프레임(손으로 단 정답 라벨)으로 물건 검출 모델 점수 내기 — 책만.

  python -m src.experiments.detect_eval --tag baseline                        # 기본 yolov8s, 책 0.15
  python -m src.experiments.detect_eval --model runs/finetune/<이름>/weights/best.pt --tag ft_v1

- 시스템과 같은 조건: 화면을 좌우 반전해서 검출 (실시간 실행과 같음) -> 박스를 원본 좌표로 되돌려 정답과 비교
- 확신도 기준: 모델 옆 thresholds.json, 없으면 기본값 (book 0.15) — 시스템이 쓰는 값 그대로
- 두 가지로 셈:
    시스템 기준 = 프레임마다 확신도가 가장 높은 책 1개만 (ObjectDetector가 실제로 쓰는 박스)
    모든 박스   = 기준을 넘은 책 박스 전부 (모델 자체 성능)
- 정답과 IoU 0.5 이상이면 맞음. 정답 하나에 박스 하나만 짝지음, 남은 박스는 '잘못 찾은 박스'
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

from src.config import YOLO_DEVICE
from src.detection.object_detector import conf_thresholds_for
from src.tools.extract_frames import CLASSES

TEST_ROOT = Path("data/finetune/test_school")
RESULT_ROOT = Path("data/detect_eval")
RECORDINGS = Path("data/recordings")
TARGET = "book"
PHASES = ["이동", "정상 동작", "가만히"]


def parse_args():
    p = argparse.ArgumentParser(description="학교 평가 프레임으로 책 검출 점수 (재현율, 잘못 찾은 박스)")
    p.add_argument("--rec", default="eval2_object_school_20261104_135118", help="학교 녹화 이름")
    p.add_argument("--model", default="yolov8s.pt", help="검출 모델 (기본 yolov8s.pt)")
    p.add_argument("--conf", type=float, default=None, help="책 확신도 기준 (안 주면 thresholds.json 또는 0.15)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--idle-trial", default="23", help="'가만히' 단계로 볼 marks.csv 시도 번호")
    p.add_argument("--no-mirror", action="store_true", help="좌우 반전 끄기 (기본은 시스템처럼 반전)")
    p.add_argument("--tag", required=True, help="결과 이름 (예: baseline, ft_v1)")
    p.add_argument("--overwrite", action="store_true", help="같은 이름의 결과가 있어도 덮어쓰기")
    return p.parse_args()


def model_label(model):
    """yolov8s.pt -> yolov8s / runs/finetune/book_laptop_v1/weights/best.pt -> book_laptop_v1"""
    m = Path(model)
    return m.parent.parent.name if m.parent.name == "weights" else m.stem


def load_phases(rec, idle_trial):
    """marks.csv로 단계 나누기: 이동 = 마지막 이동 시도 뒤 첫 정상 시도 전까지 / 가만히 = idle_trial / 나머지 = 정상 동작"""
    with open(RECORDINGS / rec / "marks.csv", encoding="utf-8") as f:
        marks = list(csv.DictReader(f))
    last_move = max(float(m["end_t"]) for m in marks if m["label"] == "object_move")
    move_end = min(float(m["start_t"]) for m in marks if m["label"] == "normal" and float(m["start_t"]) > last_move)
    idle = next(m for m in marks if m["trial"] == idle_trial)
    idle_s, idle_e = float(idle["start_t"]), float(idle["end_t"])

    def phase(t):
        if idle_s <= t <= idle_e:
            return "가만히"
        return "이동" if t < move_end else "정상 동작"
    return phase, f"이동 < {move_end:.1f}초, 가만히 = #{idle_trial} {idle_s:.1f}~{idle_e:.1f}초, 나머지 정상 동작"


def read_truth(txt, w, h):
    """YOLO 라벨 -> 책 박스 (픽셀 x1,y1,x2,y2)"""
    boxes = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if not p or CLASSES[int(p[0])] != TARGET:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(truth, preds, thr):
    """확신도 높은 박스부터 정답과 짝지음. 반환: (맞힌 정답 수, 잘못 찾은 박스 수, 정답별 최고 IoU)"""
    used, fp = set(), 0
    for box, _ in sorted(preds, key=lambda x: -x[1]):
        best, best_i = 0.0, None
        for i, g in enumerate(truth):
            if i not in used and (v := iou(box, g)) > best:
                best, best_i = v, i
        if best >= thr:
            used.add(best_i)
        else:
            fp += 1
    best_iou = max((iou(b, g) for b, _ in preds for g in truth), default=0.0)
    return len(used), fp, best_iou


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def main():
    args = parse_args()
    from ultralytics import YOLO

    frames_dir = TEST_ROOT / "frames" / args.rec
    labels_dir = TEST_ROOT / "labels" / args.rec
    out = RESULT_ROOT / f"{args.rec}_{model_label(args.model)}_{args.tag}"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"이미 있는 결과입니다: {out} (기준선을 지키려고 덮어쓰지 않음. 다른 --tag를 쓰세요)")

    conf = args.conf if args.conf is not None else conf_thresholds_for(args.model).get(TARGET, 0.4)
    model = YOLO(args.model)
    if TARGET not in model.names.values():
        raise SystemExit(f"모델에 {TARGET} 클래스가 없습니다: {args.model}")
    phase_of, phase_rule = load_phases(args.rec, args.idle_trial)

    with open(frames_dir / "index.csv", encoding="utf-8") as f:
        t_of = {Path(r["file"]).stem: float(r["t"]) for r in csv.DictReader(f)}

    rows = []
    for img_path in sorted(frames_dir.glob("*.jpg")):
        stem = img_path.stem
        if not (frames_dir / f"{stem}.json").exists():
            raise SystemExit(f"정답 확인이 안 된 이미지가 있습니다 (.json 없음): {stem}")
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        truth = read_truth(labels_dir / f"{stem}.txt", w, h)

        src = frame if args.no_mirror else cv2.flip(frame, 1)
        res = model(src, conf=conf, verbose=False, device=YOLO_DEVICE)[0]
        preds = []
        for b in res.boxes:
            if model.names[int(b.cls[0])] != TARGET or float(b.conf[0]) < conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            if not args.no_mirror:
                x1, x2 = w - x2, w - x1          # 반전 화면 좌표 -> 원본(정답) 좌표
            preds.append(((x1, y1, x2, y2), float(b.conf[0])))
        top = sorted(preds, key=lambda x: -x[1])[:1]   # 시스템이 쓰는 박스 (가장 확신 높은 1개)

        tp_all, fp_all, best_iou = match(truth, preds, args.iou)
        tp_sys, fp_sys, _ = match(truth, top, args.iou)
        t = t_of[stem]
        rows.append({"이미지": stem, "t": round(t, 1), "단계": phase_of(t), "정답 책": len(truth),
                     "찾은 박스": len(preds), "최고 확신도": round(top[0][1], 3) if top else "",
                     "최고 IoU": round(best_iou, 3),
                     "시스템 맞음": tp_sys, "시스템 잘못 찾음": fp_sys,
                     "모든 박스 맞음": tp_all, "모든 박스 잘못 찾음": fp_all})

    summary = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["단계"]].append(r)
    for name, rs in [(p, groups[p]) for p in PHASES] + [("전체", rows)]:
        n_truth = sum(r["정답 책"] for r in rs)
        s = {"단계": name, "이미지": len(rs), "책 있는 이미지": sum(r["정답 책"] > 0 for r in rs), "정답 책 박스": n_truth}
        for k in ("시스템", "모든 박스"):
            tp = sum(r[f"{k} 맞음"] for r in rs)
            s[f"{k} 맞힘"] = tp
            s[f"{k} 재현율"] = round(tp / n_truth, 3) if n_truth else ""
            s[f"{k} 잘못 찾은 박스"] = sum(r[f"{k} 잘못 찾음"] for r in rs)
            s[f"{k} 잘못 찾은 이미지"] = sum(r[f"{k} 잘못 찾음"] > 0 for r in rs)
        s["책 없는 이미지에서 잘못 찾음"] = sum(r["모든 박스 잘못 찾음"] > 0 for r in rs if r["정답 책"] == 0)
        summary.append(s)

    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_image.csv", rows), ("summary.csv", summary)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            wr.writeheader()
            wr.writerows(data)
    info = {"tag": args.tag, "label": "기준선(기본 모델)" if args.tag == "baseline" else args.tag,
            "model": args.model, "book_conf": conf, "iou": args.iou, "mirror": not args.no_mirror,
            "phases": phase_rule, "images": len(rows), "commit": git_commit(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[책 검출 평가] {info['label']} | 모델 {args.model} | 책 확신도 {conf} | IoU {args.iou} | "
          f"반전 {'켬' if info['mirror'] else '끔'}")
    print(f"  단계: {phase_rule}")
    print(f"  {'단계':<8}{'이미지':>5}{'정답':>5} | {'시스템(1개) 재현율':>16}{'잘못 찾음':>8} | {'모든 박스 재현율':>14}{'잘못 찾음':>8}")
    for s in summary:
        print(f"  {s['단계']:<8}{s['이미지']:>5}{s['정답 책 박스']:>5} | "
              f"{s['시스템 맞힘']:>5}/{s['정답 책 박스']:<3} {s['시스템 재현율']!s:>6}{s['시스템 잘못 찾은 박스']:>8} | "
              f"{s['모든 박스 맞힘']:>5}/{s['정답 책 박스']:<3} {s['모든 박스 재현율']!s:>6}{s['모든 박스 잘못 찾은 박스']:>8}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
