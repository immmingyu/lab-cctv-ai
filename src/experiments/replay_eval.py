import argparse
import csv
import math
from pathlib import Path

import cv2

from src.detection.object_detector import ObjectDetector
from src.detection.pose_estimator import PoseEstimator
from src.events.fall_detector import FallDetector
from src.events.intrusion_detector import IntrusionDetector
from src.events.object_displacement_detector import ObjectDisplacementDetector

TARGET_OBJECTS = ["laptop", "book"]
EXPECTED = {"fall": "fall_detection", "intrusion": "restricted_area_intrusion",
            "object_move": "object_displacement"}
GRACE = 2.0   # 시도 끝(e)을 조금 일찍 눌렀을 때를 위한 여유 시간(초)
RESULT_DIR = Path("data/replay_results")

# 정상 동작별로 "생기면 오경보인 이벤트" (사람 작업 기준)
NORMAL_TARGETS = {
    "pickup": {"fall_detection"},                 # 물건 줍기 -> 낙상이 울리면 오경보
    "sit": {"fall_detection"},                    # 가만히 앉기 -> 낙상이 울리면 오경보
    "pass": {"restricted_area_intrusion"},        # 빠르게 지나가기 -> 제한 구역이 울리면 오경보
}

OBJECT_VARIANTS = {
    "예전 (이동만)": dict(missing_confirm_seconds=math.inf),
    "새 로직 (사라짐 4초)": dict(missing_confirm_seconds=4),
    "새 로직 (사라짐 6초)": dict(missing_confirm_seconds=6),
}


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_roi(text):
    return [tuple(int(v) for v in p.split(",")) for p in text.split()]


def parse_groups(text):
    """'desk=1-14 out=15-24' -> {1:'desk', ..., 15:'out', ...}"""
    groups = {}
    for part in (text or "").split():
        name, rng = part.split("=")
        a, b = rng.split("-")
        for n in range(int(a), int(b) + 1):
            groups[n] = name
    return groups


def frames(cap, times, mirror):
    """영상 프레임과 녹화 당시 시각을 함께 넘겨줌."""
    idx = 0
    while idx < len(times):
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv2.flip(frame, 1)   # 실시간 실행, 제한 구역 좌표와 같은 방향으로
        if idx % 500 == 0:
            print(f"  {idx}/{len(times)} 프레임 ({times[idx]:.0f}초)")
        yield times[idx], frame
        idx += 1


def run_object(cap, times, mirror):
    detector = ObjectDetector(TARGET_OBJECTS, conf_thresholds={"book": 0.15})
    variants = {name: ObjectDisplacementDetector(TARGET_OBJECTS, **kw) for name, kw in OBJECT_VARIANTS.items()}
    events = {name: [] for name in variants}
    for t, frame in frames(cap, times, mirror):
        dets = detector.detect(frame)            # 검출은 한 번, 판단은 세 가지 로직에 동시에
        for name, det in variants.items():
            res = det.update(dets, now=t)
            if res["calibrating"]:
                continue
            for cls, r in res["results"].items():
                if r.get("new_event"):
                    events[name].append({"t": t, "type": "object_displacement", "detail": f"{cls}:{r['status']}"})
    return events


def run_people(cap, times, mirror, roi):
    pose = PoseEstimator()
    fall = FallDetector()
    intr = IntrusionDetector(roi, stay_seconds_threshold=5)
    evs = []
    for t, frame in frames(cap, times, mirror):
        for p in pose.estimate(frame):
            tid = p["track_id"]
            fs = fall.update(tid, fall.get_posture(p["keypoints"]), now=t)
            if fs["new_event"]:
                evs.append({"t": t, "type": "fall_detection", "detail": fs["new_event"]})
            x1, y1, x2, y2 = p["bbox"]
            ist = intr.update(tid, intr.is_inside((int((x1 + x2) / 2), y2)), now=t)
            if ist["new_event"]:
                evs.append({"t": t, "type": "restricted_area_intrusion", "detail": f"ID{tid}"})
    return {"현재 시스템": evs}


def score(marks, events, groups, out_dir):
    trial_rows = []
    event_rows = []
    for variant, evs in events.items():
        used = set()
        stats = {}   # (label, group) -> [시도, 성공]
        for m in marks:
            n, label = int(m["trial"]), m["label"]
            st, et = float(m["start_t"]), float(m["end_t"]) + GRACE
            inwin = [i for i, e in enumerate(evs) if st <= e["t"] <= et]
            used.update(inwin)
            g = groups.get(n, "-")
            if label == "normal":
                targets = NORMAL_TARGETS.get(g)
                if targets:   # 그 정상 동작이 시험하는 이벤트 종류만 확인
                    ok = not any(evs[i]["type"] in targets for i in inwin)
                else:         # 그룹이 지정 안 됐으면 어떤 이벤트든 없어야 성공
                    ok = len(inwin) == 0
            else:
                ok = any(evs[i]["type"] == EXPECTED[label] for i in inwin)
            s = stats.setdefault((label, g), [0, 0])
            s[0] += 1
            s[1] += ok
            trial_rows.append([variant, n, label, g, round(st, 1), round(et - GRACE, 1),
                               "; ".join(f"{evs[i]['type']}({evs[i]['detail']})@{evs[i]['t']:.1f}" for i in inwin),
                               "성공" if ok else "실패"])
        outside = [e for i, e in enumerate(evs) if i not in used]
        event_rows += [[variant, round(e["t"], 1), e["type"], e["detail"], "밖" if i not in used else "안"]
                       for i, e in enumerate(evs)]

        print(f"\n===== {variant} =====")
        for label in ("fall", "intrusion", "object_move", "normal"):
            rows = {g: v for (lb, g), v in stats.items() if lb == label}
            if not rows:
                continue
            tot = [sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())]
            what = "오경보 없음" if label == "normal" else "감지"
            print(f"  {label:<12} {what} {tot[1]}/{tot[0]} ({tot[1] / tot[0] * 100:.0f}%)")
            if len(rows) > 1 or "-" not in rows:
                for g, v in rows.items():
                    print(f"      {g:<10} {v[1]}/{v[0]}")
        print(f"  시도 구간 밖 이벤트: {len(outside)}개"
              + (" → " + ", ".join(f"{e['type']}@{e['t']:.0f}s" for e in outside[:10]) if outside else ""))

    with open(out_dir / "replay_trials.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["설정", "시도", "종류", "그룹", "시작(초)", "끝(초)", "구간 안 이벤트", "결과"])
        w.writerows(trial_rows)
    with open(out_dir / "replay_events.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["설정", "시각(초)", "이벤트", "내용", "시도 구간"])
        w.writerows(event_rows)
    print(f"\n시도별 결과: {out_dir / 'replay_trials.csv'}")
    print(f"이벤트 목록: {out_dir / 'replay_events.csv'}")


def main():
    ap = argparse.ArgumentParser(description="녹화 영상 재생 평가")
    ap.add_argument("rec_dir", help="녹화 폴더 (video.mp4, frames.csv, marks.csv)")
    ap.add_argument("--task", choices=["object", "people"], required=True)
    ap.add_argument("--roi", default="", help="제한 구역 좌표 예: '178,395 185,475 311,478 318,407'")
    ap.add_argument("--groups", default="", help="시도 번호 그룹 예: 'desk=1-14 out=15-24'")
    ap.add_argument("--no-mirror", action="store_true", help="좌우 반전 끄기")
    args = ap.parse_args()

    d = Path(args.rec_dir)
    times = [float(r["t"]) for r in load_csv(d / "frames.csv")]
    marks = load_csv(d / "marks.csv")
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    print(f"[재생 평가] {d.name}  프레임 {len(times)}, 시도 {len(marks)}개, 작업 {args.task}")

    if args.task == "object":
        events = run_object(cap, times, not args.no_mirror)
    else:
        if not args.roi:
            raise SystemExit("people 작업은 --roi 좌표가 필요합니다.")
        events = run_people(cap, times, not args.no_mirror, parse_roi(args.roi))
    cap.release()

    # 녹화 폴더(data/recordings)는 건드리지 않도록 결과는 따로 저장
    out_dir = RESULT_DIR / d.name
    out_dir.mkdir(parents=True, exist_ok=True)
    score(marks, events, parse_groups(args.groups), out_dir)


if __name__ == "__main__":
    main()