"""
녹화 영상에 실시간 화면(main.py)과 같은 판단 결과를 그려서 시연용 MP4로 저장.
(포스터 QR, 전시회 비상 시연용 — 평가 도구가 아님. 판단 설정은 실시간 시스템 그대로)

  python -m src.tools.render_video data/recordings/<녹화> --roi "271,306 267,359 388,364 380,315" \
      --object-model runs/finetune/book_laptop_v1/weights/best.pt --vlm
  핵심 장면만:  ... --clips "5-22 22-66 78-106" --name highlight

- 시각은 frames.csv 기준 (재생 평가와 같음). 판단은 영상 처음부터 끝까지 계속 돌리고,
  --start/--end/--clips는 '저장할 구간'만 고름 -> 물건 원래 위치(처음 3초)와 정지 시간이 실제와 같게 잡힘
- VLM: 호출하는 동안 재생을 멈추고 답을 기다림 (재생 평가와 같음). 배너는 사진을 자른 시각에 뜸
  (실시간에서는 응답 시간만큼 늦게 뜸)
- 한글 배너는 PIL + 맑은 고딕. H.264 MP4는 ffmpeg(PATH 또는 imageio-ffmpeg)로, 없으면 mp4v로 저장
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import OBJECT_MODEL
from src.detection.object_detector import ObjectDetector
from src.detection.pose_estimator import PoseEstimator
from src.detection.vlm_client import VlmClient, crop_person
from src.events.fall_detector import FallDetector, touches_edge
from src.events.intrusion_detector import IntrusionDetector
from src.events.object_displacement_detector import ObjectDisplacementDetector
from src.events.stillness_trigger import StillnessTrigger

OUT_ROOT = Path("data/demo_videos")
RECORDINGS = Path("data/recordings").resolve()

# --- main.py와 같은 그림 설정 (main.py를 import하면 이벤트 전송기가 만들어져서 여기 따로 둠) ---
SKELETON_LINES = [(5, 6), (5, 11), (6, 12), (11, 12)]
RISK_COLORS = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
TARGET_OBJECTS = ["laptop", "book"]

BANNER_SECONDS = 3.0
FONT_PATHS = ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
              "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]
BANNER_COLORS = {"fall": (220, 40, 40), "vlm_yes": (200, 30, 120), "vlm_no": (90, 90, 90),
                 "intrusion": (230, 120, 0), "object": (40, 90, 220)}   # RGB (PIL)
OBJ_NAMES = {"book": "책", "laptop": "노트북"}
OBJ_STATUS = {"moved": "자리 이동", "missing": "사라짐"}
RISK_KO = {"LOW": "낮음", "MEDIUM": "중간", "HIGH": "높음"}


def parse_args():
    p = argparse.ArgumentParser(description="녹화 영상으로 시연 영상 만들기 (판단 결과를 그려서 MP4)")
    p.add_argument("rec_dir", help="녹화 폴더 (video.mp4, frames.csv)")
    p.add_argument("--roi", default="", help="제한 구역 좌표 예: '271,306 267,359 388,364 380,315' (없으면 제한 구역 끔)")
    p.add_argument("--object-model", default="", help="물건 모델 (기본: .env의 OBJECT_MODEL, 파인튜닝: runs/finetune/.../best.pt)")
    p.add_argument("--vlm", action="store_true", help="조건부 VLM 재확인 켜기 (로컬 Ollama 필요)")
    p.add_argument("--start", type=float, default=0, help="저장 시작 (영상 시각, 초)")
    p.add_argument("--end", type=float, default=float("inf"), help="저장 끝 (영상 시각, 초)")
    p.add_argument("--clips", default="", help="저장할 구간 여러 개 예: '5-22 30-66' (--start/--end 대신)")
    p.add_argument("--name", default="full", help="출력 파일 이름 뒤에 붙일 이름 (기본 full)")
    p.add_argument("--no-mirror", action="store_true", help="좌우 반전 끄기 (기본은 실시간 화면처럼 반전)")
    return p.parse_args()


def parse_roi(text):
    return [tuple(int(v) for v in pt.split(",")) for pt in text.split()]


def parse_clips(args):
    if not args.clips:
        return [(args.start, args.end)]
    out = []
    for c in args.clips.split():
        a, _, b = c.partition("-")
        out.append((float(a), float(b)))
    return out


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class Writer:
    """H.264(yuv420p, faststart) — 휴대폰, 웹에서 재생됨. ffmpeg가 없으면 mp4v (PC 재생용)."""

    def __init__(self, path, size, fps):
        self.ffmpeg = find_ffmpeg()
        w, h = size
        if self.ffmpeg:
            self.proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                 "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            print("※ ffmpeg가 없어 mp4v로 저장합니다 (휴대폰/웹에서 안 열릴 수 있음). pip install imageio-ffmpeg 권장")
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        if self.ffmpeg:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.ffmpeg:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def load_font(size):
    for f in FONT_PATHS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    raise SystemExit("한글 글꼴을 찾지 못했습니다 (맑은 고딕 등).")


def draw_objects(frame, detections, res):
    """main.py draw_objects와 같은 그림."""
    if res["calibrating"]:
        cv2.putText(frame, "CALIBRATING... (don't move objects)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for cls, bbox in detections.items():
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        status = "calibrating" if res["calibrating"] else res["results"].get(cls, {}).get("status", "?")
        color = (0, 0, 255) if status == "moved" else (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} [{status}]", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if not res["calibrating"]:
        y = 60
        for cls, r in res["results"].items():
            if r.get("status") == "missing":
                cv2.putText(frame, f"{cls}: MISSING ({r.get('duration', 0)}s)", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25
            elif r.get("status") == "occluded":
                cv2.putText(frame, f"{cls}: HIDDEN BY PERSON", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                y += 25


def draw_person(frame, p, fs, edge, ist):
    """main.py의 사람 그림 (낙상 모드 + 제한 구역)."""
    x1, y1, x2, y2 = p["bbox"]
    color = RISK_COLORS[fs["risk"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {p['track_id']} [{fs['posture']}] risk={fs['risk']}"
    if fs["posture"] == "lying":
        label += f" ({fs['duration']}s)"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    kp = p["keypoints"]
    for (x, y) in kp.values():
        if (x, y) != (0, 0):
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    for a, b in SKELETON_LINES:
        pa, pb = kp.get(a, (0, 0)), kp.get(b, (0, 0))
        if pa != (0, 0) and pb != (0, 0):
            cv2.line(frame, pa, pb, (255, 0, 0), 2)
    if edge:
        cv2.putText(frame, "EDGE", (x1 + 5, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    if ist is not None:
        foot = (int((x1 + x2) / 2), y2)
        cv2.circle(frame, foot, 6, (0, 0, 255) if ist["inside"] else (0, 255, 0), -1)
        if ist["inside"]:
            cv2.putText(frame, f"IN AREA ({ist['duration']}s)", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_overlay_text(frame, banners, t, font_big, font_small):
    """
    한글 배너(아래쪽 가운데, 최근 것이 맨 아래)와 경과 시간(오른쪽 위). PIL로 한 번에 그림.
    왼쪽 위는 실시간 화면의 물건 상태 글자(CALIBRATING, MISSING, HIDDEN BY PERSON) 자리라 비워 둠.
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    bh = font_big.size + 18
    y = h - 10 - bh
    for b in reversed(banners[-3:]):            # 동시에 최대 3개 (최근 것)
        text_w = d.textlength(b["text"], font=font_big)
        x = max(8, (w - text_w) / 2 - 16)
        d.rounded_rectangle([x, y, min(w - 8, x + text_w + 32), y + bh], radius=10,
                            fill=(*BANNER_COLORS[b["kind"]], 225))
        d.text((x + 16, y + 7), b["text"], font=font_big, fill=(255, 255, 255))
        y -= bh + 6
    clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
    tw = d.textlength(clock, font=font_small)
    d.rectangle([w - tw - 20, 6, w - 6, font_small.size + 18], fill=(0, 0, 0, 160))
    d.text((w - tw - 13, 9), clock, font=font_small, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    rec = Path(args.rec_dir)
    with open(rec / "frames.csv", encoding="utf-8") as f:
        times = [float(r["t"]) for r in csv.DictReader(f)]
    clips = parse_clips(args)
    model_path = args.object_model or OBJECT_MODEL

    out_dir = OUT_ROOT / rec.name
    if RECORDINGS in out_dir.resolve().parents:
        raise SystemExit("녹화 폴더 안에는 저장하지 않습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec.name}_{args.name}.mp4"

    pose = PoseEstimator()
    fall = FallDetector()
    intr = IntrusionDetector(parse_roi(args.roi), stay_seconds_threshold=5) if args.roi else None
    obj = ObjectDetector(TARGET_OBJECTS, model_path=model_path)
    disp = ObjectDisplacementDetector(TARGET_OBJECTS)
    stillness = StillnessTrigger() if args.vlm else None
    vlm = VlmClient() if args.vlm else None
    if vlm:
        # 준비 호출: Ollama가 모델을 처음 GPU에 올리는 시간 때문에 첫 질문이 시간 초과되는 걸 막음 (답은 버림)
        warm = vlm.ask_lying(np.full((64, 64, 3), 128, np.uint8))
        print(f"  VLM 준비 호출 {warm['latency']:.1f}초{' (실패: ' + warm['reason'] + ')' if warm['lying'] is None else ''}")
    font_big, font_small = load_font(30), load_font(20)
    print(f"[시연 영상] {rec.name} -> {out_path}")
    print(f"  물건 모델 {model_path} (확신도 {obj.conf_thresholds}), VLM {'켬' if args.vlm else '끔'}, "
          f"제한 구역 {'켬' if intr else '끔'}, 저장 구간 {clips}")

    fps = (len(times) - 1) / (times[-1] - times[0])   # 녹화의 평균 프레임 속도로 저장 (실제 속도로 재생)
    cap = cv2.VideoCapture(str(rec / "video.mp4"))
    writer, banners, events, out_frames = None, [], [], 0
    for idx, t in enumerate(times):
        ret, frame = cap.read()
        if not ret or t > max(b for _, b in clips):
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        raw = frame.copy()
        new = []                                    # 이번 프레임에 생긴 배너

        if intr:
            intr.draw(frame)
        people = pose.estimate(frame)
        dets = obj.detect(frame)
        res = disp.update(dets, now=t, person_boxes=[p["bbox"] for p in people])
        draw_objects(frame, dets, res)
        if not res["calibrating"]:
            for cls, r in res["results"].items():
                if r.get("new_event"):
                    new.append(("object", f"물건 이동 · {OBJ_NAMES.get(cls, cls)} {OBJ_STATUS.get(r['status'], r['status'])}"))

        for p in people:
            tid = p["track_id"]
            edge = touches_edge(p["bbox"], frame.shape)
            fs = fall.update(tid, fall.get_posture(p["keypoints"]), now=t, hold=edge)
            ist = None
            if intr:
                x1, y1, x2, y2 = p["bbox"]
                ist = intr.update(tid, intr.is_inside((int((x1 + x2) / 2), y2)), now=t)
                if ist["new_event"]:
                    new.append(("intrusion", f"제한 구역 접근 · ID {tid} ({ist['duration']}초 머묾)"))
            draw_person(frame, p, fs, edge, ist)
            if fs["new_event"]:
                new.append(("fall", f"낙상 감지 · 위험도 {fs['new_event']}"))
            if stillness:
                st = stillness.update(tid, p["keypoints"], p["bbox"], fs["posture"], now=t, hold=edge)
                if st["ask"]:
                    ans = vlm.ask_lying(crop_person(raw, p["bbox"]))
                    print(f"  [VLM] {t:.1f}초 ID{tid} {st['still']}초 정지, 규칙={fs['posture']} -> {ans}")
                    if ans["lying"] is True:
                        new.append(("vlm_yes", "VLM 확인: 누운 자세 → 낙상 의심"))
                    elif ans["lying"] is False:
                        new.append(("vlm_no", "VLM 확인: 누운 자세 아님"))

        keep = any(a <= t <= b for a, b in clips)
        for kind, text in new:
            banners.append({"kind": kind, "text": text, "until": t + BANNER_SECONDS})
            # 결과 영상 시각 = 이 프레임이 결과 영상에서 놓이는 시각 (저장 구간 밖이면 잘림)
            events.append({"영상 시각(초)": round(t, 1), "결과 영상 시각(초)": round(out_frames / fps, 1) if keep else "(잘림)",
                           "종류": kind, "배너": text})
            print(f"  {t:6.1f}초  {text}{'' if keep else '  (저장 구간 밖)'}")
        banners = [b for b in banners if b["until"] > t]

        if not keep:
            continue
        frame = draw_overlay_text(frame, banners, t, font_big, font_small)
        if writer is None:
            writer = Writer(out_path, (frame.shape[1], frame.shape[0]), fps)
        writer.write(frame)
        out_frames += 1
    cap.release()
    if writer:
        writer.close()

    with open(out_path.with_suffix(".events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["영상 시각(초)", "결과 영상 시각(초)", "종류", "배너"])
        w.writeheader()
        w.writerows(events)
    print(f"  저장: {out_path} ({out_frames}프레임, {out_frames / fps:.1f}초, {fps:.2f}fps, "
          f"{'H.264' if find_ffmpeg() else 'mp4v'})")
    print(f"  이벤트 목록: {out_path.with_suffix('.events.csv')}")


if __name__ == "__main__":
    main()
