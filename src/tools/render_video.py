"""
녹화 영상에 실시간 화면(main.py)과 같은 판단 결과를 그려서 시연용 MP4로 저장.
(포스터 QR, 전시회 비상 시연용 — 평가 도구가 아님. 판단 설정은 실시간 시스템 그대로)

  python -m src.tools.render_video data/recordings/<녹화> --roi "271,306 267,359 388,364 380,315" \
      --object-model runs/finetune/book_laptop_v1/weights/best.pt --vlm
  핵심 장면만:  ... --clips "5-22 22-66 78-106" --name highlight
  여러 녹화를 장면 제목과 이어 붙이기: python -m src.tools.make_highlight

- 시각은 frames.csv 기준 (재생 평가와 같음). 판단은 영상 처음부터 계속 돌리고,
  --start/--end/--clips는 '저장할 구간'만 고름 -> 물건 원래 위치(처음 3초)와 정지 시간이 실제와 같게 잡힘
- VLM: 호출하는 동안 재생을 멈추고 답을 기다림 (재생 평가와 같음). 배너는 사진을 자른 시각에 뜸
  (실시간에서는 응답 시간만큼 늦게 뜸). 시작 전에 준비 호출 한 번
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
VLM_NO_MODES = ("show", "small", "hide")   # VLM '아님' 배너: 그대로 / 작게 / 숨김 (시연용 선택)


def parse_args():
    p = argparse.ArgumentParser(description="녹화 영상으로 시연 영상 만들기 (판단 결과를 그려서 MP4)")
    p.add_argument("rec_dir", help="녹화 폴더 (video.mp4, frames.csv)")
    p.add_argument("--roi", default="", help="제한 구역 좌표 예: '271,306 267,359 388,364 380,315' (없으면 제한 구역 끔)")
    p.add_argument("--object-model", default="", help="물건 모델 (기본: .env의 OBJECT_MODEL, 파인튜닝: runs/finetune/.../best.pt)")
    p.add_argument("--no-objects", action="store_true", help="물건 표시와 판단 끄기 (main.py --only 처럼 화면 범위만 줄임)")
    p.add_argument("--vlm", action="store_true", help="조건부 VLM 재확인 켜기 (로컬 Ollama 필요)")
    p.add_argument("--vlm-no", choices=VLM_NO_MODES, default="show",
                   help="VLM이 '아님'이라고 답한 배너: show(기본) / small(작게) / hide(숨김)")
    p.add_argument("--start", type=float, default=0, help="저장 시작 (영상 시각, 초)")
    p.add_argument("--end", type=float, default=float("inf"), help="저장 끝 (영상 시각, 초)")
    p.add_argument("--clips", default="", help="저장할 구간 여러 개 예: '5-22 30-66' (--start/--end 대신)")
    p.add_argument("--name", default="full", help="출력 파일 이름 뒤에 붙일 이름 (기본 full)")
    p.add_argument("--no-mirror", action="store_true", help="좌우 반전 끄기 (기본은 실시간 화면처럼 반전)")
    return p.parse_args()


def parse_roi(text):
    return [tuple(int(v) for v in pt.split(",")) for pt in text.split()]


def parse_clips(text, start=0, end=float("inf")):
    if not text:
        return [(start, end)]
    out = []
    for c in text.split():
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


class Renderer:
    """출력 영상 하나에 장면(녹화 구간)과 제목 화면을 차례로 이어 씀."""

    def __init__(self, out_path, fps=30.0):
        self.out_path = Path(out_path)
        if RECORDINGS in self.out_path.resolve().parents:
            raise SystemExit("녹화 폴더 안에는 저장하지 않습니다.")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None
        self.frames = 0
        self.events = []
        self.fonts = {"big": load_font(30), "small": load_font(20), "title": load_font(44)}
        self._vlm = None

    def vlm(self):
        if self._vlm is None:
            self._vlm = VlmClient()
            from src.tools.warmup import warm_vlm
            ok, sec, memo = warm_vlm()     # 첫 질문 시간 초과 방지
            print(f"  VLM 준비 호출 {'OK' if ok else '실패'} ({sec}초) {memo}")
        return self._vlm

    @property
    def now(self):
        return self.frames / self.fps

    def _write(self, frame):
        if self.writer is None:
            self.writer = Writer(self.out_path, (frame.shape[1], frame.shape[0]), self.fps)
            self.size = (frame.shape[1], frame.shape[0])
        self.writer.write(frame)
        self.frames += 1

    def title(self, text, seconds=1.5, size=(640, 480), sub=""):
        """장면 제목 화면 (검은 배경에 한글 제목)."""
        w, h = getattr(self, "size", size)
        img = Image.new("RGB", (w, h), (18, 22, 30))
        d = ImageDraw.Draw(img)
        f = self.fonts["title"]
        while d.textlength(text, font=f) > w - 40 and f.size > 20:   # 화면 폭에 맞게 줄임
            f = load_font(f.size - 2)
        tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2, h / 2 - f.size), text, font=f, fill=(255, 255, 255))
        if sub:
            fs = self.fonts["small"]
            d.text(((w - d.textlength(sub, font=fs)) / 2, h / 2 + 16), sub, font=fs, fill=(180, 190, 205))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for _ in range(int(round(seconds * self.fps))):
            self._write(frame)

    def overlay(self, frame, banners, t, vlm_no):
        """한글 배너(아래쪽 가운데)와 영상 시각(오른쪽 위). 왼쪽 위는 물건 상태 글자 자리라 비워 둠."""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        y = h - 10
        shown = [b for b in banners if not (b["kind"] == "vlm_no" and vlm_no == "hide")]
        for b in reversed(shown[-3:]):          # 동시에 최대 3개, 최근 것이 맨 아래
            small = b["kind"] == "vlm_no" and vlm_no == "small"
            f = self.fonts["small"] if small else self.fonts["big"]
            pad, bh = (10, f.size + 12) if small else (16, f.size + 18)
            y -= bh
            text_w = d.textlength(b["text"], font=f)
            x = max(8, (w - text_w) / 2 - pad)
            d.rounded_rectangle([x, y, min(w - 8, x + text_w + 2 * pad), y + bh], radius=8 if small else 10,
                                fill=(*BANNER_COLORS[b["kind"]], 170 if small else 225))
            d.text((x + pad, y + (5 if small else 7)), b["text"], font=f, fill=(255, 255, 255))
            y -= 6
        fs = self.fonts["small"]
        clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
        tw = d.textlength(clock, font=fs)
        d.rectangle([w - tw - 20, 6, w - 6, fs.size + 18], fill=(0, 0, 0, 160))
        d.text((w - tw - 13, 9), clock, font=fs, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def part(self, rec_dir, clips, roi="", object_model="", objects=True, use_vlm=False,
             vlm_no="show", mirror=True, label=""):
        """
        녹화 하나를 처음부터 판단하면서 clips 구간만 출력 영상에 이어 씀.
        판단 설정은 실시간 시스템 그대로. roi/objects/use_vlm은 화면에 보일 기능 범위만 고름.
        """
        rec = Path(rec_dir)
        with open(rec / "frames.csv", encoding="utf-8") as f:
            times = [float(r["t"]) for r in csv.DictReader(f)]
        model_path = object_model or OBJECT_MODEL
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack.reset_id()    # 장면마다 사람 ID를 1부터 (추적 ID 번호가 이전 장면에서 이어지지 않게, 판단과 무관)
        pose = PoseEstimator()
        fall = FallDetector()
        intr = IntrusionDetector(parse_roi(roi), stay_seconds_threshold=5) if roi else None
        obj = ObjectDetector(TARGET_OBJECTS, model_path=model_path) if objects else None
        disp = ObjectDisplacementDetector(TARGET_OBJECTS) if objects else None
        stillness = StillnessTrigger() if use_vlm else None
        vlm = self.vlm() if use_vlm else None
        print(f"[장면] {label or rec.name}: {rec.name} 구간 {clips} | 물건 "
              f"{model_path + ' ' + str(obj.conf_thresholds) if obj else '끔'} | 제한 구역 {'켬' if intr else '끔'} | "
              f"VLM {'켬' if use_vlm else '끔'} ('아님' 배너 {vlm_no})")

        cap = cv2.VideoCapture(str(rec / "video.mp4"))
        banners = []
        last = max(b for _, b in clips)
        for t in times:
            ret, frame = cap.read()
            if not ret or t > last:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            raw = frame.copy()
            new = []

            if intr:
                intr.draw(frame)
            people = pose.estimate(frame)
            if obj:
                dets = obj.detect(frame)
                res = disp.update(dets, now=t, person_boxes=[p["bbox"] for p in people])
                draw_objects(frame, dets, res)
                if not res["calibrating"]:
                    for cls, r in res["results"].items():
                        if r.get("new_event"):
                            new.append(("object", f"물건 이동 · {OBJ_NAMES.get(cls, cls)} "
                                                  f"{OBJ_STATUS.get(r['status'], r['status'])}"))
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
                        print(f"  [VLM] {t:.1f}초 ID{tid} {st['still']}초 정지, 규칙={fs['posture']} -> "
                              f"{ans['lying']} ({ans['latency']}초)")
                        if ans["lying"] is True:
                            new.append(("vlm_yes", "VLM 확인: 누운 자세 → 낙상 의심"))
                        elif ans["lying"] is False:
                            new.append(("vlm_no", "VLM 확인: 누운 자세 아님"))

            keep = any(a <= t <= b for a, b in clips)
            for kind, text in new:
                banners.append({"kind": kind, "text": text, "until": t + BANNER_SECONDS})
                shown = "숨김" if kind == "vlm_no" and vlm_no == "hide" else (
                    "작게" if kind == "vlm_no" and vlm_no == "small" else "표시")
                self.events.append({"장면": label, "녹화": rec.name, "영상 시각(초)": round(t, 1),
                                    "결과 영상 시각(초)": round(self.now, 1) if keep else "(잘림)",
                                    "종류": kind, "배너": text, "배너 표시": shown if keep else ""})
                print(f"  {t:6.1f}초  {text}{'' if keep else '  (저장 구간 밖)'}")
            banners = [b for b in banners if b["until"] > t]
            if keep:
                self._write(self.overlay(frame, banners, t, vlm_no))
        cap.release()

    def close(self):
        if self.writer:
            self.writer.close()
        with open(self.out_path.with_suffix(".events.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["장면", "녹화", "영상 시각(초)", "결과 영상 시각(초)",
                                              "종류", "배너", "배너 표시"])
            w.writeheader()
            w.writerows(self.events)
        print(f"  저장: {self.out_path} ({self.frames}프레임, {self.now:.1f}초, {self.fps:.2f}fps, "
              f"{'H.264' if find_ffmpeg() else 'mp4v'})")
        print(f"  이벤트 목록: {self.out_path.with_suffix('.events.csv')}")


def recording_fps(rec_dir):
    with open(Path(rec_dir) / "frames.csv", encoding="utf-8") as f:
        times = [float(r["t"]) for r in csv.DictReader(f)]
    return (len(times) - 1) / (times[-1] - times[0])   # 녹화의 평균 프레임 속도 (실제 속도로 재생)


def main():
    args = parse_args()
    rec = Path(args.rec_dir)
    r = Renderer(OUT_ROOT / rec.name / f"{rec.name}_{args.name}.mp4", fps=recording_fps(rec))
    r.part(rec, parse_clips(args.clips, args.start, args.end), roi=args.roi, object_model=args.object_model,
           objects=not args.no_objects, use_vlm=args.vlm, vlm_no=args.vlm_no, mirror=not args.no_mirror)
    r.close()


if __name__ == "__main__":
    main()
