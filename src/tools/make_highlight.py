"""
QR용 핵심 장면 영상 (1~1.5분): 여러 녹화의 구간을 장면 제목과 함께 이어 붙임.

  python -m src.tools.make_highlight                 # VLM '아님' 배너는 작게 (시연용)
  python -m src.tools.make_highlight --vlm-no show   # 모두 그대로 표시

- 장면마다 판단은 그 녹화의 처음부터 돌리고 정해진 구간만 저장 (render_video와 같음, 판단 설정 그대로)
- 물건 이동은 학교 시험 영상에서 가져옴 (파인튜닝 모델, 배너가 시도 직후에 뜨는 시도):
    #5 (책상 위 다른 자리, moved): 108.7초 내려놓음 -> 110.7초 이벤트, 이후 책 계속 검출
    #9 (들고 나감, missing): 233.6초 화면에서 사라짐 -> 237.6초 이벤트, 중간에 상태가 흔들리지 않음
- 나머지는 시연 녹화(demo_school). 이 녹화의 물건 장면은 쓰지 않아서 물건 표시는 끔
"""
import argparse

from src.tools.render_video import OUT_ROOT, VLM_NO_MODES, Renderer

DEMO = "data/recordings/demo_school_20261104_145408"
SCHOOL = "data/recordings/eval2_object_school_20261104_135118"
ROI = "271,306 267,359 388,364 380,315"
FT_MODEL = "runs/finetune/book_laptop_v1/weights/best.pt"

# (제목, 부제, [장면들]) — 장면: render_video.Renderer.part 인자
PLAN = [
    ("① 제한 구역 접근", "5초 이상 머물면 알림",
     [dict(rec_dir=DEMO, clips=[(19.5, 29.0)], roi=ROI, objects=False, use_vlm=True)]),
    ("② 물건 이동", "다른 자리로 옮기기 · 들고 나가기",
     [dict(rec_dir=SCHOOL, clips=[(106.0, 114.5)], object_model=FT_MODEL),
      dict(rec_dir=SCHOOL, clips=[(230.5, 241.0)], object_model=FT_MODEL)]),
    ("③ 옆으로 눕기 낙상", "누운 시간에 따라 위험도 MEDIUM → HIGH",
     [dict(rec_dir=DEMO, clips=[(43.5, 62.0)], roi=ROI, objects=False, use_vlm=True)]),
    ("④ 일직선 낙상 · VLM 확인", "규칙이 누움으로 못 본 자세를 VLM이 다시 확인",
     [dict(rec_dir=DEMO, clips=[(85.0, 99.6)], roi=ROI, objects=False, use_vlm=True)]),
]


def main():
    p = argparse.ArgumentParser(description="QR용 핵심 장면 영상 만들기")
    p.add_argument("--vlm-no", choices=VLM_NO_MODES, default="small", help="VLM '아님' 배너 (기본 small)")
    p.add_argument("--title-seconds", type=float, default=1.5)
    p.add_argument("--name", default="highlight_qr")
    args = p.parse_args()

    r = Renderer(OUT_ROOT / args.name / f"{args.name}.mp4", fps=30.0)
    for title, sub, scenes in PLAN:
        r.title(title, args.title_seconds, sub=sub)
        for s in scenes:
            r.part(**s, vlm_no=args.vlm_no, label=title)
    r.close()


if __name__ == "__main__":
    main()
