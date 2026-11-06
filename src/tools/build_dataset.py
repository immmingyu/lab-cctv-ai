import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from src.tools.extract_frames import CLASSES, OUT_ROOT as FRAMES_ROOT

LABELS_ROOT = Path("data/finetune/labels")      # X-AnyLabeling YOLO 내보내기 위치 (세션 폴더 이름 그대로)
DATASET_ROOT = Path("data/finetune/dataset")
SCHOOL_TEST_ROOT = Path("data/finetune/school_test")
FRAME_NAME = re.compile(r"^(.*)_f\d{6}_t[\d.]+$")   # extract_frames 파일 이름: <녹화이름>_f000123_t00004.1


def parse_args():
    p = argparse.ArgumentParser(description="라벨링한 세션으로 YOLO 학습 데이터셋 만들기 (세션 단위 학습/검증 분리)")
    p.add_argument("--train", nargs="+", required=True, help="학습용 세션 (frames 아래 폴더 이름 또는 경로)")
    p.add_argument("--val", nargs="+", required=True, help="검증용 세션 (학습과 겹치면 안 됨)")
    p.add_argument("--labels-root", default=str(LABELS_ROOT),
                   help="YOLO 라벨(.txt)을 내보낸 상위 폴더 (기본 data/finetune/labels/<세션 폴더 이름>)")
    p.add_argument("--out", default=str(DATASET_ROOT), help="만들 데이터셋 폴더 (기본 data/finetune/dataset)")
    return p.parse_args()


def session_dir(name):
    p = Path(name)
    return p if p.exists() else FRAMES_ROOT / name


def recording_of(d):
    """세션 폴더의 원래 녹화 이름 (이미지 파일 이름 앞부분). 세션 폴더가 _v3처럼 달라도 녹화는 같음."""
    names = {m.group(1) for img in d.glob("*.jpg") if (m := FRAME_NAME.match(img.stem))}
    return sorted(names)


def check_session(d, labels_dir):
    """세션 폴더 검사. 문제가 있으면 바로 멈춤 (학습 데이터가 잘못 섞이는 걸 막기 위함)."""
    if not d.is_dir():
        raise SystemExit(f"세션 폴더가 없습니다: {d}")
    if "school" in d.name.lower() or SCHOOL_TEST_ROOT.resolve() in d.resolve().parents:
        raise SystemExit(f"학교 영상은 평가 전용이라 학습 데이터에 넣을 수 없습니다: {d}")
    if any("school" in r.lower() for r in recording_of(d)):
        raise SystemExit(f"학교 녹화에서 뽑은 프레임이 있습니다: {d}")
    cls_file = d / "classes.txt"
    if cls_file.exists():
        names = [line.strip() for line in cls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names != CLASSES:
            raise SystemExit(f"클래스 순서가 {CLASSES}가 아닙니다: {cls_file} -> {names}")
    if not labels_dir.is_dir():
        raise SystemExit(f"라벨 폴더가 없습니다: {labels_dir} (X-AnyLabeling에서 YOLO 형식으로 내보냈는지 확인)")


def read_txt(txt):
    """YOLO 라벨 한 파일 -> 클래스 번호 목록. 형식이 틀리면 (None, 오류 설명)."""
    classes = []
    for n, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            return None, f"{txt.name}:{n} 숫자가 아님"
        if len(vals) != 4 or not 0 <= cls < len(CLASSES) or not all(0 <= v <= 1 for v in vals):
            return None, f"{txt.name}:{n} 형식 오류 '{line}'"
        classes.append(cls)
    return classes, None


def read_json(js):
    """X-AnyLabeling .json -> 클래스 번호 목록 (사각형만). 모르는 라벨이면 (None, 오류 설명)."""
    data = json.loads(js.read_text(encoding="utf-8"))
    classes = []
    for s in data.get("shapes", []):
        if s.get("shape_type") != "rectangle":
            return None, f"{js.name} 사각형이 아닌 모양 '{s.get('shape_type')}'"
        if s.get("label") not in CLASSES:
            return None, f"{js.name} 모르는 라벨 '{s.get('label')}' (book, laptop만)"
        classes.append(CLASSES.index(s["label"]))
    return classes, None


def collect(d, labels_dir):
    """
    확인한 이미지 = frames 폴더에 X-AnyLabeling .json이 있는 이미지 (passed 포함).
      .json 있음 + .txt 비어 있음 -> 확인했는데 책/노트북 없음 (학습에 포함)
      .json 없음 -> 안 본 이미지 (내보내기가 만든 빈 .txt가 있어도 제외)
    .json의 박스와 .txt의 줄(클래스)이 다르면 다시 내보내기를 안 한 것 -> 오류.
    """
    stats = Counter()
    pairs, errors = [], []
    for img in sorted(d.glob("*.jpg")):
        stats["이미지"] += 1
        js, txt = img.with_suffix(".json"), labels_dir / (img.stem + ".txt")
        if not js.exists():
            stats["안 봄"] += 1
            continue
        js_cls, err = read_json(js)
        if err:
            errors.append(err)
            continue
        if not txt.exists():
            errors.append(f"{img.stem}: 확인했는데 내보낸 .txt가 없음 (다시 내보내기)")
            continue
        txt_cls, err = read_txt(txt)
        if err:
            errors.append(err)
            continue
        if sorted(js_cls) != sorted(txt_cls):
            errors.append(f"{img.stem}: .json 박스 {len(js_cls)}개와 .txt {len(txt_cls)}줄이 다름 (라벨 수정 후 다시 내보내기)")
            continue
        stats["확인"] += 1
        stats["책 있음" if 0 in txt_cls else "책 없음"] += 1
        stats["노트북 있음"] += 1 in txt_cls
        stats["book 박스"] += txt_cls.count(0)
        stats["laptop 박스"] += txt_cls.count(1)
        pairs.append((img, txt))
    return pairs, errors, stats


def main():
    args = parse_args()
    labels_root = Path(args.labels_root)
    train = [session_dir(s) for s in args.train]
    val = [session_dir(s) for s in args.val]
    overlap = {d.resolve() for d in train} & {d.resolve() for d in val}
    if overlap:
        raise SystemExit(f"같은 세션이 학습과 검증에 모두 있습니다: {[p.name for p in overlap]}")
    rec_overlap = {r for d in train for r in recording_of(d)} & {r for d in val for r in recording_of(d)}
    if rec_overlap:
        raise SystemExit(f"같은 녹화에서 뽑은 프레임이 학습과 검증에 모두 있습니다: {sorted(rec_overlap)}")
    for d in train + val:
        check_session(d, labels_root / d.name)

    out = Path(args.out)
    if out.exists():
        if not (out / "dataset.yaml").exists():
            raise SystemExit(f"{out}는 이 도구가 만든 폴더가 아니라서 지우지 않습니다. 다른 --out을 쓰세요.")
        shutil.rmtree(out)                  # 이 도구가 만든 데이터셋만 새로 만듦

    rows, all_errors = [], []
    for split, sessions in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for d in sessions:
            pairs, errors, st = collect(d, labels_root / d.name)
            all_errors += [f"{d.name}/{e}" for e in errors]
            for img, txt in pairs:
                if (out / "images" / split / img.name).exists():
                    raise SystemExit(f"파일 이름이 다른 세션과 겹칩니다: {d.name}/{img.name} "
                                     "(같은 녹화에서 뽑은 세션 폴더를 두 번 넣었는지 확인)")
                shutil.copy2(img, out / "images" / split / img.name)
                shutil.copy2(txt, out / "labels" / split / txt.name)
            rows.append([split, d.name, st["이미지"], st["확인"], st["책 있음"], st["책 없음"], st["노트북 있음"],
                         st["book 박스"], st["laptop 박스"], st["안 봄"], len(errors)])

    if all_errors:
        print(f"[라벨 오류 {len(all_errors)}건] 아래 이미지는 빼고 만들었습니다:")
        for e in all_errors[:20]:
            print("  ", e)

    yaml = (f"path: {out.resolve().as_posix()}\n"
            "train: images/train\nval: images/val\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    (out / "dataset.yaml").write_text(yaml, encoding="utf-8")
    # 재생 평가가 '학습에 쓴 영상/장소'를 표시할 수 있게 세션 폴더와 원래 녹화 이름을 남김
    (out / "sessions.json").write_text(json.dumps(
        {"train": [d.name for d in train], "val": [d.name for d in val],
         "recordings": {"train": sorted({r for d in train for r in recording_of(d)}),
                        "val": sorted({r for d in val for r in recording_of(d)})}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "split.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["구분", "세션", "이미지", "확인", "책 있음", "책 없음", "노트북 있음",
                    "book 박스", "laptop 박스", "안 봄(제외)", "라벨 오류(제외)"])
        w.writerows(rows)

    print(f"[데이터셋] {out}")
    for r in rows:
        print(f"  {r[0]:<5} {r[1]}")
        print(f"        이미지 {r[2]}장 중 확인 {r[3]}장 (책 있음 {r[4]}, 책 없음 {r[5]}, 노트북 있음 {r[6]}) "
              f"/ 안 봄 {r[9]}장 제외 / 라벨 오류 {r[10]}건 제외")
    total = [sum(r[3] for r in rows if r[0] == s) for s in ("train", "val")]
    print(f"  합계: 학습 {total[0]}장, 검증 {total[1]}장")


if __name__ == "__main__":
    main()
