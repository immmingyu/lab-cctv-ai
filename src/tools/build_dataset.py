import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from src.tools.extract_frames import CLASSES, OUT_ROOT as FRAMES_ROOT

DATASET_ROOT = Path("data/finetune/dataset")
SCHOOL_TEST_ROOT = Path("data/finetune/school_test")


def parse_args():
    p = argparse.ArgumentParser(description="라벨링한 세션 폴더로 YOLO 학습 데이터셋 만들기 (세션 단위 학습/검증 분리)")
    p.add_argument("--train", nargs="+", required=True, help="학습용 세션 (frames 아래 폴더 이름 또는 경로)")
    p.add_argument("--val", nargs="+", required=True, help="검증용 세션 (학습과 겹치면 안 됨)")
    p.add_argument("--out", default=str(DATASET_ROOT), help="만들 데이터셋 폴더 (기본 data/finetune/dataset)")
    return p.parse_args()


def session_dir(name):
    p = Path(name)
    return p if p.exists() else FRAMES_ROOT / name


def check_session(d):
    """세션 폴더 검사. 문제가 있으면 바로 멈춤 (학습 데이터가 잘못 섞이는 걸 막기 위함)."""
    if not d.is_dir():
        raise SystemExit(f"세션 폴더가 없습니다: {d}")
    if "school" in d.name.lower() or SCHOOL_TEST_ROOT.resolve() in d.resolve().parents:
        raise SystemExit(f"학교 영상은 평가 전용이라 학습 데이터에 넣을 수 없습니다: {d}")
    cls_file = d / "classes.txt"
    if not cls_file.exists():
        raise SystemExit(f"classes.txt가 없습니다: {d} (X-AnyLabeling에서 YOLO 형식으로 저장했는지 확인)")
    names = [line.strip() for line in cls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if names != CLASSES:
        raise SystemExit(f"클래스 순서가 {CLASSES}가 아닙니다: {d} -> {names}")


def read_labels(txt):
    """YOLO 라벨 한 파일 읽기. 형식이 틀리면 (None, 오류 설명)."""
    boxes = []
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
        boxes.append(cls)
    return boxes, None


def collect(d):
    """(이미지, 라벨) 목록과 통계. 라벨 파일(.txt)이 없는 이미지는 '아직 안 봄'으로 보고 건너뜀."""
    pairs, errors, unlabeled, boxes, negatives = [], [], 0, Counter(), 0
    for img in sorted(d.glob("*.jpg")):
        txt = img.with_suffix(".txt")
        if not txt.exists():
            unlabeled += 1
            continue
        cls_list, err = read_labels(txt)
        if err:
            errors.append(err)
            continue
        boxes.update(CLASSES[c] for c in cls_list)
        negatives += not cls_list           # 빈 라벨 = 확인했는데 책/노트북 없음 (비슷한 물건만 있는 장면 등)
        pairs.append((img, txt))
    return pairs, errors, unlabeled, boxes, negatives


def main():
    args = parse_args()
    train = [session_dir(s) for s in args.train]
    val = [session_dir(s) for s in args.val]
    overlap = {d.resolve() for d in train} & {d.resolve() for d in val}
    if overlap:
        raise SystemExit(f"같은 세션이 학습과 검증에 모두 있습니다: {[p.name for p in overlap]}")
    for d in train + val:
        check_session(d)

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
            pairs, errors, unlabeled, boxes, negatives = collect(d)
            all_errors += [f"{d.name}/{e}" for e in errors]
            for img, txt in pairs:
                if (out / "images" / split / img.name).exists():
                    raise SystemExit(f"파일 이름이 다른 세션과 겹칩니다: {d.name}/{img.name} "
                                     "(extract_frames로 뽑으면 녹화 이름이 들어가 겹치지 않음)")
                shutil.copy2(img, out / "images" / split / img.name)
                shutil.copy2(txt, out / "labels" / split / txt.name)
            rows.append([split, d.name, len(pairs), boxes["book"], boxes["laptop"], negatives, unlabeled, len(errors)])

    if all_errors:
        print("[라벨 오류] 아래 파일은 빼고 만들었습니다:")
        for e in all_errors[:20]:
            print("  ", e)

    yaml = (f"path: {out.resolve().as_posix()}\n"
            "train: images/train\nval: images/val\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    (out / "dataset.yaml").write_text(yaml, encoding="utf-8")
    # 재생 평가가 '학습에 쓴 영상/장소'를 표시할 수 있게 세션 이름을 남김 (세션 이름 = 녹화 이름)
    (out / "sessions.json").write_text(json.dumps(
        {"train": [d.name for d in train], "val": [d.name for d in val]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with open(out / "split.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["구분", "세션", "이미지", "book 박스", "laptop 박스", "빈 라벨(물건 없음)", "라벨 없음(건너뜀)", "라벨 오류"])
        w.writerows(rows)

    print(f"[데이터셋] {out}")
    for r in rows:
        print(f"  {r[0]:<5} {r[1]:<40} 이미지 {r[2]:4d}  book {r[3]:4d}  laptop {r[4]:4d}  "
              f"빈 라벨 {r[5]:3d}  라벨 없음 {r[6]:3d}  오류 {r[7]}")


if __name__ == "__main__":
    main()
