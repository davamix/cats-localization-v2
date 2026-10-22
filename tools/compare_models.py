"""Compare exported NCNN models on the labelled Pi-camera captures of some splits (default: the test days).

Runs pi/detector.py (ncnn + numpy, the code the Pi runs) on the PC on the saved JPEGs, matches the boxes to the labels
in data/pi-camera/captures/cats-annotations.json as tools/label_report.py does (IoU >= 0.5, greedy), and prints per
day and in total, at each --conf: cats found with the right name, wrong name, missed, false boxes. At 0.5 (the app's
threshold) it checks the phase 6 target: >= 85% of the cats found, and at most 1 in 10 of the model's boxes wrong (a
false box or the wrong name).

Every disagreement at 0.5 (a missed cat, a false box or a wrong name, of any model) is drawn on sheets in
runs/compare/<date-time>/ for the full-size check of the labels: fix a label only when the picture clearly shows it is
wrong, the same rule for every model.

    python tools/compare_models.py yolo26n_320_scale0.9 yolo26n_320_scale0.9_pi_seed0     # run names in runs/train/
    python tools/compare_models.py path/to/model_folder --splits val                      # or exported model folders
    python tools/compare_models.py yolo26n_320_scale0.9 --pi                              # + the Pi's own pre-labels

--pi adds the detections the Pi saved with each capture (its model on the raw frame, before JPEG) as one more
"model": a check that the PC run of the same model gives the same numbers.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from label_report import COLORS, assign, load_labels, match
from pi_camera_splits import CAPTURES, REPO_ROOT, SPLITS, capture_days, split_days

sys.path.insert(0, str(REPO_ROOT / "pi"))
from detector import Detector  # noqa: E402  (pi/ is not a package)

TARGET_FOUND, TARGET_WRONG_BOXES, TARGET_CONF = 0.85, 0.10, 0.5
COLS, ROWS = 3, 3
ERROR_COLOR = (0, 0, 255)


def resolve_model(arg: str) -> tuple[Path, str]:
    """Run name in runs/train/ or an exported model folder -> (folder, short name)."""
    folder = Path(arg)
    if folder.is_dir():
        name = folder.parent.parent.name if folder.name == "best_ncnn_model" else folder.name
        return folder, name
    folder = REPO_ROOT / "runs" / "train" / arg / "weights" / "best_ncnn_model"
    if not (folder / "model.json").is_file():
        raise SystemExit(f"no exported model for {arg} ({folder}): run train/export.py {arg} first")
    return folder, arg


def summary(counts: Counter) -> str:
    cats = counts["right"] + counts["wrong"] + counts["missed"]
    boxes = counts["right"] + counts["wrong"] + counts["false"]
    return (f"found {counts['right']:3d}/{cats:<3d} ({counts['right'] / cats if cats else 0:4.0%})  "
            f"wrong name {counts['wrong']:2d}  missed {counts['missed']:3d}  false {counts['false']:3d} "
            f"({counts['false_empty']} on empty)  wrong boxes {counts['wrong'] + counts['false']}/{boxes}")


def target(counts: Counter) -> str:
    cats = counts["right"] + counts["wrong"] + counts["missed"]
    boxes = counts["right"] + counts["wrong"] + counts["false"]
    found = counts["right"] / cats if cats else 0.0
    wrong = (counts["wrong"] + counts["false"]) / boxes if boxes else 0.0
    met = found >= TARGET_FOUND and wrong <= TARGET_WRONG_BOXES
    return (f"target at {TARGET_CONF:g} ({TARGET_FOUND:.0%} found, <= {TARGET_WRONG_BOXES:.0%} wrong boxes): "
            f"{'MET' if met else 'not met'} ({found:.0%} found, {wrong:.0%} wrong boxes)")


def draw_tile(image_path: Path, labels, errors: list[str], boxes) -> np.ndarray:
    """The image with its labels (class colours, thin) and the models' wrong boxes (red), plus a caption."""
    image = cv2.imread(str(image_path))
    for cls, (x1, y1, x2, y2) in labels:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls], 1)
        cv2.putText(image, cls, (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[cls], 1)
    for text, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), ERROR_COLOR, 2)
        cv2.putText(image, text, (int(x1) + 2, int(y2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ERROR_COLOR, 1)
    lines = [image_path.stem] + errors
    caption = np.zeros((18 * len(lines) + 6, image.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(caption, line, (4, 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([image, caption])


def write_sheets(tiles: list[np.ndarray], out_dir: Path) -> list[Path]:
    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((height - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    sheets = []
    for start in range(0, len(tiles), COLS * ROWS):
        chunk = tiles[start:start + COLS * ROWS]
        chunk += [np.zeros_like(chunk[0])] * (-len(chunk) % COLS)
        rows = [np.hstack(chunk[i:i + COLS]) for i in range(0, len(chunk), COLS)]
        path = out_dir / f"disagreements_{start // (COLS * ROWS) + 1:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheets.append(path)
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="run names in runs/train/ or exported model folders")
    parser.add_argument("--pi", action="store_true", help="also score the pre-labels the Pi saved with each capture")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["test"])
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--annotations", type=Path, default=CAPTURES / "cats-annotations.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, help="sheet folder (default: runs/compare/<date-time>/)")
    args = parser.parse_args()
    if not args.models and not args.pi:
        parser.error("name at least one model (or --pi)")
    confs = sorted(set(args.conf) | {TARGET_CONF})

    labels = load_labels(args.annotations)
    day_splits = split_days(capture_days())
    images = {name: CAPTURES / f"{name[:4]}-{name[4:6]}-{name[6:8]}" / name for name in labels}
    images = {name: path for name, path in sorted(images.items()) if day_splits[path.parent.name][0] in args.splits}
    if not images:
        raise SystemExit(f"no labelled images in the splits {args.splits}")

    # predictions[model][image] = [(class, box, score), ...] with score >= the lowest --conf
    predictions = {}
    for arg in args.models:
        folder, name = resolve_model(arg)
        detector = Detector(folder, conf=min(confs), threads=args.threads)
        predictions[name] = {}
        for image_name, path in images.items():
            frame = cv2.imread(str(path))
            predictions[name][image_name] = [(detector.names[int(c)], (x1, y1, x2, y2), s)
                                             for x1, y1, x2, y2, s, c in detector.detect(frame).tolist()]
    if args.pi:
        predictions["pi-prelabels"] = {}
        for image_name, path in images.items():
            info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            predictions["pi-prelabels"][image_name] = [(d["class"], tuple(d["box"]), d["score"])
                                                       for d in info["detections"]]

    letters = {name: chr(ord("A") + i) for i, name in enumerate(predictions)}
    print(f"{len(images)} labelled images ({', '.join(args.splits)}), "
          f"{sum(len(b) for n, b in labels.items() if n in images)} cats")
    tiles = []
    for name, by_image in predictions.items():
        print(f"\n{letters[name]} = {name}")
        for conf in confs:
            per_day = defaultdict(Counter)
            for image_name, path in images.items():
                kept = [(c, b) for c, b, s in by_image[image_name] if s >= conf]
                per_day[path.parent.name].update(match(labels[image_name], kept))
            total = sum(per_day.values(), Counter())
            print(f"  conf {conf:<4g} all        {summary(total)}")
            for day, counts in sorted(per_day.items()):
                print(f"            {day} {summary(counts)}")
            if conf == TARGET_CONF:
                print(f"  {target(total)}")

    for image_name, path in images.items():
        errors, boxes = [], []
        for name, by_image in predictions.items():
            kept = [(c, b, s) for c, b, s in by_image[image_name] if s >= TARGET_CONF]
            assigned = assign(labels[image_name], [(c, b) for c, b, _ in kept])
            missed = [cls for (cls, _), i in zip(labels[image_name], assigned) if i is None]
            matched = {i: cls for (cls, _), i in zip(labels[image_name], assigned) if i is not None}
            wrong = [(c, b, s) for i, (c, b, s) in enumerate(kept) if i not in matched or matched[i] != c]
            boxes += [(f"{letters[name]} {c} {s:.2f}", b) for c, b, s in wrong]
            parts = ([f"missed {', '.join(missed)}"] if missed else []) + \
                    ([f"wrong {', '.join(f'{c} {s:.2f}' for c, _, s in wrong)}"] if wrong else [])
            if parts:
                errors.append(f"{letters[name]}: {'; '.join(parts)}")
        if errors:
            tiles.append(draw_tile(path, labels[image_name], errors, boxes))
    if tiles:
        out_dir = args.out or REPO_ROOT / "runs" / "compare" / time.strftime("%Y-%m-%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        sheets = write_sheets(tiles, out_dir)
        print(f"\n{len(tiles)} images with a disagreement at {TARGET_CONF:g} on {len(sheets)} sheet(s) in {out_dir}")


if __name__ == "__main__":
    main()
