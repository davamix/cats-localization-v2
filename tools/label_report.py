"""Report on the labelled Pi-camera captures, and draw them for a visual check.

    python tools/label_report.py                                            # after tools/labelstudio_to_via.py
    python tools/label_report.py --export data/annotations/<export>.json    # + the review counts of the export

Reads data/pi-camera/captures/cats-annotations.json (from tools/labelstudio_to_via.py) and the capture JSON files next
to the images, and prints:
- per capture day and per split (tools/pi_camera_splits.py): images, empty scenes, images with both cats, boxes per
  class, small boxes (narrower than --small of the frame width); val/test days that lack a category are flagged;
- how the review changed the pre-labels: kept, edited, deleted, boxes added where the model had none;
- the deployed model (v0.1.0, which made the pre-labels on the Pi on exactly these frames) against the labels at
  confidence 0.25 and 0.5, IoU >= 0.5: cats found with the right / wrong name, missed, false boxes; per split;
- checks: every labelled image has its capture JSON; with --export, the submitted tasks match the annotation file.
Then draws every labelled image with its boxes onto contact sheets in --sheets (default runs/labels/<today>/, not in
git) and prints their paths: look at them to catch misplaced boxes and missed cats.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from pi_camera_splits import CAPTURES, REPO_ROOT, SPLITS, split_days

CLASSES = ["Blacky", "Niche"]
COLORS = {"Blacky": (0, 140, 255), "Niche": (255, 200, 0)}  # BGR, as in tools/visualize_labels.py
IOU_MATCH = 0.5
CONFIDENCES = (0.25, 0.5)
THUMB = (320, 240)
COLS, ROWS = 6, 4


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def load_labels(annotations: Path) -> dict[str, list[tuple[str, tuple]]]:
    """VIA file -> {image file name: [(class, (x1, y1, x2, y2)), ...]} (empty list = empty scene)."""
    via = json.loads(annotations.read_text(encoding="utf-8"))
    labels = {}
    for key, entry in via.items():
        if key.startswith("_via_"):
            continue
        boxes = []
        for region in entry["regions"]:
            s = region["shape_attributes"]
            if s["name"] != "rect":
                raise ValueError(f"{entry['filename']}: expected rect regions, got {s['name']}")
            boxes.append((region["region_attributes"]["Class"], (s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"])))
        labels[entry["filename"]] = boxes
    return labels


def assign(labels, predictions) -> list[int | None]:
    """Greedy matching of label boxes to predicted boxes (best IoU first come) -> for each label box, the index of its
    predicted box (IoU >= IOU_MATCH) or None."""
    assigned, used = [], set()
    for cls, box in labels:
        best, best_i = 0.0, None
        for i, (pcls, pbox) in enumerate(predictions):
            if i not in used and (value := iou(box, pbox)) > best:
                best, best_i = value, i
        if best >= IOU_MATCH:
            used.add(best_i)
            assigned.append(best_i)
        else:
            assigned.append(None)
    return assigned


def match(labels, predictions) -> Counter:
    """Counts of the greedy matching: right / wrong (name) / missed label boxes, false (unmatched) predicted boxes."""
    counts = Counter()
    assigned = assign(labels, predictions)
    for (cls, _), i in zip(labels, assigned):
        if i is None:
            counts["missed"] += 1
        else:
            counts["right" if predictions[i][0] == cls else "wrong"] += 1
    counts["false"] += len(predictions) - sum(i is not None for i in assigned)
    counts["false_empty"] += 0 if labels else len(predictions)
    return counts


def review_change(labels, prelabels) -> str:
    """How the review changed the pre-labels of one image (boxes compared within 1.5 px)."""
    if not prelabels:
        return "no pre-label, boxes added" if labels else "no pre-label, kept empty"
    if not labels:
        return "pre-labels deleted (empty scene)"
    same = len(labels) == len(prelabels) and all(
        any(c == pc and max(abs(u - v) for u, v in zip(b, pb)) <= 1.5 for pc, pb in prelabels) for c, b in labels)
    return "pre-labels kept" if same else "pre-labels edited"


def draw_sheets(items, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()
    tiles = []
    for image_path, boxes, caption in items:
        image = cv2.imread(str(image_path))
        for cls, (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls], 2)
            cv2.putText(image, cls, (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 1)
        image = cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, THUMB[1] - 18), (THUMB[0], THUMB[1]), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, caption, (4, THUMB[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    sheets, per_sheet = [], COLS * ROWS
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        chunk += [np.zeros_like(tiles[0])] * (-len(chunk) % COLS)
        rows = [np.hstack(chunk[i:i + COLS]) for i in range(0, len(chunk), COLS)]
        path = out_dir / f"sheet_{start // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheets.append(path)
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=CAPTURES / "cats-annotations.json")
    parser.add_argument("--export", type=Path, help="the Label Studio JSON export the annotation file was made from")
    parser.add_argument("--small", type=float, default=0.15, help="a box narrower than this share of the width is small")
    parser.add_argument("--prelabel-conf", type=float, default=0.25, help="score of the imported pre-labels")
    parser.add_argument("--sheets", type=Path, default=REPO_ROOT / "runs" / "labels" / time.strftime("%Y-%m-%d"))
    args = parser.parse_args()

    labels = load_labels(args.annotations)
    sidecars = {p.with_suffix(".jpg").name: p for p in CAPTURES.glob("*/*.json")}
    missing = sorted(name for name in labels if name not in sidecars)
    if missing:
        raise SystemExit(f"labelled images without a capture JSON (pull them first?): {', '.join(missing[:5])}")
    splits = split_days(p.parent.name for p in sidecars.values())

    per_day, per_split = defaultdict(Counter), defaultdict(Counter)
    model = {conf: defaultdict(Counter) for conf in CONFIDENCES}
    review, sheet_items = Counter(), []
    for name in sorted(labels):
        boxes, sidecar = labels[name], sidecars[name]
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        day = sidecar.parent.name
        split = splits[day][0]
        stats = Counter(images=1, empty=not boxes, both=len({c for c, _ in boxes}) == 2,
                        small=sum((b[2] - b[0]) < args.small * info["width"] for _, b in boxes),
                        **{c: sum(cls == c for cls, _ in boxes) for c in CLASSES})
        per_day[day].update(stats)
        per_split[split].update(stats)
        detections = [(d["class"], tuple(d["box"])) for d in info["detections"]]
        for conf in CONFIDENCES:
            counts = match(boxes, [d for d, s in zip(detections, info["detections"]) if s["score"] >= conf])
            model[conf]["all"].update(counts)
            model[conf][split].update(counts)
        review[review_change(boxes, [d for d, s in zip(detections, info["detections"])
                                     if s["score"] >= args.prelabel_conf])] += 1
        sheet_items.append((sidecar.with_suffix(".jpg"), boxes, f"{day} {split} {Path(name).stem[9:]}"))

    columns = ["images", "empty", "both", *CLASSES, "small"]
    header = f"{'':12} {'split':6}" + "".join(f"{c:>8}" for c in columns)
    print(f"labelled images per capture day ({args.annotations.relative_to(REPO_ROOT)}; small = box < "
          f"{args.small:.0%} of the width):\n{header}")
    for day in sorted(per_day):
        print(f"{day:12} {splits[day][0]:6}" + "".join(f"{per_day[day][c]:8d}" for c in columns))
    for split in SPLITS:
        print(f"{'total':12} {split:6}" + "".join(f"{per_split[split][c]:8d}" for c in columns))
    total = sum(per_split.values(), Counter())
    print(f"{'total':12} {'all':6}" + "".join(f"{total[c]:8d}" for c in columns))
    for split in ("val", "test"):
        lacking = [c for c in ("empty", "both", "small", *CLASSES) if not per_split[split][c]]
        if lacking:
            print(f"  {split} has no {', '.join(lacking)} yet: capture them on a {split} day")

    print("\nreview of the pre-labels (score >= {:g}): ".format(args.prelabel_conf)
          + ", ".join(f"{k} {v}" for k, v in review.most_common()))

    print(f"\ndeployed model (pre-labels on the Pi) vs the labels, IoU >= {IOU_MATCH}:")
    for conf in CONFIDENCES:
        for group in ("all", *SPLITS):
            c = model[conf][group]
            cats = c["right"] + c["wrong"] + c["missed"]
            if group != "all" and not per_split[group]["images"]:
                continue
            found = f"{c['right']}/{cats} ({c['right'] / cats:.0%})" if cats else "0/0"
            print(f"  conf {conf:<4} {group:5}: found {found}, wrong name {c['wrong']}, missed {c['missed']}, "
                  f"false boxes {c['false']} ({c['false_empty']} on empty scenes)")

    if args.export:
        tasks = json.loads(args.export.read_text(encoding="utf-8"))
        status = Counter()
        for task in tasks:
            done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
            status["submitted" if done else "skipped" if task.get("annotations") else "not reviewed"] += 1
        print(f"\nexport {args.export.name}: {len(tasks)} tasks: "
              + ", ".join(f"{k} {v}" for k, v in status.most_common()))
        if status["submitted"] != len(labels):
            print(f"  MISMATCH: {status['submitted']} submitted tasks but {len(labels)} images in the annotation file: "
                  "re-run tools/labelstudio_to_via.py on this export")

    sheets = draw_sheets(sheet_items, args.sheets)
    print(f"\n{len(sheet_items)} labelled images drawn on {len(sheets)} sheet(s): "
          + ", ".join(str(s.relative_to(REPO_ROOT)) for s in sheets))


if __name__ == "__main__":
    main()
