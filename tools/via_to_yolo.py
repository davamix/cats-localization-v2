"""Convert the VIA 2.0.8 polygon annotations in data/ into an Ultralytics YOLO dataset.

Input (see data/README.md):
    data/<split>/cats-annotations.json      VIA project export, polygon regions with region_attributes.Class (rect
                                            regions work too: the Pi-camera boxes from tools/labelstudio_to_via.py)
    data/<split>/<Class>/<filename>         images

Output (rebuilt from scratch on every run):
    datasets/cats/images/{train,val}/<class folder>_<filename>
    datasets/cats/labels/{train,val}/<class folder>_<filename stem>.txt
    datasets/cats/cats.yaml

Image names are prefixed with their class folder because the validation frame names collide between folders
(Blacky/frame220.jpg and Niche/frame220.jpg are different images).

--pi-camera adds the labelled Pi-camera captures (data/pi-camera/captures/cats-annotations.json, images in
data/pi-camera/captures/<day>/, names prefixed with the day) to the 2020 photos, split by whole capture day with
tools/pi_camera_splits.py: train days -> train, the validation day -> val, test days -> test. Ultralytics trains on
train and picks best.pt on val; the test split is only for comparing models (never tune anything on it). Default
output: datasets/cats_pi/.

    python tools/via_to_yolo.py                    # detection labels (class cx cy w h)
    python tools/via_to_yolo.py --task segment     # polygon labels (class x1 y1 x2 y2 ...)
    python tools/via_to_yolo.py --pi-camera        # 2020 photos + Pi-camera captures -> datasets/cats_pi/
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from pi_camera_splits import CAPTURES as PI_CAMERA, REPO_ROOT, capture_days, split_days

# Class names in label-index order. Append new classes at the end so existing indices stay valid.
CLASSES = ["Blacky", "Niche"]

# YOLO split name -> folder under data/
SPLITS = {"train": "train", "val": "validation"}

ANNOTATION_FILE = "cats-annotations.json"


def find_image(split_dir: Path, filename: str, size: int) -> Path:
    """Locate an annotated image in the class folders of a split.

    VIA identifies images by file name + file size, and names repeat across class folders, so match both.
    """
    matches = [
        p
        for p in sorted(split_dir.glob(f"*/{filename}"))
        if p.stat().st_size == size
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one image {filename} ({size} bytes) in {split_dir}, found {matches}")
    return matches[0]


def polygon_points(region: dict, width: int, height: int) -> list[tuple[float, float]]:
    """Points of a VIA region: a polygon (2020 annotations) or a rect, as its 4 corners (Pi-camera boxes from
    tools/labelstudio_to_via.py)."""
    shape = region["shape_attributes"]
    if shape["name"] == "polygon":
        points = zip(shape["all_points_x"], shape["all_points_y"])
    elif shape["name"] == "rect":
        x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        raise ValueError(f"unsupported VIA shape: {shape['name']}")
    return [(min(max(x, 0), width), min(max(y, 0), height)) for x, y in points]


def label_line(class_id: int, points: list[tuple[float, float]], width: int, height: int, task: str) -> str:
    if task == "segment":
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in points)
        return f"{class_id} {coords}"
    xs, ys = [x for x, _ in points], [y for _, y in points]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box {x1, y1, x2, y2}")
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_split(split: str, split_dir: Path, out_dir: Path, task: str, keep=None) -> Counter:
    """Add the images of split_dir's VIA file to the YOLO split (keep(image path) -> bool selects some of them)."""
    via = json.loads((split_dir / ANNOTATION_FILE).read_text(encoding="utf-8"))
    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for key, entry in sorted(via.items()):
        if key.startswith("_via_"):
            continue
        source = find_image(split_dir, entry["filename"], entry["size"])
        if keep and not keep(source):
            continue
        with Image.open(source) as image:
            width, height = image.size

        lines = []
        for region in entry["regions"]:
            class_name = region["region_attributes"]["Class"]
            lines.append(label_line(CLASSES.index(class_name), polygon_points(region, width, height), width, height, task))
            counts[class_name] += 1

        name = f"{source.parent.name.lower()}_{source.name}"
        shutil.copy2(source, images_dir / name)
        # An empty label file marks a background image (no objects), which YOLO uses as a negative.
        (labels_dir / name).with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts["images"] += 1
        counts["empty"] += not lines
    return counts


def write_dataset_yaml(out_dir: Path, test: bool = False) -> Path:
    yaml_path = out_dir / f"{out_dir.name}.yaml"
    lines = [
        "# Generated by tools/via_to_yolo.py - do not edit, re-run the script instead.",
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        *(["test: images/test  # Pi-camera test days: only for comparing models"] if test else []),
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(CLASSES)),
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data", help="folder with the VIA splits")
    parser.add_argument("--out", type=Path, help="output dataset folder (default: datasets/cats, or datasets/cats_pi "
                                                 "with --pi-camera)")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--pi-camera", action="store_true",
                        help="add the labelled Pi-camera captures, split by capture day (train / val / test)")
    args = parser.parse_args()
    if args.pi_camera and args.task == "segment":
        parser.error("the Pi-camera labels are boxes: --pi-camera needs --task detect")

    out_dir = (args.out or REPO_ROOT / "datasets" / ("cats_pi" if args.pi_camera else "cats")).resolve()
    if out_dir == REPO_ROOT or out_dir in args.data.resolve().parents or out_dir == args.data.resolve():
        parser.error(f"refusing to overwrite {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    def report(split, source, counts):
        per_class = ", ".join(f"{name}: {counts[name]}" for name in CLASSES)
        print(f"{split:5} {source:9} {counts['images']:4d} images ({counts['empty']} empty) | {per_class}")

    for split, folder in SPLITS.items():
        report(split, "2020", convert_split(split, args.data / folder, out_dir, args.task))
    if args.pi_camera:
        day_splits = split_days(capture_days(PI_CAMERA))
        for split in ("train", "val", "test"):
            counts = convert_split(split, PI_CAMERA, out_dir, args.task,
                                   keep=lambda source, split=split: day_splits[source.parent.name][0] == split)
            report(split, "Pi-camera", counts)
    print(f"dataset: {write_dataset_yaml(out_dir, test=args.pi_camera)}")


if __name__ == "__main__":
    main()
