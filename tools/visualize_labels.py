"""Draw YOLO labels on the dataset images for a visual check.

Writes one annotated image per input image and contact sheets (grids of thumbnails) per split:
    datasets/cats/preview/<split>/<image>.jpg
    datasets/cats/preview/<split>_sheet_01.jpg, ...

    python tools/visualize_labels.py
    python tools/visualize_labels.py --dataset datasets/cats --splits val --cols 6 --rows 4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# BGR colours per class index; cycles if there are more classes.
COLOURS = [(0, 140, 255), (255, 200, 0), (80, 220, 80), (200, 80, 255), (60, 60, 230)]


def draw_labels(image: np.ndarray, label_file: Path, names: dict) -> np.ndarray:
    height, width = image.shape[:2]
    thickness = max(2, round(width / 400))
    font_scale = width / 1000
    lines = label_file.read_text(encoding="utf-8").split("\n") if label_file.exists() else []
    for line in filter(None, lines):
        values = line.split()
        class_id, coords = int(values[0]), [float(v) for v in values[1:]]
        colour = COLOURS[class_id % len(COLOURS)]
        if len(coords) == 4:  # detection: cx cy w h
            cx, cy, w, h = coords
            x1, y1 = int((cx - w / 2) * width), int((cy - h / 2) * height)
            x2, y2 = int((cx + w / 2) * width), int((cy + h / 2) * height)
        else:  # segmentation: polygon x1 y1 x2 y2 ...
            points = (np.array(coords).reshape(-1, 2) * [width, height]).astype(np.int32)
            cv2.polylines(image, [points], isClosed=True, color=colour, thickness=thickness)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
        text = names.get(class_id, str(class_id))
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(y1, th + baseline)
        cv2.rectangle(image, (x1, ty - th - baseline), (x1 + tw, ty), colour, -1)
        cv2.putText(image, text, (x1, ty - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return image


def fit_tile(image: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    """Scale an image to fit a tile, keeping the aspect ratio, and pad the rest."""
    scale = min(tile_w / image.shape[1], tile_h / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
    tile[:resized.shape[0], :resized.shape[1]] = resized
    return tile


def write_sheets(images: list[tuple[str, np.ndarray]], out_prefix: Path, cols: int, rows: int, tile_w: int):
    per_sheet = cols * rows
    tile_h = tile_w * 9 // 16
    for start in range(0, len(images), per_sheet):
        sheet = np.full((rows * tile_h, cols * tile_w, 3), 40, dtype=np.uint8)
        for i, (name, image) in enumerate(images[start:start + per_sheet]):
            r, c = divmod(i, cols)
            tile = fit_tile(image, tile_w, tile_h)
            cv2.putText(tile, name, (5, tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
        cv2.imwrite(str(out_prefix.parent / f"{out_prefix.name}_sheet_{start // per_sheet + 1:02d}.jpg"), sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "datasets" / "cats")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=400)
    args = parser.parse_args()

    names = yaml.safe_load((args.dataset / f"{args.dataset.name}.yaml").read_text(encoding="utf-8"))["names"]
    preview_dir = args.dataset / "preview"
    for split in args.splits:
        out_dir = preview_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        thumbnails = []
        for image_path in sorted((args.dataset / "images" / split).glob("*.jpg")):
            label_file = args.dataset / "labels" / split / f"{image_path.stem}.txt"
            image = draw_labels(cv2.imread(str(image_path)), label_file, names)
            cv2.imwrite(str(out_dir / image_path.name), image)
            scale = min(1.0, args.thumb_width / image.shape[1])
            thumbnails.append((image_path.stem, cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)))
        write_sheets(thumbnails, preview_dir / split, args.cols, args.rows, args.thumb_width)
        print(f"{split}: {len(thumbnails)} previews in {out_dir}")


if __name__ == "__main__":
    main()
