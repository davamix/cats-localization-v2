"""Evaluate trained weights on the validation set: mAP, per-class precision/recall, confusion matrix, failure cases.

mAP50 and mAP50-95 come from Ultralytics validation (confidence 0.001, as usual for mAP). Precision, recall and
the confusion matrix are computed here at a fixed confidence (--conf), the way the detector runs on the Pi. Each
prediction is matched to the unmatched ground-truth box it overlaps most (IoU >= --iou, any class), so a cat found
with the wrong name counts as a class confusion instead of a miss plus a false positive.

--shrink also evaluates on copies of the validation images scaled down by each factor and pasted at a random
(seeded) position on a grey canvas of the original size: a rough proxy for cats far from the camera, since every
cat in the dataset fills a large part of the frame. The copies are written to datasets/<dataset>_val_x<factor>/.

Output in runs/eval/<name>/ (default name: <run>_imgsz<imgsz>, plus _ncnn for an exported NCNN model folder):
    summary.json                    all metrics, per shrink factor
    x<factor>/failures/<image>.jpg  images with a wrong class, missed cat, false positive or duplicate box
                                    (ground truth in white, predictions in class colours)

    python train/evaluate.py yolo26n_320                  # run name in runs/train/; imgsz from its args.yaml
    python train/evaluate.py yolo26n_320 --shrink 1 0.5 0.25
    python train/evaluate.py path/to/best.pt --imgsz 416 --conf 0.4
    python train/evaluate.py runs/train/yolo26n_320/weights/best_ncnn_model     # exported model (train/export.py)
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_RUNS_DIR = REPO_ROOT / "runs" / "train"
EVAL_RUNS_DIR = REPO_ROOT / "runs" / "eval"

GREY = 114  # Ultralytics letterbox padding value
MAX_PRINTED_FAILURES = 20
# BGR colours per class index, same as tools/visualize_labels.py; cycles if there are more classes.
COLOURS = [(0, 140, 255), (255, 200, 0), (80, 220, 80), (200, 80, 255), (60, 60, 230)]


def resolve_weights(weights: str) -> tuple[Path, str, int | None]:
    """Return (weights path, run name, training imgsz) for a weights file, an exported model folder
    (e.g. best_ncnn_model/) or a run name in runs/train/."""
    path = Path(weights)
    if not path.exists():
        path = TRAIN_RUNS_DIR / weights / "weights" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"no weights file or model folder {weights} and no run {path.parent.parent}")
    args_file = path.parent.parent / "args.yaml"
    if args_file.is_file():
        return path, path.parent.parent.name, yaml.safe_load(args_file.read_text(encoding="utf-8"))["imgsz"]
    return path, path.stem, None


def label_path(image: Path) -> Path:
    # Same convention as Ultralytics: .../images/<split>/x.jpg -> .../labels/<split>/x.txt
    return image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"


def read_labels(image: Path, width: int, height: int) -> np.ndarray:
    """Ground truth as an (N, 5) array of class, x1, y1, x2, y2 in pixels."""
    rows = []
    file = label_path(image)
    for line in filter(None, file.read_text(encoding="utf-8").split("\n") if file.exists() else []):
        class_id, cx, cy, w, h = (float(v) for v in line.split()[:5])
        rows.append([class_id, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def make_shrunk_dataset(data_yaml: Path, factor: float) -> Path:
    """Write the validation split scaled down by `factor` on a grey canvas, with adjusted labels. Returns its yaml."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    source_root = Path(data["path"])
    out_root = source_root.parent / f"{source_root.name}_val_x{factor:.2f}"
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "val").mkdir(parents=True)
    (out_root / "labels" / "val").mkdir(parents=True)

    for image_path in sorted((source_root / data["val"]).glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        small_w, small_h = round(width * factor), round(height * factor)
        rng = random.Random(f"{image_path.name}:{factor}")
        x0, y0 = rng.randint(0, width - small_w), rng.randint(0, height - small_h)
        canvas = np.full_like(image, GREY)
        canvas[y0:y0 + small_h, x0:x0 + small_w] = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_root / "images" / "val" / image_path.name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        lines = []
        for class_id, x1, y1, x2, y2 in read_labels(image_path, width, height):
            x1, x2 = (x0 + x1 * factor) / width, (x0 + x2 * factor) / width
            y1, y2 = (y0 + y1 * factor) / height, (y0 + y2 * factor) / height
            lines.append(f"{int(class_id)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        (out_root / "labels" / "val" / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    out_yaml = out_root / f"{out_root.name}.yaml"
    out_yaml.write_text(
        yaml.safe_dump(
            {"path": out_root.as_posix(), "train": "images/val", "val": "images/val", "names": data["names"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return out_yaml


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_image(gt: np.ndarray, pred: np.ndarray, iou_threshold: float, matrix: np.ndarray, names: dict) -> list[str]:
    """Greedy matching by confidence. Updates matrix[predicted, true] (last index = background), returns problems.

    gt: (N, 5) class, x1, y1, x2, y2.  pred: (M, 6) x1, y1, x2, y2, conf, class.
    """
    background = len(names)
    problems = []
    iou = box_iou(pred[:, :4], gt[:, 1:]) if len(gt) and len(pred) else np.zeros((len(pred), len(gt)))
    matched = np.zeros(len(gt), dtype=bool)
    for i in np.argsort(-pred[:, 4]):
        p_cls, conf = int(pred[i, 5]), pred[i, 4]
        candidates = np.where(~matched & (iou[i] >= iou_threshold))[0]
        if len(candidates):
            j = candidates[np.argmax(iou[i, candidates])]
            matched[j] = True
            g_cls = int(gt[j, 0])
            matrix[p_cls, g_cls] += 1
            if p_cls != g_cls:
                problems.append(f"wrong class: {names[g_cls]} detected as {names[p_cls]} ({conf:.2f})")
            continue
        matrix[p_cls, background] += 1
        overlapped = np.where(iou[i] >= iou_threshold)[0]
        if len(overlapped):
            problems.append(f"duplicate: extra {names[p_cls]} ({conf:.2f}) on {names[int(gt[overlapped[0], 0])]}")
        else:
            problems.append(f"false positive: {names[p_cls]} ({conf:.2f})")
    for j in np.where(~matched)[0]:
        matrix[background, int(gt[j, 0])] += 1
        problems.append(f"missed: {names[int(gt[j, 0])]}")
    return problems


def draw_failure(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, names: dict) -> np.ndarray:
    thickness = max(2, round(image.shape[1] / 400))
    font_scale = image.shape[1] / 1200
    for class_id, x1, y1, x2, y2 in gt.astype(int):
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness * 2)
    for x1, y1, x2, y2, conf, class_id in pred:
        colour = COLOURS[int(class_id) % len(COLOURS)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
        text = f"{names[int(class_id)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 + th + 2 * thickness), colour, -1)
        cv2.putText(image, text, (x1, y1 + th + thickness), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return image


def evaluate_split(model, data_yaml: Path, imgsz: int, conf: float, iou: float, device: str, out_dir: Path) -> dict:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in data["names"].items()}
    nc = len(names)

    metrics = model.val(
        data=str(data_yaml), imgsz=imgsz, batch=16, device=device, plots=False, verbose=False,
        project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
    )
    classes = {name: {"instances": 0} for name in names.values()}
    for k, class_index in enumerate(metrics.box.ap_class_index):
        _, _, ap50, ap = metrics.box.class_result(k)
        classes[names[int(class_index)]].update({"AP50": float(ap50), "AP50-95": float(ap)})

    matrix = np.zeros((nc + 1, nc + 1), dtype=int)
    failures = []
    failures_dir = out_dir / "failures"
    if failures_dir.exists():
        shutil.rmtree(failures_dir)
    images = sorted((Path(data["path"]) / data["val"]).glob("*.jpg"))
    for result in model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=conf, device=device, stream=True, verbose=False):
        image_path = Path(result.path)
        height, width = result.orig_shape
        gt = read_labels(image_path, width, height)
        boxes = result.boxes
        pred = np.concatenate(
            [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None], boxes.cls.cpu().numpy()[:, None]], axis=1
        ).astype(np.float64)
        problems = match_image(gt, pred, iou, matrix, names)
        if problems:
            failures.append({"image": image_path.name, "problems": problems})
            failures_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(failures_dir / image_path.name), draw_failure(result.orig_img.copy(), gt, pred, names))

    for c, name in names.items():
        predicted, actual = matrix[c].sum(), matrix[:, c].sum()
        classes[name].update(
            {
                "instances": int(actual),
                f"precision@{conf:g}": float(matrix[c, c] / predicted) if predicted else 0.0,
                f"recall@{conf:g}": float(matrix[c, c] / actual) if actual else 0.0,
            }
        )
    errors = {
        f"{names[t]}->{names[p]}": int(matrix[p, t]) for t in names for p in names if p != t
    } | {"missed": int(matrix[nc, :nc].sum()), "false_positive_or_duplicate": int(matrix[:nc, nc].sum())}
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "classes": classes,
        "confusion_matrix": {"labels": [*names.values(), "background"], "rows_predicted_cols_true": matrix.tolist()},
        "errors": errors,
        "failures": failures,
    }


def print_split(label: str, result: dict, conf: float, failures_dir: Path):
    print(f"\n== {label}: mAP50 {result['mAP50']:.3f}  mAP50-95 {result['mAP50-95']:.3f}")
    print(f"  {'class':12} {'AP50':>6} {'AP50-95':>8} {f'P@{conf:g}':>7} {f'R@{conf:g}':>7} {'cats':>5}")
    for name, c in result["classes"].items():
        print(
            f"  {name:12} {c.get('AP50', 0):6.3f} {c.get('AP50-95', 0):8.3f} "
            f"{c[f'precision@{conf:g}']:7.3f} {c[f'recall@{conf:g}']:7.3f} {c['instances']:5d}"
        )
    labels = result["confusion_matrix"]["labels"]
    print("  confusion matrix (rows: predicted, columns: true)")
    print("  " + " " * 12 + "".join(f"{name:>12}" for name in labels))
    for name, row in zip(labels, result["confusion_matrix"]["rows_predicted_cols_true"]):
        print(f"  {name:12}" + "".join(f"{v:12d}" for v in row))
    print("  errors: " + ", ".join(f"{k} {v}" for k, v in result["errors"].items()))
    if result["failures"]:
        print(f"  {len(result['failures'])} images with problems -> {failures_dir} (all listed in summary.json)")
        for failure in result["failures"][:MAX_PRINTED_FAILURES]:
            problems = failure["problems"]
            more = f"; +{len(problems) - 3} more" if len(problems) > 3 else ""
            print(f"    {failure['image']}: {'; '.join(problems[:3])}{more}")
        if len(result["failures"]) > MAX_PRINTED_FAILURES:
            print(f"    ... +{len(result['failures']) - MAX_PRINTED_FAILURES} more images")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "weights", help="run name in runs/train/ (uses weights/best.pt), a weights file or an exported model folder"
    )
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "cats" / "cats.yaml")
    parser.add_argument("--imgsz", type=int, help="inference size (default: the run's training imgsz, else 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for P/R, confusions, failures")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to match a prediction to a cat")
    parser.add_argument("--shrink", type=float, nargs="+", default=[1.0], help="image scale factors to evaluate")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", help="output folder in runs/eval/ (default: <run>_imgsz<imgsz>)")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help stays fast

    weights, run_name, train_imgsz = resolve_weights(args.weights)
    imgsz = args.imgsz or train_imgsz or 640
    # An exported model folder (best_ncnn_model/) gets its format in the default name, next to the .pt evaluation.
    export_format = weights.name.removesuffix("_model").rsplit("_", 1)[-1] if weights.is_dir() else None
    out_dir = EVAL_RUNS_DIR / (args.name or f"{run_name}_imgsz{imgsz}" + (f"_{export_format}" if export_format else ""))
    model = YOLO(weights)
    data_yaml = args.data.resolve()

    summary = {"weights": str(weights), "imgsz": imgsz, "conf": args.conf, "iou": args.iou, "results": {}}
    for factor in args.shrink:
        split_yaml = data_yaml if factor == 1 else make_shrunk_dataset(data_yaml, factor)
        split_dir = out_dir / f"x{factor:.2f}"
        summary["results"][f"x{factor:.2f}"] = evaluate_split(model, split_yaml, imgsz, args.conf, args.iou, args.device, split_dir)

    print(f"\nweights {weights}  imgsz {imgsz}  conf {args.conf}  IoU {args.iou}")
    for label, result in summary["results"].items():
        print_split(f"val {label}", result, args.conf, out_dir / label / "failures")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
