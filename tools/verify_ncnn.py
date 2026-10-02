"""Check on the PC that pi/detector.py (ncnn + numpy) gives the same detections as the Ultralytics PyTorch model.

For each validation split (the normal one and, with --shrink, the shrunken copies made by train/evaluate.py):
  1. Raw output: the detector's preprocessed input goes through both PyTorch and ncnn; prints the largest
     difference in box values (input pixels) and scores. This checks the conversion alone.
  2. Detections at --conf: Ultralytics predict (square letterbox like the detector, rect=False) against
     Detector.detect, pairing boxes of the same class with IoU >= 0.5. A difference is a box without a partner,
     a pair with IoU < --iou-tol or a pair whose scores differ by more than --score-tol. A box without a partner
     whose score is within --score-tol of --conf is "borderline" (the other side is just below the threshold)
     and does not fail the check.
  3. mAP50 and mAP50-95 of both pipelines (confidence 0.001), computed with the same code. Fails if they differ
     by more than --map-tol.
It also prints the detector's time per stage on this PC. Exit code 1 if any check fails.

The validation set is leaky (near-duplicates of training photos): these numbers show that NCNN matches PyTorch,
not how well the model works on the Pi camera.

    python tools/verify_ncnn.py yolo26n_320_scale0.9            # uses weights/best.pt and weights/best_ncnn_model/
    python tools/verify_ncnn.py yolo26n_320_scale0.9 --shrink 1 0.5 0.25
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "pi"), str(REPO_ROOT / "train")]
from detector import Detector  # noqa: E402
from evaluate import box_iou, make_shrunk_dataset, read_labels, resolve_weights  # noqa: E402

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # mAP50-95
MAP_CONF = 0.001  # confidence for mAP, as in Ultralytics validation


def pytorch_detections(results) -> np.ndarray:
    boxes = results.boxes
    return np.concatenate(
        [boxes.xyxy.numpy(), boxes.conf.numpy()[:, None], boxes.cls.numpy()[:, None]], axis=1
    ).astype(np.float64)


def compare_detections(reference: np.ndarray, ncnn: np.ndarray, conf: float, iou_tol: float, score_tol: float
                       ) -> dict:
    """Pair same-class boxes (IoU >= 0.5, best score first) and count the differences."""
    iou = np.zeros((len(reference), len(ncnn)))
    if len(reference) and len(ncnn):
        iou = box_iou(reference[:, :4], ncnn[:, :4]) * (reference[:, 5][:, None] == ncnn[:, 5][None, :])
    paired = np.zeros(len(ncnn), dtype=bool)
    result = {"pairs": [], "different": [], "borderline": []}
    for i in np.argsort(-reference[:, 4]):
        candidates = np.where(~paired & (iou[i] >= 0.5))[0]
        if not len(candidates):
            unpaired = result["borderline" if reference[i, 4] <= conf + score_tol else "different"]
            unpaired.append(f"only PyTorch: class {int(reference[i, 5])} {reference[i, 4]:.3f}")
            continue
        j = candidates[np.argmax(iou[i, candidates])]
        paired[j] = True
        pair_iou, score_diff = iou[i, j], abs(reference[i, 4] - ncnn[j, 4])
        result["pairs"].append((pair_iou, score_diff))
        if pair_iou < iou_tol or score_diff > score_tol:
            result["different"].append(
                f"class {int(reference[i, 5])}: IoU {pair_iou:.3f}, score {reference[i, 4]:.3f} vs {ncnn[j, 4]:.3f}"
            )
    for j in np.where(~paired)[0]:
        unpaired = result["borderline" if ncnn[j, 4] <= conf + score_tol else "different"]
        unpaired.append(f"only NCNN: class {int(ncnn[j, 5])} {ncnn[j, 4]:.3f}")
    return result


def correct_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(M, 10) true positives of each prediction at IoU 0.5..0.95, with Ultralytics' validation matching."""
    correct = np.zeros((len(pred), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(gt) or not len(pred):
        return correct
    iou = box_iou(gt[:, 1:], pred[:, :4]) * (gt[:, 0][:, None] == pred[:, 5][None, :])
    for k, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if len(matches) > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), k] = True
    return correct


def mean_ap(stats: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """(mAP50, mAP50-95) from per-image (correct, predictions, ground truth)."""
    from ultralytics.utils.metrics import ap_per_class

    correct = np.concatenate([s[0] for s in stats])
    pred = np.concatenate([s[1] for s in stats])
    gt = np.concatenate([s[2] for s in stats])
    ap = ap_per_class(correct, pred[:, 4], pred[:, 5], gt[:, 0])[5]  # (classes, 10)
    return float(ap[:, 0].mean()), float(ap.mean())


def verify_split(images: list[Path], weights: Path, torch_model, detector: Detector, map_detector: Detector,
                 imgsz: int, args) -> bool:
    from ultralytics import YOLO

    yolo = YOLO(weights)
    max_box_diff = max_score_diff = 0.0
    counts = {"pytorch": 0, "ncnn": 0}
    pairs, different, borderline = [], [], []
    map_stats = {"pytorch": [], "ncnn": []}
    times = {"preprocess": [], "infer": [], "postprocess": []}

    for image_path in images:
        frame = cv2.imread(str(image_path))
        height, width = frame.shape[:2]

        # 1. raw output for the same input
        t0 = time.perf_counter()
        mat, scale, pad = detector.preprocess(frame)
        t1 = time.perf_counter()
        ncnn_raw = detector.infer(mat)
        t2 = time.perf_counter()
        ncnn_dets = detector.postprocess(ncnn_raw, scale, pad, (height, width)).astype(np.float64)
        t3 = time.perf_counter()
        for stage, seconds in zip(times, (t1 - t0, t2 - t1, t3 - t2)):
            times[stage].append(seconds * 1000)
        with torch.no_grad():
            torch_raw = torch_model(torch.from_numpy(np.array(mat))[None])[0][0].numpy()
        max_box_diff = max(max_box_diff, float(np.abs(torch_raw[:4] - ncnn_raw[:4]).max()))
        max_score_diff = max(max_score_diff, float(np.abs(torch_raw[4:] - ncnn_raw[4:]).max()))

        # 2. detections at the deployment threshold
        torch_dets = pytorch_detections(
            yolo.predict(str(image_path), imgsz=imgsz, conf=args.conf, rect=False, device="cpu", verbose=False)[0]
        )
        counts["pytorch"] += len(torch_dets)
        counts["ncnn"] += len(ncnn_dets)
        result = compare_detections(torch_dets, ncnn_dets, args.conf, args.iou_tol, args.score_tol)
        pairs += result["pairs"]
        different += [f"{image_path.name}: {d}" for d in result["different"]]
        borderline += [f"{image_path.name}: {d}" for d in result["borderline"]]

        # 3. mAP inputs
        gt = read_labels(image_path, width, height)
        torch_all = pytorch_detections(
            yolo.predict(str(image_path), imgsz=imgsz, conf=MAP_CONF, rect=False, device="cpu", verbose=False)[0]
        )
        ncnn_all = map_detector.detect(frame).astype(np.float64)
        map_stats["pytorch"].append((correct_matrix(gt, torch_all), torch_all, gt))
        map_stats["ncnn"].append((correct_matrix(gt, ncnn_all), ncnn_all, gt))

    map_torch, map_ncnn = mean_ap(map_stats["pytorch"]), mean_ap(map_stats["ncnn"])
    map_diff = max(abs(a - b) for a, b in zip(map_torch, map_ncnn))
    min_iou = min((p[0] for p in pairs), default=float("nan"))
    max_pair_score_diff = max((p[1] for p in pairs), default=float("nan"))
    print(f"  raw output, same input:  max |diff| boxes {max_box_diff:.4f} px, scores {max_score_diff:.6f}")
    print(f"  detections @{args.conf:g}:  PyTorch {counts['pytorch']}, NCNN {counts['ncnn']}, paired {len(pairs)} "
          f"(min IoU {min_iou:.4f}, max |score diff| {max_pair_score_diff:.4f}), "
          f"different {len(different)}, borderline {len(borderline)}")
    for line in different + [f"{b} (borderline)" for b in borderline]:
        print(f"    {line}")
    print(f"  mAP50     PyTorch {map_torch[0]:.4f}  NCNN {map_ncnn[0]:.4f}")
    print(f"  mAP50-95  PyTorch {map_torch[1]:.4f}  NCNN {map_ncnn[1]:.4f}   (max diff {map_diff:.4f})")
    print(f"  detector on this PC ({args.threads} threads): "
          + ", ".join(f"{stage} {np.median(ms):.1f} ms" for stage, ms in times.items()) + " (median)")
    return not different and map_diff <= args.map_tol


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", help="run name in runs/train/ (uses weights/best.pt) or a weights file")
    parser.add_argument("--model-dir", type=Path, help="NCNN model folder (default: <weights>_ncnn_model next to it)")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "cats" / "cats.yaml")
    parser.add_argument("--shrink", type=float, nargs="+", default=[1.0], help="image scale factors to check")
    parser.add_argument("--conf", type=float, default=0.5, help="detection threshold for the box comparison")
    parser.add_argument("--iou-tol", type=float, default=0.9, help="minimum IoU between paired boxes")
    parser.add_argument("--score-tol", type=float, default=0.05, help="maximum score difference between paired boxes")
    parser.add_argument("--map-tol", type=float, default=0.01, help="maximum mAP difference")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    from ultralytics import YOLO

    weights, _, _ = resolve_weights(args.weights)
    model_dir = args.model_dir or weights.with_name(f"{weights.stem}_ncnn_model")
    detector = Detector(model_dir, conf=args.conf, threads=args.threads)
    map_detector = Detector(model_dir, conf=MAP_CONF, threads=args.threads)
    imgsz = detector.input_width
    if detector.input_height != imgsz:
        parser.error("rectangular models are not supported here: Ultralytics predict takes one imgsz")

    torch_model = YOLO(weights).model.float().eval()
    torch_model.end2end = False  # the one-to-many head, as exported to NCNN and used by Ultralytics by default
    print(f"weights {weights}\nNCNN    {model_dir}  (imgsz {imgsz}, FP32)")

    data_yaml = args.data.resolve()
    passed = True
    for factor in args.shrink:
        split_yaml = data_yaml if factor == 1 else make_shrunk_dataset(data_yaml, factor)
        data = yaml.safe_load(split_yaml.read_text(encoding="utf-8"))
        images = sorted((Path(data["path"]) / data["val"]).glob("*.jpg"))
        print(f"\n== val x{factor:.2f} ({len(images)} images)")
        ok = verify_split(images, weights, torch_model, detector, map_detector, imgsz, args)
        print(f"  {'PASS' if ok else 'FAIL'}")
        passed &= ok

    print(f"\n{'PASS: NCNN matches PyTorch' if passed else 'FAIL: NCNN differs from PyTorch'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
