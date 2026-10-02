"""Compare two detection files written by `pi/detector.py --json`, e.g. the same images run on the PC and on the Pi.

In each image, every box of the first file is paired with the unpaired box of the same class in the second file that
overlaps it most. The script prints, per image, the box counts, the smallest IoU of the pairs, the largest score
difference and the largest corner difference in pixels. It fails (exit code 1) if the files do not have the same
images, an image has a different number of boxes of some class, or a pair has IoU < --min-iou or a score difference
> --max-score-diff. The JSON files round coordinates and scores to 3 decimals, so 0.001 is the finest difference seen.

    python tools/compare_detections.py runs/pi/compare_pc.json runs/pi/compare_pi.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N, 4) and (M, 4) x1, y1, x2, y2 boxes."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def pair_boxes(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Greedy pairing of (N, 6) and (M, 6) detections of the same class, highest IoU first -> index pairs."""
    if not len(a) or not len(b):
        return []
    iou = box_iou(a[:, :4], b[:, :4])
    iou[a[:, None, 5] != b[None, :, 5]] = -1
    pairs = []
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < 0:
            break
        if all(i != p and j != q for p, q in pairs):
            pairs.append((int(i), int(j)))
    return pairs


def class_counts(detections: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(detections[:, 5].astype(int), return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="reference detections (e.g. PC)")
    parser.add_argument("b", type=Path, help="detections to check (e.g. Pi)")
    parser.add_argument("--min-iou", type=float, default=0.99)
    parser.add_argument("--max-score-diff", type=float, default=0.01)
    args = parser.parse_args()

    a_all = json.loads(args.a.read_text(encoding="utf-8"))
    b_all = json.loads(args.b.read_text(encoding="utf-8"))
    problems = [f"{name}: only in {args.a.name}" for name in sorted(a_all.keys() - b_all.keys())]
    problems += [f"{name}: only in {args.b.name}" for name in sorted(b_all.keys() - a_all.keys())]

    print(f"{'image':32} {'boxes A/B':>9} {'min IoU':>8} {'max dscore':>10} {'max dpx':>8}")
    total_a = total_b = total_pairs = 0
    worst_iou, worst_score, worst_px = 1.0, 0.0, 0.0
    for name in sorted(a_all.keys() & b_all.keys()):
        a = np.array(a_all[name], dtype=np.float64).reshape(-1, 6)
        b = np.array(b_all[name], dtype=np.float64).reshape(-1, 6)
        pairs = pair_boxes(a, b)
        total_a, total_b, total_pairs = total_a + len(a), total_b + len(b), total_pairs + len(pairs)
        if class_counts(a) != class_counts(b) or len(pairs) != len(a):
            problems.append(f"{name}: boxes per class {class_counts(a)} vs {class_counts(b)}")
        min_iou = max_score = max_px = float("nan")
        if pairs:
            i, j = map(list, zip(*pairs))
            min_iou = float(box_iou(a[i, :4], b[j, :4]).diagonal().min())
            max_score = float(np.abs(a[i, 4] - b[j, 4]).max())
            max_px = float(np.abs(a[i, :4] - b[j, :4]).max())
            worst_iou, worst_score, worst_px = min(worst_iou, min_iou), max(worst_score, max_score), max(worst_px, max_px)
            if min_iou < args.min_iou:
                problems.append(f"{name}: IoU {min_iou:.4f} < {args.min_iou}")
            if max_score > args.max_score_diff:
                problems.append(f"{name}: score difference {max_score:.4f} > {args.max_score_diff}")
        print(f"{name:32} {f'{len(a)}/{len(b)}':>9} {min_iou:8.4f} {max_score:10.4f} {max_px:8.3f}")

    print(f"\n{len(a_all.keys() & b_all.keys())} images, boxes {total_a} / {total_b}, {total_pairs} paired; "
          f"min IoU {worst_iou:.4f}, max score diff {worst_score:.4f}, max corner diff {worst_px:.3f} px")
    if problems:
        print("\nFAIL:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"OK: same boxes and classes (IoU >= {args.min_iou}, score diff <= {args.max_score_diff})")


if __name__ == "__main__":
    main()
