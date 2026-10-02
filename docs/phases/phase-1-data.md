# Phase 1 — Dataset conversion (VIA → YOLO)

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 0 |

## Goal

Turn the 2020 VIA polygon annotations into an Ultralytics YOLO detection dataset, and check visually that the
boxes are right.

## Input

- `data/train/cats-annotations.json`, `data/validation/cats-annotations.json` — VIA 2.0.8 project exports.
  - The top-level key `_via_attributes` holds the attribute definitions; every other key is an image entry with
    `filename` and `regions`.
  - Each region has `shape_attributes` (`name: "polygon"`, `all_points_x`, `all_points_y`) and
    `region_attributes.Class` (`"Blacky"` or `"Niche"`).
- Images live in `data/<split>/<Class>/<filename>`. All are 1920×1080.

## Steps

- [ ] `tools/via_to_yolo.py`
  - Read both JSON files, skip `_via_attributes`.
  - Class map in one place (`Blacky: 0`, `Niche: 1`) so new classes can be added later (phase 8).
  - Polygon → bounding box (min/max of the points, clipped to the image), written as normalised YOLO
    `class cx cy w h` lines.
  - Copy images to `datasets/cats/images/{train,val}` and write labels to `datasets/cats/labels/{train,val}`.
    **Prefix file names with the class folder** (e.g. `blacky_frame220.jpg`): validation frame names collide
    between `Blacky/` and `Niche/` (both have `frame220.jpg`, etc.).
  - Write `datasets/cats/cats.yaml` (`path`, `train`, `val`, `names`).
  - Optional flag to also write polygon (segmentation) labels, for later use.
- [ ] `tools/visualize_labels.py`: draw the converted boxes and class names on the images into
      `datasets/cats/preview/` for a visual check.
- [ ] Sanity checks: 98 train / 43 val images, one box per image, all coordinates in `[0, 1]`, class counts
      51/47 and 21/22.
- [ ] Confirm Ultralytics accepts the dataset (`ultralytics.data.utils.check_det_dataset("datasets/cats/cats.yaml")`).

## Done when

- `python tools/via_to_yolo.py` rebuilds `datasets/cats/` from scratch, deterministically.
- The preview images show tight boxes with the right names.
- Ultralytics loads `cats.yaml` without errors.

## Handover notes

_None yet._
