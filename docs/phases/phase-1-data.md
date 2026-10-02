# Phase 1 — Dataset conversion (VIA → YOLO)

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 0 |

## Goal

Turn the 2020 VIA polygon annotations into an Ultralytics YOLO detection dataset, and check visually that the
boxes are right.

## Input

- `data/train/cats-annotations.json`, `data/validation/cats-annotations.json` — VIA 2.0.8 project exports.
  - The top-level key `_via_attributes` holds the attribute definitions; every other key is an image entry with
    `filename`, `size` (file size in bytes) and `regions`.
  - Each region has `shape_attributes` (`name: "polygon"`, `all_points_x`, `all_points_y`) and
    `region_attributes.Class` (`"Blacky"` or `"Niche"`).
- Images live in `data/<split>/<Class>/<filename>`. All are 1920×1080 RGB with no EXIF orientation tag.

## Steps

- [x] [tools/via_to_yolo.py](../../tools/via_to_yolo.py)
  - Reads both JSON files, skips `_via_attributes`.
  - Class map in one place (`CLASSES = ["Blacky", "Niche"]`); append new classes at the end (phase 8).
  - Finds each image by file name **and** file size (that is how VIA identifies images), so the colliding
    validation names (`Blacky/frame220.jpg` vs `Niche/frame220.jpg`) resolve correctly.
  - Polygon → bounding box (min/max of the points, clipped to the image) as normalised `class cx cy w h`.
  - Copies images to `datasets/cats/images/{train,val}` with the class folder as prefix
    (`blacky_frame220.jpg`) and writes labels to `datasets/cats/labels/{train,val}`.
  - Writes `datasets/cats/cats.yaml` with an **absolute** `path` (Ultralytics resolves relative dataset paths
    against its global `datasets_dir` setting when they are not found from the working directory).
  - `--task segment` writes polygon labels instead (tested, works).
  - Rebuilds the output folder from scratch on every run; images without regions would get an empty label file
    (background image).
- [x] [tools/visualize_labels.py](../../tools/visualize_labels.py): draws boxes (or polygons) and class names into
      `datasets/cats/preview/<split>/` and contact sheets `datasets/cats/preview/<split>_sheet_NN.jpg`.
- [x] Sanity checks (see Results).
- [x] Ultralytics accepts the dataset (`check_det_dataset` and `YOLODataset` label scan).

## How to run

```powershell
.\.venv\Scripts\python.exe tools\via_to_yolo.py          # -> datasets/cats/ + cats.yaml
.\.venv\Scripts\python.exe tools\visualize_labels.py     # -> datasets/cats/preview/
```

## Done when

- [x] `python tools/via_to_yolo.py` rebuilds `datasets/cats/` from scratch, deterministically.
- [x] The preview images show tight boxes with the right names.
- [x] Ultralytics loads `cats.yaml` without errors.

## Results

| Split | Images | Boxes | Blacky | Niche | Box width (fraction of image) | Box height |
|---|---|---|---|---|---|---|
| train | 98 | 98 | 51 | 47 | 0.24 – 0.99 | 0.34 – 1.00 |
| val | 43 | 43 | 21 | 22 | 0.18 – 0.93 | 0.34 – 1.00 |

- Ultralytics label scan: 0 corrupt, 0 backgrounds in both splits.
- Re-running the converter produces byte-identical labels.
- All 141 previews were checked visually: boxes are tight and every class is correct.

## Handover notes

- **The cats are easy to tell apart:** Blacky is completely black, Niche is white with large dark patches. Identity
  confusion should be rare; the harder parts will be detection in poor light and small/partial cats.
- **Validation is leaky:** many validation frames are near-duplicates of training photos taken in the same session
  (e.g. `blacky_frame400` ≈ `b_22`, `blacky_frame200–240` ≈ `b_15–19`, `niche_frame340` ≈ `n_37`,
  `niche_frame480` ≈ `n_45`). Expect very high validation scores that overstate real performance. The trustworthy
  test set will be Pi-camera images (phase 6).
- **Cats are always large in the frame** (smallest box is 18% of the image width). A fixed Pi camera across a room
  will often see them much smaller, which the training data does not cover. Phase 2 should try stronger scale
  augmentation; phase 6 fixes it with real data.
- Many Blacky photos are dark/underexposed; a black cat in a dim room is the hardest case.
- No image contains both cats, and there are no empty scenes (phase 6).
- Ultralytics writes `labels/train.cache` and `labels/val.cache` next to the labels; they are rebuilt automatically
  when the labels change. It also downloaded `Arial.ttf` to `%APPDATA%\Ultralytics` (used for its plots).
- **Next:** phase 2 (training).
