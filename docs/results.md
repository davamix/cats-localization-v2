# Results

Metrics and benchmarks for the cats detector. Newest phase at the top of each section.

## Phase 2 — training on the PC (2026-10-02)

> **Read these numbers with care.** The validation set (43 video frames) is leaky: many frames are near-duplicates
> of training photos taken in the same session (see the phase 1 handover notes). Scores on the normal validation
> set (×1) are optimistic and almost every run looks perfect there. The trustworthy test set will be Pi-camera
> images (phases 4–6). With 43 images, one image is worth ~2.3 percentage points of recall.

### Setup

- Model: YOLO26n fine-tuned from the COCO-pretrained `yolo26n.pt` (Ultralytics 8.4.171, torch 2.14.1+cu126,
  RTX 2080 Ti).
- Data: 98 training photos, 43 validation frames, classes Blacky and Niche, one cat per image.
- Training (all runs): 100 epochs, early-stopping patience 30, batch 16, deterministic, `optimizer=auto` (resolves
  to AdamW, lr 0.001667, momentum 0.9, weight decay 5e-4), 3 warmup epochs, linear decay to `lrf` 0.01, AMP.
  Default augmentation: mosaic 1.0 (off for the last 10 epochs), horizontal flip 0.5, HSV 0.015/0.7/0.4,
  translate 0.1. `best.pt` is chosen by validation mAP50-95 (so it is optimistic too).
- Evaluation with [train/evaluate.py](../train/evaluate.py):
  - mAP50 and mAP50-95 from Ultralytics validation (confidence 0.001).
  - Precision, recall, the confusion matrix and the error counts at a fixed confidence (0.25 unless stated) and
    IoU 0.5, matching each prediction to the cat it overlaps most regardless of class.
  - **Shrunken validation (×0.5, ×0.25)**: the same frames scaled down and pasted at a random position on a grey
    canvas of the original size, as a rough proxy for a cat far from the camera. At ×0.25 the cats are 4.5–23% of
    the frame width (the training photos only have cats ≥ 24% wide). Grey borders are artificial, so this is a
    proxy, not a substitute for real small-cat images.

### Runs

| Run | Train imgsz | `scale` | Epochs (best) | Training time |
|---|---|---|---|---|
| `yolo26n_640` | 640 | 0.5 | 99, early stop (69) | 3.4 min |
| `yolo26n_416` | 416 | 0.5 | 100 (89) | 3.4 min |
| `yolo26n_320` | 320 | 0.5 | 100 (90) | 3.2 min |
| `yolo26n_320_scale0.9` | 320 | **0.9** | 100 (93) | 3.3 min |

`scale` is the zoom range of the scale augmentation, [1 − s, 1 + s]. With the default 0.5 the smallest cat the
model sees during training is ~12% of the frame width; with 0.9 it is ~2.4%.

### Normal validation (×1), confidence 0.25

| Run | Eval imgsz | mAP50 | mAP50-95 | Blacky→Niche | Niche→Blacky | Missed | False pos. / duplicates |
|---|---|---|---|---|---|---|---|
| `yolo26n_640` | 640 | 0.995 | 0.969 | 0 | 0 | 0 | 3 |
| `yolo26n_640` | 320 | 0.973 | 0.916 | 0 | 0 | 1 | 1 |
| `yolo26n_416` | 416 | 0.995 | 0.967 | 0 | 0 | 0 | 1 |
| `yolo26n_320` | 320 | 0.995 | 0.943 | 0 | 0 | 0 | 2 |
| `yolo26n_320_scale0.9` | 320 | 0.995 | 0.944 | 0 | 0 | 0 | 6 |

### Small-cat proxy (shrunken validation), confidence 0.25

| Run | Eval imgsz | ×0.5 mAP50 | ×0.5 missed / FP | ×0.25 mAP50 | ×0.25 mAP50-95 | ×0.25 recall Blacky / Niche | ×0.25 missed / FP |
|---|---|---|---|---|---|---|---|
| `yolo26n_640` | 640 | 0.992 | 2 / 2 | 0.771 | 0.493 | 0.52 / 0.50 | 21 / 0 |
| `yolo26n_640` | 320 | 0.843 | 13 / 4 | 0.048 | 0.015 | 0.00 / 0.00 | 43 / 0 |
| `yolo26n_416` | 416 | 0.995 | 0 / 1 | 0.542 | 0.294 | 0.33 / 0.18 | 32 / 0 |
| `yolo26n_320` | 320 | 0.973 | 2 / 1 | 0.649 | 0.366 | 0.43 / 0.05 | 33 / 1 |
| `yolo26n_320_scale0.9` | 320 | 0.963 | 2 / 8 | **0.960** | **0.762** | **1.00 / 1.00** | **0 / 18** |

No run confused Blacky with Niche on a matched box, at any scale.

### Confidence threshold (320 models)

Errors as missed / false positives (including duplicates), 43 cats per column.

| Run | Confidence | ×1 | ×0.5 | ×0.25 |
|---|---|---|---|---|
| `yolo26n_320` | 0.25 | 0 / 2 | 2 / 1 | 33 / 1 |
| `yolo26n_320` | 0.5 | 0 / 1 | 3 / 1 | 37 / 0 |
| `yolo26n_320_scale0.9` | 0.25 | 0 / 6 | 2 / 8 | 0 / 18 |
| `yolo26n_320_scale0.9` | 0.4 | 0 / 4 | 2 / 6 | 0 / 9 |
| `yolo26n_320_scale0.9` | 0.5 | 0 / 4 | 2 / 5 | 1 / 6 |

### Seed repeats (320 models), confidence 0.5

The single-seed results above differ by a few images, so both 320 configs were trained again with seeds 1 and 2
(runs `yolo26n_320_seed<N>`, `yolo26n_320_scale0.9_seed<N>`). Errors as missed / false positives (incl. duplicates).

| Config | Seed | ×1 mAP50-95 | ×1 | ×0.5 | ×0.25 mAP50 | ×0.25 |
|---|---|---|---|---|---|---|
| `scale` 0.5 | 0 | 0.943 | 0 / 1 | 3 / 1 | 0.649 | 37 / 0 |
| `scale` 0.5 | 1 | 0.931 | 0 / 5 | 0 / 1 | 0.779 | 18 / 2 |
| `scale` 0.5 | 2 | 0.945 | 1 / 1 | 2 / 1 | 0.597 | 33 / 2 |
| `scale` 0.9 | 0 | 0.944 | 0 / 4 | 2 / 5 | 0.960 | 1 / 6 |
| `scale` 0.9 | 1 | 0.930 | 1 / 3 | 1 / 0 | 0.886 | 6 / 7 |
| `scale` 0.9 | 2 | 0.929 | 1 / 3 | 1 / 4 | 0.970 | 2 / 3 |

- On normal validation the two configs are equivalent; the seed-to-seed spread (e.g. 1 vs 5 false positives) is
  as large as the difference between configs.
- On small cats (×0.25) `scale` 0.9 wins on every seed: 1–6 of 43 cats missed instead of 18–37.
- The only Blacky ↔ Niche confusion in all evaluations: `yolo26n_320_seed1` at ×0.25, confidence 0.25, a dim
  quarter-size Blacky labelled Niche (0.27, loose box, `blacky_frame100`). It disappears at confidence 0.5.

### Failure cases

Looked at the failure images written by `evaluate.py` (`runs/eval/<run>/x<factor>/failures/`):

- **Dark objects detected as Blacky** — the most common error. A black cloth, a dark bag, chair legs or a dark
  cushion at the edge of the frame get a "Blacky" box (`blacky_frame340/360/920/1120`, `niche_frame140`; up to
  0.91 confidence at ×0.25). The baseline 320 model already does this on `blacky_frame340/360`; the scale-0.9
  model does it more often. The dataset has no images without cats, so nothing teaches the model that a dark
  blob is not Blacky. Fix in phase 6: background images (empty rooms, dark objects) from the Pi camera.
- **Part of Niche detected as a second cat.** In dark or cut-off frames the model sometimes adds a box on part of
  Niche. In `niche_frame40` (×0.5 and ×0.25) **Niche's dark patch gets a "Blacky" box** (0.56–0.76) next to the
  correct Niche box. This is an identity error in practice, but the evaluation counts it as a false positive
  (plus a miss when the Niche box is lost), because the partial box does not overlap the whole cat enough.
- **Loose boxes on small cats** (`blacky_frame160` ×0.5): the right cat with the right name, but the box is too
  large (IoU < 0.5), counted as a miss plus a false positive.
- **Missed small cats** (all baseline runs at ×0.25): the cat is found with low confidence or not at all. The
  640-trained model run at 320 misses every cat at ×0.25 — training at the deployment size matters.

### Recommendation

**Deployment candidate: `yolo26n_320_scale0.9` (seed 0) — `runs/train/yolo26n_320_scale0.9/weights/best.pt`,
input size 320, confidence threshold 0.5 as the starting point.**

- **Why `scale` 0.9:** it is the only configuration that still finds small cats (×0.25 mAP50 0.89–0.97 across
  three seeds, vs 0.60–0.78 for the default), and it costs nothing measurable on normal validation. The price is
  a few more false positives, mostly dark objects labelled Blacky. Raising the confidence from 0.25 to 0.5 cuts
  them by half to two thirds at ×0.25 (seed 0: 18 → 6) while still finding 42/43 small cats.
- **Why 320:** every input size scores the same on the (leaky) normal validation, and the larger sizes do not
  help with small cats unless trained for it. YOLO26n is 5.3 GFLOPs at 640, ~2.2 at 416 and ~1.3 at 320, and the
  Pi 3B is a 4× Cortex-A53, so 320 is the size most likely to give a usable frame rate. Training at the deployment
  size matters: the 640-trained model run at 320 misses every cat at ×0.25.
- **Why seed 0:** the three seeds are equivalent within the noise of this validation set; picking the "best" one
  by these numbers would only fit the leaky validation set, so the planned run is kept.
- **To revisit in phase 4:** if the Pi turns out fast enough at 416, train a 416 + `scale` 0.9 run (~3.5 min) and
  compare on Pi-camera images. Tune the confidence threshold on real frames.
- **Known weaknesses** (for phases 4–6): dark objects detected as Blacky (no background images in the dataset);
  extra boxes on parts of Niche, sometimes labelled Blacky; closed set (an unknown cat will be called Blacky or
  Niche); all numbers come from a leaky validation set and a synthetic small-cat proxy.
