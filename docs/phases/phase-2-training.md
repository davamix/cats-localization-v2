# Phase 2 — Training on the PC

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 1 |

## Goal

Fine-tune YOLO26n on the cats dataset with the RTX 2080 Ti and pick a model and input size for the Pi.

## Steps

- [x] [train/train.py](../../train/train.py): thin wrapper around `ultralytics.YOLO(...).train()` with CLI arguments
      (`--model`, `--imgsz`, `--epochs`, `--patience`, `--batch`, `--scale`, `--seed`, `--name`, plus any other
      Ultralytics argument as `KEY=VALUE`). Has the `if __name__ == "__main__":` guard. Pretrained weights are
      downloaded to `models/`, runs go to `runs/train/<name>/` (both ignored by git).
- [x] Baseline runs from the COCO-pretrained `yolo26n.pt` at `imgsz` 640, 416 and 320 (100 epochs, patience 30,
      batch 16, seed 0), plus one run at 320 with stronger scale augmentation (`scale` 0.9).
- [x] Seed repeats (seeds 1 and 2) of both 320 configurations, because single runs differed by only a few images.
- [x] [train/evaluate.py](../../train/evaluate.py): mAP50, mAP50-95, per-class precision/recall, the confusion matrix
      (Blacky ↔ Niche) and failure images, on the validation set and on shrunken copies of it (small-cat proxy).
- [x] Looked at the failure cases (wrong class, missed cat, false positives, duplicate boxes).
- [x] [docs/results.md](../results.md) with the tables of runs and the chosen candidate.

## How to run

```powershell
.\.venv\Scripts\python.exe train\train.py --imgsz 320 --scale 0.9          # -> runs/train/yolo26n_320_scale0.9/
.\.venv\Scripts\python.exe train\evaluate.py yolo26n_320_scale0.9 --shrink 1 0.5 0.25 --conf 0.5
#   -> runs/eval/yolo26n_320_scale0.9_imgsz320/ (summary.json, failures/), use --name to keep several evaluations
```

`evaluate.py` takes a run name (uses its `weights/best.pt` and training `imgsz`) or a weights file, and writes the
shrunken validation copies to `datasets/cats_val_x<factor>/` (regenerated on every run).

## Done when

- [x] At least one model detects both cats on the validation set with no or rare Blacky ↔ Niche confusion.
- [x] The deployment candidate (weights + input size) is recorded in `docs/results.md`.

## Results

Full tables in [docs/results.md](../results.md). All numbers come from a **leaky validation set** (near-duplicates
of training photos), so they are optimistic.

- Every run reaches mAP50 0.97–0.995 and mAP50-95 0.93–0.97 on the normal validation set; no run confuses
  Blacky with Niche on a matched box at confidence 0.5 (one low-confidence confusion at 0.25 on a quarter-size
  dim Blacky).
- **Small cats are the real difference.** On the validation frames shrunk to ¼ (cats 4.5–23% of the frame width)
  the default-augmentation models miss most cats (320: 18–37 of 43 missed over three seeds), while `scale` 0.9
  misses 1–6 of 43 and reaches ×0.25 mAP50 0.89–0.97.
- A model trained at 640 and run at 320 misses every cat at ×0.25: train at the deployment size.
- Training takes ~3.5 minutes per run (100 epochs); all runs plateau by epochs 70–90.

**Deployment candidate:** `runs/train/yolo26n_320_scale0.9/weights/best.pt` (seed 0), input size **320**,
confidence threshold **0.5** to start with.

## Handover notes

- **Candidate weights** are only on this PC (`runs/` is ignored by git); phase 3 publishes them in the first
  GitHub Release. If they are lost, `python train/train.py --imgsz 320 --scale 0.9` retrains them in ~3.5 min
  (seeded and deterministic, but GPU training is not guaranteed to be bit-identical, so numbers may move slightly).
- **Other weights available for the phase 4 speed benchmark:** `runs/train/yolo26n_640/` and
  `runs/train/yolo26n_416/` (default augmentation). Speed does not depend on the weights, only on the input size.
  If the Pi turns out fast enough at 416, train `--imgsz 416 --scale 0.9` and compare on real frames.
- **Known failure modes** to watch on the Pi camera:
  - Dark objects (black cloth, bags, chair legs, dark cushions) detected as Blacky, up to 0.9 confidence. The
    dataset has no background images; phase 6 must add empty scenes with dark objects.
  - Extra boxes on part of Niche; in one dark frame her dark patch was labelled Blacky next to the correct Niche
    box. The evaluation counts this as a false positive, not as a confusion, so watch for it in the live stream.
  - Loose boxes on small cats.
- **Seed noise is large** with 43 validation images (e.g. 1 vs 5 false positives between seeds of the same
  config). Compare configurations over at least three seeds before drawing conclusions.
- **`evaluate.py` details:** P/R, the confusion matrix and failures are computed at `--conf` (default 0.25) with
  its own matching, because Ultralytics' built-in confusion matrix uses the validation confidence (0.001). It
  currently requires a weights *file*; to evaluate an exported NCNN model folder in phase 3, relax the
  `is_file()` check in `resolve_weights` (Ultralytics can load `*_ncnn_model/` folders).
- `train.py` points Ultralytics' AMP self-check at `models/yolo26n.pt` (by default it would download a copy to a
  relative `weights/` folder). The global Ultralytics settings on this PC have `datasets_dir` pointing to another
  project; it does not matter because `cats.yaml` uses an absolute path.
- Evaluation outputs are in `runs/eval/<run>_imgsz<N>[_conf<c>]/`.
- **Next:** phase 3 (NCNN export of the candidate at 320 and PC-side verification of the `ncnn` + `numpy`
  detector).
