# Phase 2 — Training on the PC

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 1 |

## Goal

Fine-tune YOLO26n on the cats dataset with the RTX 2080 Ti and pick a model and input size for the Pi.

## Steps

- [ ] `train/train.py`: thin wrapper around `ultralytics.YOLO(...).train()` with CLI arguments (`--model`,
      `--imgsz`, `--epochs`, `--batch`, `--name`, `--seed`). Keep the `if __name__ == "__main__":` guard
      (Windows data loader workers need it). Pretrained weights and runs go to ignored folders (`models/`,
      `runs/`).
- [ ] Baseline runs from the COCO-pretrained `yolo26n.pt` at `imgsz` 640, 416 and 320
      (start with ~100 epochs, early-stopping patience ~30, batch 16, fixed seed).
- [ ] `train/evaluate.py`: validation metrics for a given weights file and image size — mAP50, mAP50-95,
      per-class precision/recall, and the confusion matrix. **Blacky ↔ Niche confusion is the key error.**
- [ ] Look at failure cases (wrong class, missed cat, duplicate boxes) on the validation images.
- [ ] Create `docs/results.md` with a table of runs (model, imgsz, epochs, mAP50, mAP50-95, confusions,
      training time) and the chosen candidate.

## Done when

- At least one model detects both cats on the validation set with no or rare Blacky ↔ Niche confusion.
- The deployment candidate (weights + input size) is recorded in `docs/results.md`.

## Notes

- The validation set is video frames (highly correlated), so its scores are optimistic. The real test is the Pi
  camera (phases 4–6).
- Training at the deployment input size usually works better than training at 640 and running at 320.

## Handover notes

_None yet._
