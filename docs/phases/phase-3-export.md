# Phase 3 — NCNN export and PC-side verification

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 2 |

## Goal

Export the chosen model to NCNN and write the light inference code the Pi will use, verifying it on the PC
first so that preprocessing bugs are caught before touching the Pi.

## Steps

- [ ] `train/export.py`: export `best.pt` to NCNN at the chosen input size (`format="ncnn"`, FP32 first;
      FP16 later in phase 7). Output: a folder with `model.ncnn.param`, `model.ncnn.bin` and `metadata.yaml`.
- [ ] Inspect the exported model's input/output blobs and output shape. YOLO26 is end-to-end (NMS-free); confirm
      the actual output layout (expected: up to 300 detections × `[x1, y1, x2, y2, score, class]`, but verify).
- [ ] `pi/detector.py` — depends only on `ncnn` and `numpy`:
  - Letterbox resize to the model input size (keep aspect ratio, pad), BGR → RGB, scale to `[0, 1]`.
  - Run the network with a configurable number of threads.
  - Decode the output, apply the score threshold, map boxes back to the original frame.
  - Class names read from the exported `metadata.yaml` (or a small JSON next to the model).
- [ ] `tools/verify_ncnn.py`: run `pi/detector.py` on the validation images on the PC and compare with the
      Ultralytics PyTorch predictions (same boxes within IoU tolerance, same classes, similar mAP).
- [ ] Publish the first GitHub Release (`v0.1.0`): `best.pt`, the zipped NCNN model and a short metrics summary.

## Done when

- The NCNN detector on the PC matches the PyTorch model on the validation set.
- Release `v0.1.0` exists with the weights.

## Handover notes

_None yet._
