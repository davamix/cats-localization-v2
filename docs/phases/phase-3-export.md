# Phase 3 — NCNN export and PC-side verification

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 2 |

## Goal

Export the chosen model to NCNN and write the light inference code the Pi will use, verifying it on the PC
first so that preprocessing bugs are caught before touching the Pi.

## Steps

- [x] [train/export.py](../../train/export.py): export `best.pt` to NCNN at the chosen input size (`format="ncnn"`,
      FP32; FP16 later in phase 7). Output: a folder with `model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`, plus
      a `model.json` (class names, input size) for the Pi.
- [x] Inspect the exported model's input/output blobs and output shape. **Not** end-to-end: Ultralytics disables
      YOLO26's NMS-free branch for NCNN, so the output is the raw one-to-many head, (6, 2100) =
      `[cx, cy, w, h, score Blacky, score Niche]` per anchor, which needs NMS (details below).
- [x] [pi/detector.py](../../pi/detector.py) — depends only on `ncnn` and `numpy`:
  - Letterbox resize to the model input size (keep aspect ratio, pad), BGR → RGB, scale to `[0, 1]`.
  - Run the network with a configurable number of threads.
  - Decode the output, apply the score threshold, map boxes back to the original frame.
  - Class names read from `model.json`, written by `export.py` from the exported `metadata.yaml`.
- [x] [tools/verify_ncnn.py](../../tools/verify_ncnn.py): run `pi/detector.py` on the validation images on the PC and
      compare with the Ultralytics PyTorch predictions (same boxes within IoU tolerance, same classes, similar mAP).
      Also ran [train/evaluate.py](../../train/evaluate.py) on the NCNN model folder.
- [x] Publish the first GitHub Release
      ([`v0.1.0`](https://github.com/davamix/cats-localization-v2/releases/tag/v0.1.0)): `best.pt`, the zipped NCNN
      model and a short metrics summary.

## How to run

```powershell
.\.venv\Scripts\python.exe train\export.py yolo26n_320_scale0.9
#   -> runs/train/yolo26n_320_scale0.9/weights/best_ncnn_model/ (~15 s), prints the blob names and output shape
.\.venv\Scripts\python.exe tools\verify_ncnn.py yolo26n_320_scale0.9 --shrink 1 0.5 0.25     # exit code 1 on failure
.\.venv\Scripts\python.exe train\evaluate.py runs\train\yolo26n_320_scale0.9\weights\best_ncnn_model --shrink 1 0.5 0.25 --conf 0.5
#   -> runs/eval/yolo26n_320_scale0.9_imgsz320_ncnn/
.\.venv\Scripts\python.exe pi\detector.py <model folder> image.jpg [...] [--json detections.json]   # also on the Pi
```

The first export installs `pnnx==20260526` (the version Ultralytics pins); `ncnn` is installed separately with
`--no-deps` (see [requirements-train.txt](../../requirements-train.txt)). Nothing else was added to the venv.

## Done when

- [x] The NCNN detector on the PC matches the PyTorch model on the validation set.
- [x] Release `v0.1.0` exists with the weights.

## Results

Full tables in [docs/results.md](../results.md#phase-3--ncnn-export-and-pc-side-verification-2026-10-02).

- **NCNN = PyTorch.** On the same input the raw outputs differ by at most 0.005 px (boxes) and 7e-6 (scores). At
  confidence 0.5, all 143 boxes over ×1/×0.5/×0.25 pair up with IoU 1.0000 and identical scores (to 4 decimals);
  mAP is identical. The detector's letterbox is bit-identical to Ultralytics' `LetterBox`. A deliberately broken
  detector (BGR fed as RGB) makes the check fail, so it is not vacuous.
- **Square vs rectangular input.** The NCNN model takes a fixed square 320×320 input; Phase 2's PyTorch evaluation
  used 320×192 for the 16:9 frames. mAP is the same or slightly better (×0.25 mAP50 0.978 vs 0.960), and at
  confidence 0.5 there are one or two more errors per split (×1: 1 missed / 5 FP vs 0 / 4), all of the known kinds
  (dark objects as Blacky, Niche's dark patch as Blacky in `niche_frame40`).
- **Speed on the PC** (i7-8700K, 4 threads): ~2.5 ms preprocess, 17–20 ms inference, 0.4 ms postprocess. Reference
  only.

## Handover notes

- **Model for phase 4:** `runs/train/yolo26n_320_scale0.9/weights/best_ncnn_model/` on this PC, or
  `yolo26n_320_scale0.9_ncnn_model.zip` from Release
  [v0.1.0](https://github.com/davamix/cats-localization-v2/releases/tag/v0.1.0) (same files, inside a
  `yolo26n_320_scale0.9_ncnn_model/` folder). The detector needs `model.ncnn.param`, `model.ncnn.bin` and
  `model.json`; `metadata.yaml` is only for Ultralytics. For other input sizes (phase 4 benchmark):
  `train/export.py yolo26n_416` / `yolo26n_640` (speed depends only on the input size).
- **Output layout:** `in0` (3, 320, 320) → `out0` (6, 2100); rows `cx, cy, w, h` in input pixels and two sigmoid
  scores. **NMS is needed** and done in numpy (best class per anchor, score > conf, class-aware NMS at IoU 0.7,
  max 300), exactly like Ultralytics. YOLO26's NMS-free head is *not* used: Ultralytics cannot export it to NCNN,
  it is weaker on this model (mAP50-95 0.901 vs 0.944), and every Phase 2 number came from the one-to-many head
  anyway (Ultralytics' default unless `nms=False`).
- **Detector API:** `Detector(model_dir, conf=0.5, iou=0.7, threads=4, fp16=False)`, `detect(frame_bgr)` →
  (N, 6) float32 `x1, y1, x2, y2, score, class_id` in frame pixels; `names[class_id]`. `preprocess` / `infer` /
  `postprocess` are separate methods so `pi/benchmark.py` can time each stage. The CLI (`python pi/detector.py
  <model> <images> --json out.json`) reads images with OpenCV and writes the detections, for the Pi-vs-PC
  comparison in phase 4: run it on the same images on both and compare the JSON files.
- **FP32 on the Pi:** ncnn enables FP16 storage/arithmetic by default on ARM, which would not happen on the PC (x86).
  The detector turns it off unless `fp16=True`, so the Pi should match the PC; FP16 is a phase 7 experiment
  (`--fp16` in the CLI, `train/export.py --half` for FP16 weights).
- **ncnn gotcha:** `ncnn.Mat(numpy_array)` shares the array's memory without copying. If the array is a temporary
  that is freed before `extract()`, ncnn reads freed memory: garbage outputs or a segfault (seen while writing
  `export.py`). The detector avoids it (`from_pixels_resize` copies); keep both the array and the Mat referenced
  in any new code that builds a Mat from numpy.
- **Camera frames are 4:3.** A 640×480 frame becomes 320×240 + 40 px grey bands top and bottom. Exporting a
  rectangular 320×256 model (phase 7) would cut ~20% of the compute and be closer to the rectangular inference
  Phase 2 measured. `export.py` and `verify_ncnn.py` assume a square input today (`verify_ncnn.py` refuses
  rectangular models).
- `evaluate.py` now accepts an exported model folder (Ultralytics loads it; runs on the CPU) and names its output
  `<run>_imgsz<N>_ncnn`.
- The validation set is still leaky; real accuracy comes from Pi-camera images (phases 4–6).
- **Next:** phase 4 — `scripts/deploy.py`, run `pi/detector.py` on the Pi on the same validation images and compare
  with the PC JSON, then `pi/benchmark.py`.
