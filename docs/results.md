# Results

Metrics and benchmarks for the cats detector. Newest phase at the top of each section.

## Phase 4 — deployment to the Pi and baseline benchmark (2026-10-02)

Raspberry Pi 3 Model B (4× Cortex-A53 at 1.2 GHz, 905 MiB), Raspberry Pi OS Lite 64-bit (trixie, kernel 6.18),
Python 3.13.5, ncnn 1.0.20260526 (pip), numpy 2.2.4 and OpenCV 4.10 (apt), `ondemand` governor. Official power
supply, **no heatsink or fan** (the user added thermal pads during the long runs). Detector in FP32 with the phase 3
models (`yolo26n_320_scale0.9`, `yolo26n_416`, `yolo26n_640`; speed depends only on the input size), confidence
0.5. Measured with [pi/benchmark.py](../pi/benchmark.py) on camera frames (640×480, full-field-of-view 1640×1232
sensor mode scaled by the ISP, as in the live app), 10 warm-up frames, every run started at ≤ 55 °C.

> **Power and heat limit every number below.** With 3 or 4 busy cores the Pi reports **under-voltage** and the
> firmware caps the CPU at **600 MHz**; a sustained 4-thread run made it **reboot** after ~4.5 min (most likely a
> brownout). With 2 threads, the CPU reaches **80 °C in 1–8 min** and then runs at 820–1000 MHz. These are the
> numbers of the Pi on 2026-10-02. **The 2026-10-03 rerun below, after a cooling upgrade, replaces the 2-thread
> numbers**; the power limit at 3–4 threads is still there.

### Rerun after the cooling and power-supply upgrade (2026-10-03)

The user upgraded the cooling and changed the supply to a USB-C charger (5 V, 3.6 A, 18 W) feeding the Pi through a
USB-C → micro-USB adapter. Idle temperature dropped from 53 °C to 39 °C. Same models, camera frames and
settings as above, except that runs start at ≤ 45 °C. Every run was guarded by a watcher on the Pi: it logged
temperature, clock and throttle flags every 2 s (synced to disk, so the log would survive a reset) and stopped
the benchmark after 30 s of continuous under-voltage. There was no reset.

**4 threads is still blocked by under-voltage.** A 60 s probe at 320 / 4 threads hit under-voltage
(`0x50005`, 600 MHz) as soon as the 4 threads started, and the watcher stopped it after 30 s. The temperature was
only 47 °C at that point. A check without the camera (4 threads on test images) did the same, so the camera is not
the trigger. There was also one 4-s under-voltage during boot. The likely cause is the power path: 5.0 V instead
of the official supply's 5.1 V, plus the adapter's contact resistance and the cable.

**2 threads, 10 minutes each:**

| Input | Frames | First 30 s: total (FPS) | Last 2 min: total (FPS) | Inference mean / p95 | Capture / preprocess / postprocess | Peak RSS | Temp start → max | Clock | Under-voltage samples | FPS (10 min) |
|---|---|---|---|---|---|---|---|---|---|---|
| **320** | 2628 | 226 ms (4.43) | 234 ms (4.27) | 211 / 224 ms | 7.5 / 7.7 / 1.4 ms | 205 MiB | 43 → 69 °C | 1200 MHz (4 short dips) | 2 / 121 | **4.39** |
| 416 | 1615 | 371 ms (2.70) | 371 ms (2.70) | 351 / 365 ms | 7.6 / 10.8 / 1.5 ms | 226 MiB | 50 → 70 °C | 1200 MHz | 0 / 121 | **2.70** |
| 640 | 705 | 847 ms (1.18) | 850 ms (1.18) | 827 / 843 ms | 7.7 / 13.9 / 2.0 ms | 277 MiB | 53 → 70 °C | 1200 MHz | 0 / 121 | **1.18** |

- **The thermal limit is gone at 2 threads.** The CPU settles at ~70 °C (it reached 82–84 °C before), stays at
  1.2 GHz, and the latency stays flat for the whole 10 minutes. FPS over 10 minutes improved by 9–13% (320:
  4.01 → 4.39, 416: 2.39 → 2.70, 640: 1.08 → 1.18), and by up to 21% in the last 2 minutes (416: 2.23 → 2.70).
  These match the cool-CPU numbers of the first run, now held for the whole run.
- 320 still had 4 short under-voltage dips (kernel log; 2 of them fell on the benchmark's 5-s samples), which
  cost it a few percent in the last 2 minutes. The other two sizes had none.

### Pi vs PC ([tools/compare_detections.py](../tools/compare_detections.py))

`pi/detector.py --json` on the same 12 images on both (8 validation frames including the hard cases
`blacky_frame1180`, `blacky_frame340/360`, `niche_frame40/140`, plus 4 frames from the ×0.25 shrunken set),
4 threads, FP32, confidence 0.25:

| Images | Boxes PC / Pi / paired | Min IoU | Max score diff | Max corner diff |
|---|---|---|---|---|
| 12 | 20 / 20 / 20 | 1.0000 | 0.000 | 0.001 px |

The JSON files round to 3 decimals, so the Pi and the PC agree to within that rounding. A doctored file (one
shifted box, one changed class, one score −0.05, one missing box) fails all four checks.

### Threads (320, camera, 300 frames)

| Threads | Capture | Preprocess | Inference mean / p95 | Total mean | FPS | ARM clock mean | Under-voltage samples | Max temp |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.7 ms | 7.0 ms | 319 / 338 ms | 335 ms | 2.98 | 1200 MHz | 0 / 21 | 75 °C |
| **2** | 7.5 ms | 7.3 ms | **213 / 219 ms** | 229 ms | **4.36** | 1153 MHz | 1 / 14 | 81 °C |
| 3 | 14.8 ms | 12.0 ms | 310 / 323 ms | 340 ms | 2.94 | 714 MHz | 17 / 21 | 70 °C |
| 4 | 14.8 ms | 12.3 ms | 285 / 298 ms | 314 ms | 3.18 | 632 MHz | 18 / 19 | 69 °C |

- **2 threads is the fastest setting on this supply.** At 3 and 4 threads the supply voltage drops below the
  threshold almost at once (`get_throttled` 0x50005: under-voltage + throttled), the clock halves, and every stage
  slows down (capture and preprocessing too). Postprocessing (NMS in numpy) is 1.3–2.4 ms.
- At full clock, 2 threads are 1.5× faster than 1. How fast 4 threads would be at 1.2 GHz is unknown until the
  power is fixed.
- On 1920×1080 images (no camera), preprocessing takes 6 ms at full clock; on 640×480 camera frames it takes
  7–9 ms, because picamera2 does its own work in the background.

### Input size (2 threads, camera, 10 minutes each)

| Input | Frames | First 30 s: total (FPS) | Last 2 min: total (FPS) | Inference mean / p95 | Capture / preprocess / postprocess | Peak RSS | Temp start → max | 80 °C after | Clock in last 2 min |
|---|---|---|---|---|---|---|---|---|---|
| **320** | 2401 | 225 ms (**4.44**) | 248 ms (**4.03**) | 231 / 350 ms | 8.8 / 8.2 / 1.6 ms | 203 MiB | 56 → 83 °C | 465 s | ~1000 MHz |
| 416 | 1429 | 368 ms (2.72) | 448 ms (2.23) | 396 / 432 ms | 9.2 / 12.4 / 1.8 ms | 225 MiB | 61 → 84 °C | 85 s | ~870 MHz |
| 640 | 645 | 847 ms (1.18) | 902 ms (1.11) | 903 / 1003 ms | 8.9 / 14.6 / 3.2 ms | 278 MiB | 64 → 84 °C | 76 s | ~1070 MHz |

Overall FPS (whole 10 min): 4.01 / 2.39 / 1.08.

- **Inference time scales with the number of pixels**: on a cool CPU, 416 costs 1.67× and 640 3.95× the time of 320
  ((416/320)² = 1.69, (640/320)² = 4).
- **Thermal throttling**: every size ends at 82–84 °C with the clock capped to 820–1070 MHz (`get_throttled`
  0x70002, "ARM frequency capped"), and the latency grows by 10–20% after 1–3 minutes (416: 368 → 448 ms). The
  320 run reached 80 °C later because it started cooler and its many under-voltage dips (27 in 7 minutes, each
  4–16 s at 600 MHz) also cut the heat; those dips also cause its high p95 (350 ms).
- **4 threads, sustained (320)**: the Pi rebooted ~4.5 minutes into the run (no log survives: the journal is not
  persistent). The filesystem recovered cleanly. Not repeated, to avoid more hard resets of the SD card.
- **Memory**: ~130 MiB after the imports (apt OpenCV, numpy, ncnn), +24 MiB for the model, 200–280 MiB peak
  with the camera running. At least 565 MiB of the 905 MiB stay available, so memory is not a problem.
- **SSH under load**: while the Pi is at its thermal limit, SSH logins take 20 s to over 2 minutes (ping stays at
  5 ms; no SD-card errors in the kernel log).
- The camera pointed at the ceiling; frame content changes only the postprocessing time, which is negligible.

### Is 320 fast enough, or is 416 worth revisiting?

**Stay at 320.** With the new cooling, 320 gives ~4.4 FPS sustained at 2 threads, 416 ~2.7 FPS and 640 ~1.2 FPS
(before the upgrade: ~4.0 / 2.2 / 1.1). With the live stream decoupled from detection (phase 5 design), ~4
detections per second keeps the boxes ~0.23 s behind the video, which is fine for watching cats. At 416 the boxes
would lag ~0.37 s, for an accuracy gain that the (leaky) phase 2 validation cannot show. So a 416 + `scale` 0.9
model is **not** worth training now. Revisit 416 only if both:

1. Pi-camera images (phase 6) show the 320 model missing cats far from the camera, and
2. the Pi's supply holds 4 threads (the cooling is now good enough). A rectangular 416×320 input (1.3× the
   pixels of 320²) and FP16 (phase 7) might then make 416 affordable; that has to be measured.

## Phase 3 — NCNN export and PC-side verification (2026-10-02)

Model: `yolo26n_320_scale0.9` (seed 0), exported with [train/export.py](../train/export.py) to NCNN (PNNX 20260526,
ncnn 1.0.20260526), FP32, fixed input 1×3×320×320. Published in Release
[v0.1.0](https://github.com/davamix/cats-localization-v2/releases/tag/v0.1.0). Same leaky validation set as
phase 2: these numbers show that NCNN matches PyTorch, not real-world accuracy.

### Exported model

| | |
|---|---|
| Files | `model.ncnn.param` (26 KB, 320 layers), `model.ncnn.bin` (9.5 MB), `metadata.yaml`, `model.json` |
| Input | `in0`: (3, 320, 320) RGB in [0, 1], letterboxed (aspect ratio kept, grey 114 padding) |
| Output | `out0`: (6, 2100) — one column per anchor (40² + 20² + 10² at strides 8/16/32), rows `cx, cy, w, h` (input pixels), `score Blacky`, `score Niche` (sigmoid) |
| Postprocessing | not in the graph: best class per anchor, score > 0.5, class-aware NMS at IoU 0.7, undo the letterbox ([pi/detector.py](../pi/detector.py)) |

**Which head.** YOLO26 has two detection heads: a one-to-many head that needs NMS and an end-to-end (NMS-free)
one-to-one head. Ultralytics 8.4.171 disables the end-to-end branch for NCNN (no TopK in NCNN), and its PyTorch
predict/val also use the one-to-many head + NMS unless `nms=False` is passed, so **every phase 2 number already
comes from the one-to-many head**. The NMS-free head is weaker on this fine-tuned model (PyTorch, ×1 validation):

| Head | mAP50 | mAP50-95 | Boxes at confidence 0.5 (43 cats) |
|---|---|---|---|
| one-to-many + NMS (Ultralytics default, exported) | 0.995 | 0.944 | 47 |
| one-to-one, NMS-free (`nms=False`) | 0.966 | 0.901 | 35 |

### NCNN detector vs PyTorch ([tools/verify_ncnn.py](../tools/verify_ncnn.py))

`pi/detector.py` (ncnn + numpy) against Ultralytics PyTorch predict with the same square 320×320 letterbox (CPU,
FP32), confidence 0.5 for the box comparison and 0.001 for mAP:

| Validation | Raw output max \|diff\| (boxes / scores) | Boxes PyTorch / NCNN / paired | Min IoU of pairs | Max score diff | mAP50 (both) | mAP50-95 (both) |
|---|---|---|---|---|---|---|
| ×1 | 0.0025 px / 4e-6 | 47 / 47 / 47 | 1.0000 | < 0.0001 | 0.9926 | 0.9474 |
| ×0.5 | 0.0052 px / 4e-6 | 47 / 47 / 47 | 1.0000 | < 0.0001 | 0.9703 | 0.8914 |
| ×0.25 | 0.0029 px / 7e-6 | 49 / 49 / 49 | 1.0000 | < 0.0001 | 0.9776 | 0.7610 |

- The detector's preprocessing (ncnn's `from_pixels_resize` + `copy_make_border`) is bit-identical to Ultralytics'
  `LetterBox` on the validation frames (max pixel difference 0/255).
- Negative check: feeding the network BGR instead of RGB makes the script fail (10 box differences, mAP50-95
  −0.026 at ×1), so the comparison does catch preprocessing bugs.
- Detector time on the PC (i7-8700K, 4 threads, median): preprocess 2.5 ms, inference 17–20 ms, postprocess
  0.4 ms. Only a reference; the Pi is measured in phase 4.

### NCNN model through train/evaluate.py, confidence 0.5

Ultralytics' own NCNN backend. NCNN models take a fixed square 320×320 input; the phase 2 PyTorch evaluation used
rectangular 320×192 input for the 16:9 frames. Errors as missed / false positives (incl. duplicates), 43 cats per
column; no Blacky ↔ Niche confusion in either.

| Model (input) | ×1 mAP50-95 | ×1 | ×0.5 mAP50-95 | ×0.5 | ×0.25 mAP50 | ×0.25 mAP50-95 | ×0.25 |
|---|---|---|---|---|---|---|---|
| PyTorch (rect 320×192, phase 2) | 0.944 | 0 / 4 | 0.882 | 2 / 5 | 0.960 | 0.762 | 1 / 6 |
| NCNN (square 320×320) | 0.947 | 1 / 5 | 0.891 | 2 / 6 | 0.978 | 0.761 | 2 / 8 |

The differences come from the input shape, not from NCNN (NCNN and PyTorch give identical boxes on the same input,
see above), and are within the seed-to-seed noise of phase 2. The extra errors are the known ones: dark objects
labelled Blacky (`blacky_frame1180`) and Niche's dark patch labelled Blacky in `niche_frame40`, which with the
square input now also happens at ×1 (Blacky 0.71, and the Niche box is lost).

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
  - Predictions from YOLO26's one-to-many head + NMS (IoU 0.7), Ultralytics' default; not the NMS-free head (see
    phase 3).
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
