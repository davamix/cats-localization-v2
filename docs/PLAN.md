# Project plan — cats-localization-v2

Detect and identify our two cats (**Blacky** and **Niche**) in real time with a Raspberry Pi 3 Model B and its
camera, and watch the result as a live stream in the browser. More classes (specific people) come later.

This is the successor of [davamix/cats-localization](https://github.com/davamix/cats-localization)
(Detectron2 Mask R-CNN, 2020). Detectron2 is no longer maintained and is far too heavy for a Pi 3B, so v2 uses a
YOLO model exported to NCNN.

## How to use these docs

- This file is the overview: architecture, decisions, environment and the phase index.
- Each phase has its own document in [phases/](phases/) with the goal, the steps (checkboxes), the "done when"
  criteria, the current **status** and **handover notes**.
- At the end of every work session, update the phase document you worked on: tick steps, set the status, write
  what the next session needs to know (results, decisions, problems, next step). Then update the status column
  in the table below.
- Status values: `Not started`, `In progress`, `Blocked`, `Done`.

## Phases

| Phase | Document | Status |
|---|---|---|
| 0 | [Environment and repository setup](phases/phase-0-setup.md) | Done |
| 1 | [Dataset conversion (VIA → YOLO)](phases/phase-1-data.md) | Done |
| 2 | [Training on the PC](phases/phase-2-training.md) | Done |
| 3 | [NCNN export and PC-side verification](phases/phase-3-export.md) | Done |
| 4 | [Deploy to the Pi and baseline benchmark](phases/phase-4-deploy-benchmark.md) | Done |
| 5 | [Live stream web app on the Pi](phases/phase-5-live-stream.md) | Not started |
| 6 | [Real camera data and retraining](phases/phase-6-real-data.md) | Not started |
| 7 | [Performance optimisation](phases/phase-7-performance.md) | Not started |
| 8 | [People (future)](phases/phase-8-people.md) | Not started |

Order rationale: get an end-to-end pipeline working first (phases 1–5), then improve accuracy with data from the
real camera (6), then speed (7). Performance work is intentionally left until everything works.

## Architecture

```
 PC (Windows, RTX 2080 Ti)                          Raspberry Pi 3B (Raspberry Pi OS Lite, arm64)
 ─────────────────────────                          ─────────────────────────────────────────────
 data/ (VIA polygons) ──► tools/via_to_yolo.py      Camera Module v2.1 (IMX219)
        │                                                   │ picamera2 (ISP scales to small frames)
        ▼                                                   ▼
 datasets/cats (YOLO format)                        pi/detector.py  (ncnn + numpy, no torch)
        │                                                   │ boxes + class + score
        ▼                                                   ▼
 train/train.py (Ultralytics, YOLO26n, CUDA)        pi/app.py  (draw boxes, MJPEG over HTTP)
        │                                                   │
        ▼                                                   ▼
 train/export.py ──► NCNN model ──► scripts/deploy.py ──►  browser: http://<pi-ip>:8000
        │
        └──► GitHub Release (weights)
```

## Key decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | **YOLO26n** (Ultralytics) instead of RF-DETR | The Pi 3B is CPU-only (4× Cortex-A53 @ 1.2 GHz, 1 GB RAM). RF-DETR is a transformer meant for GPUs/accelerators and would take seconds per frame; YOLO26n is designed for CPU/edge inference. (Its NMS-free head is not used: see the one-to-many head decision below.) |
| 2026-10-02 | **NCNN** runtime on the Pi; inference code uses only `ncnn` + `numpy` (+ OpenCV for drawing) | NCNN is the fastest Ultralytics export format on Raspberry Pi; avoiding PyTorch/Ultralytics on a 1 GB board saves RAM and install pain. |
| 2026-10-02 | **One class per individual** (`Blacky`, `Niche`) | Simplest design that runs fully on the Pi. Known limitation: closed-set, so an unknown cat will be labelled as one of ours. Revisit in phase 8. |
| 2026-10-02 | Pi output is a **live MJPEG stream** viewed in the browser | User choice. Performance tuning is deferred to phase 7. |
| 2026-10-02 | Trained weights are published as **GitHub Releases**, not committed | Keeps the repo small. Git LFS is available if ever needed. |
| 2026-10-02 | Annotations (`data/*/cats-annotations.json`) are committed; **images are not** | Images are shared as a zip on Google Drive (link in [data/README.md](../data/README.md)). |
| 2026-10-02 | Repo license **AGPL-3.0** | The project uses Ultralytics, which is AGPL-3.0. |
| 2026-10-02 | Pi access via **password SSH** with a paramiko helper ([scripts/pi_remote.py](../scripts/pi_remote.py)), credentials in an untracked `pi.env` | SSH key login was not set up (see phase 0 handover notes). Everything runs on the local network. |
| 2026-10-02 | Deployment candidate: **YOLO26n at input size 320, trained with `scale` 0.9**, confidence 0.5 | Same scores as 416/640 on the (leaky) validation set, ~4× less compute than 640, and the stronger scale augmentation is the only setting that still finds small cats (see [results.md](results.md)). Revisit 416 after the Pi benchmark (phase 4). |
| 2026-10-02 | Detector uses YOLO26's **one-to-many head + NMS in numpy**, not the NMS-free head | Ultralytics cannot export the end-to-end branch to NCNN (no TopK), its PyTorch predict/val use the one-to-many head by default (so every phase 2 number comes from it), and the NMS-free head is weaker on this model (mAP50-95 0.901 vs 0.944). NMS over the few boxes above 0.5 costs almost nothing. |
| 2026-10-02 | The Pi model folder carries a **`model.json`** (class names, input size) written by `train/export.py` | Read with the standard library: the Pi needs no PyYAML for Ultralytics' `metadata.yaml`. |
| 2026-10-02 | The Pi detector runs ncnn in **FP32** by default (FP16 storage/arithmetic off) | ncnn turns FP16 on by default on ARM but not on the PC, so FP32 keeps Pi and PC results comparable. FP16 is tried in phase 7. |
| 2026-10-02 | **Stay at input 320** after the Pi benchmark; no 416 + `scale` 0.9 run for now | On the Pi, 320 gives ~4 FPS sustained, 416 ~2.2 and 640 ~1.1 (2 threads). Revisit 416 only if phase 6 shows missed far-away cats and the Pi has cooling and a supply that holds 4 threads (see [results.md](results.md)). |
| 2026-10-02 | ncnn runs with **2 threads** on the Pi for now | On the current supply, 3–4 busy cores trigger under-voltage (600 MHz cap) and a sustained 4-thread run rebooted the Pi; 2 threads is the fastest setting (4.4 FPS at 320 vs 3.2 at 4 threads). Re-measure 4 threads after the power/cooling fix. |

## Environment

| | |
|---|---|
| **PC** | Windows 11, NVIDIA RTX 2080 Ti (11 GB, sm_75), driver 610.88. Python 3.12 venv in `.venv/` (torch 2.14.1+cu126, Ultralytics 8.4.171, ncnn 1.0.20260526, pnnx 20260526). |
| **Pi** | Raspberry Pi 3 Model B Rev 1.2, Raspberry Pi OS Lite (Debian 13 "trixie", 64-bit, kernel 6.18), Python 3.13.5, 905 MiB RAM + 904 MiB swap. |
| **Pi power / cooling** | Official supply, no heatsink or fan (thermal pads only; heatsink pending). Under-voltage at ≥ 3 busy cores, thermal cap above 80 °C after 1–8 min at 2 threads (phase 4). |
| **Camera** | Camera Module v2.1 (Sony IMX219), detected by libcamera. Full field of view needs the 1640×1232 or 3280×2464 sensor mode; the 640×480 mode is a crop. |
| **Network** | Pi at `192.168.2.112` on the local network (configured in `pi.env`). |

## Dataset (as of 2026-10-02)

| Split | Blacky | Niche | Total | Notes |
|---|---|---|---|---|
| train | 51 | 47 | 98 | Photos, 1920×1080 |
| validation | 21 | 22 | 43 | Video frames, 1920×1080 — consecutive frames are highly correlated, so validation scores will be optimistic |

Every image contains exactly one polygon (one cat). There are no images with both cats together and no images
without cats; phase 6 adds those using the Pi camera. Two more limitations found in phase 1: many validation frames
are near-duplicates of training photos (validation scores will be inflated), and the cats are always large in the
frame (box width ≥ 18% of the image), unlike what a fixed camera across a room will see.

## Repository layout

```
cats-localization-v2/
├── data/                 # images (not in git) + VIA annotation JSONs (in git)
├── datasets/             # generated YOLO dataset (not in git)
├── docs/
│   ├── PLAN.md           # this file
│   ├── results.md        # metrics and benchmarks (created in phase 2)
│   └── phases/           # one document per phase
├── pi/                   # code that runs on the Raspberry Pi
├── scripts/              # PC-side helpers: Pi remote access, deploy
├── tools/                # dataset conversion and visualisation
├── train/                # training, evaluation, export (PC)
├── pi.env.example        # template for the untracked pi.env
├── requirements-train.txt
└── requirements-pi.txt
```
