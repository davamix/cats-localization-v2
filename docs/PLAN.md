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
| 2 | [Training on the PC](phases/phase-2-training.md) | Not started |
| 3 | [NCNN export and PC-side verification](phases/phase-3-export.md) | Not started |
| 4 | [Deploy to the Pi and baseline benchmark](phases/phase-4-deploy-benchmark.md) | Not started |
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
| 2026-10-02 | **YOLO26n** (Ultralytics) instead of RF-DETR | The Pi 3B is CPU-only (4× Cortex-A53 @ 1.2 GHz, 1 GB RAM). RF-DETR is a transformer meant for GPUs/accelerators and would take seconds per frame; YOLO26n is designed for CPU/edge inference and is NMS-free. |
| 2026-10-02 | **NCNN** runtime on the Pi; inference code uses only `ncnn` + `numpy` (+ OpenCV for drawing) | NCNN is the fastest Ultralytics export format on Raspberry Pi; avoiding PyTorch/Ultralytics on a 1 GB board saves RAM and install pain. |
| 2026-10-02 | **One class per individual** (`Blacky`, `Niche`) | Simplest design that runs fully on the Pi. Known limitation: closed-set, so an unknown cat will be labelled as one of ours. Revisit in phase 8. |
| 2026-10-02 | Pi output is a **live MJPEG stream** viewed in the browser | User choice. Performance tuning is deferred to phase 7. |
| 2026-10-02 | Trained weights are published as **GitHub Releases**, not committed | Keeps the repo small. Git LFS is available if ever needed. |
| 2026-10-02 | Annotations (`data/*/cats-annotations.json`) are committed; **images are not** | Images are shared as a zip on Google Drive (link in [data/README.md](../data/README.md)). |
| 2026-10-02 | Repo license **AGPL-3.0** | The project uses Ultralytics, which is AGPL-3.0. |
| 2026-10-02 | Pi access via **password SSH** with a paramiko helper ([scripts/pi_remote.py](../scripts/pi_remote.py)), credentials in an untracked `pi.env` | SSH key login was not set up (see phase 0 handover notes). Everything runs on the local network. |

## Environment

| | |
|---|---|
| **PC** | Windows 11, NVIDIA RTX 2080 Ti (11 GB, sm_75), driver 610.88. Python 3.12 venv in `.venv/`. |
| **Pi** | Raspberry Pi 3 Model B Rev 1.2, Raspberry Pi OS Lite (Debian 13 "trixie", 64-bit, kernel 6.18), Python 3.13.5, 905 MiB RAM + 904 MiB swap. |
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
