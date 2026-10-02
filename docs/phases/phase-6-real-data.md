# Phase 6 — Real camera data and retraining

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 5 |

## Goal

Improve accuracy where it matters: images from the Pi camera, at its real position and lighting. The 2020
dataset was taken with a different camera, has one cat per image and no empty scenes.

## Steps

- [ ] Capture tool on the Pi: save frames periodically and/or on demand (a snapshot button or endpoint in the
      web app from phase 5).
- [ ] Download the captures to the PC (`data/pi-camera/...`, not in git; add to the Drive zip).
- [ ] Pre-label the new images with the current model, then review/correct them in an annotation tool
      (decide which: VIA as in 2020, CVAT, Label Studio…). Keep the format convertible by `tools/`.
- [ ] Include:
  - images with **both cats** in the same frame,
  - **empty scenes** (no cats) as negatives — YOLO uses images with empty label files as background,
  - different times of day / lighting.
- [ ] Keep a held-out **Pi-camera test set** that is never used for training.
- [ ] Retrain (phase 2 scripts), re-export and verify (phase 3), redeploy (phase 4), and compare against the
      previous model on the Pi-camera test set.
- [ ] Publish a new Release when the new model is better.

## Done when

- The Pi-camera test set exists and the new model beats `v0.1.0` on it.

## Handover notes

_None yet._
