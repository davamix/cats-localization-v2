# Phase 7 — Performance optimisation

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phases 5 and 6 |

## Goal

Make the live stream as fast and smooth as the Pi 3B allows without losing the accuracy reached in phase 6.
Every change is measured with `pi/benchmark.py` and the app's `/stats` endpoint, and recorded in
`docs/results.md`.

## Candidate optimisations

Try roughly in this order (cheapest and safest first):

- [ ] **Input size**: smaller `imgsz` (320) and rectangular input matching the 4:3 camera (e.g. 320×256)
      instead of square padding.
- [ ] **NCNN settings**: FP16 storage/arithmetic, thread count, `ncnn` light mode; INT8 quantisation if
      accuracy holds.
- [ ] **Decouple rates**: stream at camera rate, detect at a lower rate, and reuse the last boxes in between.
- [ ] **Cheaper streaming**: use the Pi's hardware MJPEG encoder for the video and send detections separately
      (JSON / Server-Sent Events) so the browser draws the boxes on a `<canvas>` — removes OpenCV drawing and
      JPEG encoding from the CPU.
- [ ] **Motion gating**: run the detector only when frame differencing detects movement.
- [ ] **Thermals**: heatsink/fan if `get_throttled` reports throttling under load.

## Done when

- The chosen settings and their measured gains are documented, and the app runs with them by default.

## Handover notes

_None yet._
