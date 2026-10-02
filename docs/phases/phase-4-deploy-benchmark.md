# Phase 4 — Deploy to the Pi and baseline benchmark

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 3 |

## Goal

Run the NCNN detector on the Pi, confirm it gives the same results as on the PC, and measure the baseline speed.
This is a measurement phase, not an optimisation phase (that is phase 7).

## Steps

- [ ] `scripts/deploy.py`: upload `pi/`, `requirements-pi.txt` and a model folder to
      `~/cats-localization-v2/` on the Pi using [scripts/pi_remote.py](../../scripts/pi_remote.py).
- [ ] Run `pi/detector.py` on a few validation images on the Pi and compare the detections with the PC output.
- [ ] `pi/benchmark.py`: time the detector over many frames (camera frames and/or test images) and report mean
      and p95 latency, FPS, peak memory (RSS), CPU temperature and `vcgencmd get_throttled` flags.
- [ ] Benchmark the exported model sizes from phase 2 (e.g. 320 / 416 / 640) with 4 threads.
- [ ] Add the results to `docs/results.md`.

## Done when

- Detections on the Pi match the PC for the same images.
- Baseline latency/FPS per input size is recorded.

## Notes

- A Pi 3B under sustained load can reach its thermal limit (~80 °C) and throttle. Watch the temperature and
  `get_throttled` flags during long runs; a heatsink may be needed.
- Expected order of magnitude (estimate, to be measured): ~2 FPS at 640, more at 320.

## Handover notes

_None yet._
