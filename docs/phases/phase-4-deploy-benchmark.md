# Phase 4 — Deploy to the Pi and baseline benchmark

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 3 |

## Goal

Run the NCNN detector on the Pi, confirm it gives the same results as on the PC, and measure the baseline speed.
This is a measurement phase, not an optimisation phase (that is phase 7).

## Steps

- [x] [scripts/deploy.py](../../scripts/deploy.py): upload `pi/`, `requirements-pi.txt`, model folders and test
      images to `~/cats-localization-v2/` on the Pi using [scripts/pi_remote.py](../../scripts/pi_remote.py).
- [x] Run `pi/detector.py` on a few validation images on the Pi and compare the detections with the PC output
      ([tools/compare_detections.py](../../tools/compare_detections.py)).
- [x] [pi/benchmark.py](../../pi/benchmark.py): time the detector over many frames (camera frames and/or test images)
      and report mean and p95 latency, FPS, peak memory (RSS), CPU temperature and `vcgencmd get_throttled` flags.
- [x] Threads 1 / 2 / 3 / 4 at 320 (300 camera frames each).
- [x] Benchmark the exported model sizes from phase 2 (320 / 416 / 640), 10 minutes each — **with 2 threads, not 4**:
      on the current supply 3–4 threads trigger under-voltage (600 MHz cap), and the sustained 4-thread run at 320
      rebooted the Pi, so the user chose to run the long tests at 2 threads only.
- [x] Add the results to [docs/results.md](../results.md#phase-4--deployment-to-the-pi-and-baseline-benchmark-2026-10-02).

## How to run

```powershell
# PC: export the models (FP32), then deploy code + models (+ test images) to ~/cats-localization-v2/ on the Pi
.\.venv\Scripts\python.exe train\export.py yolo26n_416                # and yolo26n_640; 320_scale0.9 is from phase 3
.\.venv\Scripts\python.exe scripts\deploy.py yolo26n_320_scale0.9 yolo26n_416 yolo26n_640 --images runs\pi\compare --set compare
.\.venv\Scripts\python.exe scripts\deploy.py                           # code only (pi/ + requirements-pi.txt)
```

Layout on the Pi (`pi/`, each `models/<name>` and `images/<set>` are replaced as a whole on every deploy):

```
~/cats-localization-v2/
├── .venv/                 ncnn (pip) + system numpy / OpenCV / picamera2 (not touched by deploy.py)
├── requirements-pi.txt
├── pi/                    detector.py, benchmark.py, camera_test.py
├── models/<name>/         model.ncnn.param, model.ncnn.bin, model.json   (<name> = training run name)
├── images/<set>/          test images
└── results/               outputs of benchmark.py / detector.py --json (copied to runs/pi/ on the PC by hand)
```

Pi vs PC comparison (Git Bash; the images are 8 validation frames + 4 ×0.25 frames copied to `runs/pi/compare/`):

```bash
.venv/Scripts/python.exe pi/detector.py runs/train/yolo26n_320_scale0.9/weights/best_ncnn_model runs/pi/compare/*.jpg --conf 0.25 --json runs/pi/compare_pc.json
.venv/Scripts/python.exe scripts/pi_remote.py run 'cd ~/cats-localization-v2 && mkdir -p results && .venv/bin/python pi/detector.py models/yolo26n_320_scale0.9 images/compare/*.jpg --conf 0.25 --json results/compare_pi.json'
MSYS_NO_PATHCONV=1 .venv/Scripts/python.exe scripts/pi_remote.py get '~/cats-localization-v2/results/compare_pi.json' runs/pi/compare_pi.json
.venv/Scripts/python.exe tools/compare_detections.py runs/pi/compare_pc.json runs/pi/compare_pi.json    # exit 1 on mismatch
```

Benchmark (on the Pi, from `~/cats-localization-v2`):

```bash
.venv/bin/python pi/benchmark.py models/yolo26n_320_scale0.9 --camera --threads 2 --duration 600 --cool-to 55 --json results/bench.json
.venv/bin/python pi/benchmark.py models/yolo26n_320_scale0.9 --images images/compare --frames 300
```

Options: `--threads`, `--frames N` or `--duration S`, `--warmup 10`, `--conf 0.5`, `--cool-to °C` (wait before
starting; the camera is opened afterwards), `--sample-every 5` (temperature / clock / throttle samples),
`--width/--height` and `--sensor-size` for the camera. Long runs must survive an SSH disconnect: start them with
`(setsid nohup ... > log 2>&1 < /dev/null &)` (see the handover notes).

The phase 4 suite ran these from a `setsid nohup` job on the Pi (scripts kept in `runs/pi/phase4/` on the PC, not
in git), each with `--camera --cool-to 55 --cool-timeout 900 --json results/phase4/<name>.json`:
`--threads 1|2|3|4 --frames 300` on `models/yolo26n_320_scale0.9`, and `--threads 2 --duration 600` on each of the
three models. The results (`*.json`, `*.log`) were copied to `runs/pi/phase4/`.

## Done when

- [x] Detections on the Pi match the PC for the same images.
- [x] Baseline latency/FPS per input size is recorded (at 2 threads; see the power and heat limits below).

## Notes

- A Pi 3B under sustained load can reach its thermal limit (~80 °C) and throttle. Watch the temperature and
  `get_throttled` flags during long runs; a heatsink may be needed.
- Expected order of magnitude (estimate, to be measured): ~2 FPS at 640, more at 320. Measured: ~1.1 FPS at 640,
  ~4 FPS at 320 (2 threads, throttled Pi).

## Results

Full tables in [docs/results.md](../results.md#phase-4--deployment-to-the-pi-and-baseline-benchmark-2026-10-02).

- **Pi = PC.** On 12 images (8 validation frames incl. the hard cases + 4 ×0.25 frames), confidence 0.25: 20/20
  boxes paired, IoU 1.0000, identical scores and corners to the 3 decimals of the JSON files.
- **Speed (320, 2 threads, camera 640×480):** ~4.4 FPS on a cool CPU, **~4.0 FPS sustained** (inference
  208 → 229 ms, capture ~8 ms, preprocess ~8 ms, postprocess ~1.5 ms). 416: 2.7 → 2.2 FPS. 640: 1.2 → 1.1 FPS.
  Inference time scales with the number of pixels.
- **Threads (320):** 1 → 2.98 FPS, **2 → 4.36 FPS**, 3 → 2.94 FPS, 4 → 3.18 FPS. 3–4 threads are slower because
  of under-voltage (see below).
- **Memory:** 200–280 MiB peak RSS (imports ~130 MiB, model +24 MiB, camera and the rest), ≥ 565 MiB still
  available. Not a constraint.
- **Power and heat (the main finding):**
  - With 3–4 busy cores the official supply sags: `get_throttled` reports under-voltage at once and the firmware
    caps the CPU at 600 MHz. A sustained 4-thread camera run **rebooted the Pi** after ~4.5 min (most likely a
    brownout).
  - Even at 2 threads there were 27 short under-voltage dips in the 320 run.
  - Without a heatsink, 2 threads reach 80 °C in 1–8 min and the firmware caps the clock to 820–1000 MHz
    (+10–20% latency).
- **Decision:** stay at input 320 (`yolo26n_320_scale0.9`). A 416 + `scale` 0.9 model is not worth training now.
  Revisit only if Pi-camera images (phase 6) show missed far-away cats *and* the Pi has cooling and a supply that
  holds 4 threads.

## Handover notes

- **Hardware first.** The user is getting a heatsink. The Pi has only thermal pads now.
  - The user asked to **stop load tests if the Pi resets again** until then.
  - The under-voltage also needs a look: the official supply should hold a Pi 3B at full load. Check the cable and
    connector, and whether anything else draws power from the Pi; try another 5.1 V / 2.5 A supply.
  - After the fix, rerun the 4-thread runs (`--threads 4 --duration 600` at 320, then 416) to see whether 4
    threads beat 2 at full clock. The `get_throttled` "since boot" bits are sticky, so judge each run by the
    per-sample `throttled` values in the JSON timeline (bit 0 = under-voltage now, bit 1 = frequency capped now).
- **For phase 5 (`pi/app.py`):**
  - Default to **`--threads 2`**. The app also captures, draws and JPEG-encodes. If that keeps a third core busy,
    the Pi may hit the under-voltage cap even with 2 ncnn threads, so put temperature, ARM clock and the throttle
    flags in `/stats`. `pi/benchmark.py` has the helpers (`cpu_temp`, `arm_clock_mhz`, `throttled`) and can be
    imported.
  - Expect about 4 detections/s at 320. Capture (picamera2 `capture_array`, 640×480) costs ~8 ms.
- **Deploy:** `scripts/deploy.py [models...] [--images ... --set name]`.
  - It replaces `pi/`, `models/<name>/` and `images/<set>/` on the Pi as a whole. Model names are the training run
    names, so the app should take `--model models/yolo26n_320_scale0.9`.
  - Pi results go in `~/cats-localization-v2/results/`; copy them to `runs/pi/` (git-ignored) with `pi_remote.py get`.
- **Long jobs on the Pi:**
  - Start them with `(setsid nohup bash job.sh > job.log 2>&1 < /dev/null &)`. Without the subshell and
    redirections, `pi_remote.py run` keeps the SSH channel open until the job ends.
  - `pi_remote.py run` has no timeout, and logins can take minutes while the Pi is at its thermal limit. Wrap polls
    in `timeout 90` (Git Bash) and don't add load with extra SSH sessions.
  - The journal is not persistent (`journalctl -b -1` finds nothing after a reboot), so a crash leaves no log.
- The Pi now holds `models/yolo26n_320_scale0.9`, `models/yolo26n_416`, `models/yolo26n_640`, `images/compare/` and
  `results/` (~35 MB in total); 2.5 GB free.
