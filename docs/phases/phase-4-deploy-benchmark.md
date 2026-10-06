# Phase 4 — Deploy to the Pi and baseline benchmark

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-06 |
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
- [x] 2026-10-03, after the user upgraded the cooling and the supply: probe 320 / 4 threads (still under-voltage,
      stopped by the watcher after 30 s), then rerun the three 10-minute 2-thread runs (no thermal capping).
- [x] 2026-10-06, after the user switched to a 5.1 V adapter: ramp at 320, with 2 threads for 60 s, then 4 threads
      for 60 s, then 4 threads for 180 s. The first two runs were clean, but the 3-minute 4-thread run ended in a
      **reset** (see Results).
- [x] 2026-10-06: under-voltage stop changed to ≥ 10 s or ≥ 3 dips within 60 s (benchmark, app, `pi/watch.sh`),
      tested with a fake `vcgencmd`. Then a live app test at `--threads 2` + 1 viewer (3 + 10 min) had no
      under-voltage at all.

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
starting; the camera is opened afterwards), `--sample-every 2` (temperature / clock / throttle samples; 5 before
2026-10-06), `--width/--height` and `--sensor-size` for the camera, `--stop-on-undervoltage 10`, `--stop-on-dips 3`
and `--undervoltage-window 60` (see below). Long runs must
survive an SSH disconnect: start them with `(setsid nohup ... > log 2>&1 < /dev/null &)` (see the handover notes).

**Under-voltage stop (default on).** The run ends early when the last `--undervoltage-window` seconds (default 60)
hold either of these:
- `--stop-on-undervoltage` seconds of under-voltage in total (default 10; each under-voltage sample counts for the
  time since the previous one);
- `--stop-on-dips` separate dips (default 3; a dip is a run of consecutive under-voltage samples).

The summary and the JSON are still written, with `"stopped_early"` set, and the exit code is 2. Pass `0` to turn a
limit off (only for deliberate under-voltage tests).
- History: added on 2026-10-03 after the brownout, as "30 s of continuous under-voltage". On 2026-10-06 that rule
  missed a reset that came after 2–6 s dips, so it became the window rule above. The rule is `UndervoltageGuard`
  in `pi/benchmark.py`, and `pi/app.py` uses the same class.
- Replayed on every phase 4 / 2026-10-06 watcher log, the window rule stops the 2026-10-06 run at 18:39:42, ~70 s
  before the reset. It stops the old supply's 4-thread probes after ~20 s instead of 30 s, and it doesn't stop any
  run that went fine (the 10-minute 2-thread runs, the phase 5 soak).
- Tested on the Pi with a fake `vcgencmd` that reports under-voltage on demand (`runs/pi/psu51/guardtest.sh`). The
  benchmark stopped at the 3rd dip and after 10 s of one long dip, the app at the 3rd dip, and `pi/watch.sh` stopped
  a benchmark whose own stop was off.

**Watcher: [pi/watch.sh](../../pi/watch.sh)** (in git since 2026-10-06; before that it was `runs/pi/phase4b/watch.sh`).
- Every 2 s it appends uptime, temperature, clock, throttle flags and load to a log synced to disk, which survives
  a reset.
- Independently of the programs' own stop, it applies the same rule (10 s or 3 dips in 60 s) and sends SIGTERM to
  `pi/benchmark.py` and `pi/app.py`.
- Usage: `bash pi/watch.sh <log> [max seconds] [max dips] &` next to the job, then `kill` it afterwards.

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
- Expected order of magnitude (estimate, to be measured): ~2 FPS at 640, more at 320. Measured: ~1.2 FPS at 640,
  ~4.4 FPS at 320 (2 threads, after the cooling upgrade).

## Results

Full tables in [docs/results.md](../results.md#phase-4--deployment-to-the-pi-and-baseline-benchmark-2026-10-02).

- **Pi = PC.** On 12 images (8 validation frames incl. the hard cases + 4 ×0.25 frames), confidence 0.25: 20/20
  boxes paired, IoU 1.0000, identical scores and corners to the 3 decimals of the JSON files.
- **Speed (2 threads, camera 640×480, 10 minutes):**
  - After the cooling upgrade (2026-10-03): **320 → 4.39 FPS**, 416 → 2.70 FPS, 640 → 1.18 FPS, flat for the
    whole run. At 320: inference 211 ms, capture ~7.5 ms, preprocess ~7.7 ms, postprocess ~1.4 ms.
  - Before, without a heatsink: 4.01 / 2.39 / 1.08 FPS, with the latency growing 10–20% once the CPU hit 80 °C.
  - Inference time scales with the number of pixels.
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
  - **2026-10-03:** the user upgraded the cooling and changed the supply (USB-C charger, 5 V 3.6 A 18 W, through a
    USB-C → micro-USB adapter).
    - Cooling is solved: idle 39 °C, ~70 °C after 10 minutes at 2 threads, no thermal cap.
    - Under-voltage is not solved: 4 threads still trigger it within seconds, with or without the camera, and
      there was one dip during boot. 2 threads is clean, apart from 4 short dips in the 320 run.
  - **2026-10-06:** the user switched to a 5.1 V adapter. It is better, but 4 threads is still not safe.
    - Better: the boot was clean, 2 threads and 4 threads each ran 60 s with no dip, and 4 threads gave 4.66 FPS
      vs 4.39.
    - Not safe: in a 3-minute 4-thread run, 2–6 s dips started after ~1 min of load, and the Pi **reset** ~1.5 min
      later.
    - Neither 30-s guard fired, because each dip was short. Temperature peaked at 71 °C, so heat is not the issue.
    - The **live app at `--threads 2` with a viewer** (~2.7 busy cores) ran 3 + 10 minutes with **no under-voltage
      at all**, at 1200 MHz, 4.0 detections/s, max 74.7 °C. The old supply failed this within seconds.
- **Decision:** stay at input 320 (`yolo26n_320_scale0.9`). A 416 + `scale` 0.9 model is not worth training now.
  Revisit only if Pi-camera images (phase 6) show missed far-away cats *and* the Pi's supply holds 4 threads (the
  cooling is now good enough).

## Handover notes

- **Hardware.** The cooling is fixed (2026-10-03). The 5.1 V adapter of 2026-10-06 holds the live app at 2 threads
  with a viewer (no dip in 13 min). It can't hold 4 threads: dips start after ~1 min, then the Pi resets. So the
  benchmark stays at 2 threads, and the live app is back at 2 threads (user decision, 2026-10-06; see phase 6).
  - **The old 30-s continuous under-voltage stop did not protect against the 2026-10-06 pattern** (short 2–6 s
    dips, then a brownout). Since 2026-10-06 `pi/benchmark.py`, `pi/app.py` and `pi/watch.sh` stop at ≥ 10 s of
    under-voltage or ≥ 3 dips within 60 s (see "Under-voltage stop" above).
  - Supply history: from 2026-10-03 it was a USB-C charger (5 V, 3.6 A, 18 W) feeding the Pi through a USB-C →
    micro-USB adapter. On 2026-10-06 the user changed it to a **5.1 V 3 A (15.3 W) adapter, still through the
    USB-C → micro-USB adapter**. Raising the voltage helped but was not enough.
    - The current rating is not the problem: a Pi 3B needs ≤ 2.5 A.
    - Remaining suspects: the USB-C → micro-USB adapter's contact resistance, the cable, and Wi-Fi current peaks.
    - The usual fix is a 5.1 V / 2.5 A micro-USB supply with an attached cable, plugged straight into the Pi (no
      adapter).
  - After a power fix, rerun 320 / 4 threads with the new guard: 60 s, then 3 minutes (2026-10-06 reset at
    ~2.6 min), then 10 minutes. 4 threads gave only +6% FPS at full clock (4.66 vs 4.39), so the main gain from a
    power fix is room for the live app at 2 threads, not 4-thread speed.
  - The `get_throttled` "since boot" bits are sticky, so judge each run by the per-sample `throttled` values in the
    JSON timeline or the watcher log (bit 0 = under-voltage now, bit 1 = frequency capped now).
  - **Control temperatures and resets in load tests** (the user's request):
    - Ramp up: a short probe before any 10-minute run.
    - `pi/benchmark.py` and `pi/app.py` stop by themselves at ≥ 10 s of under-voltage or ≥ 3 dips within 60 s
      (see above). Exit code 2 means the run was stopped.
    - For a log that survives a reset, also run the watcher [pi/watch.sh](../../pi/watch.sh), which applies the
      same stop. `runs/pi/psu51/live.sh` on the PC (app + `pi/soak.py`, not in git) shows how to use it.
    - From the PC, compare the boot time (`uptime -s`) between polls to detect a reset.
- **For phase 5 (`pi/app.py`):**
  - Default to **`--threads 2`**. The app also captures, draws and JPEG-encodes. If that keeps a third core busy,
    the Pi may hit the under-voltage cap even with 2 ncnn threads, so put temperature, ARM clock and the throttle
    flags in `/stats`. `pi/benchmark.py` has the helpers (`cpu_temp`, `arm_clock_mhz`, `throttled`) and can be
    imported.
  - Expect ~4.4 detections/s at 320 (2 threads, new cooling). Capture (picamera2 `capture_array`, 640×480) costs
    ~8 ms.
- **Deploy:** `scripts/deploy.py [models...] [--images ... --set name]`.
  - It replaces `pi/`, `models/<name>/` and `images/<set>/` on the Pi as a whole. Model names are the training run
    names, so the app should take `--model models/yolo26n_320_scale0.9`.
  - Pi results go in `~/cats-localization-v2/results/`; copy them to `runs/pi/` (git-ignored) with `pi_remote.py get`.
- **Long jobs on the Pi:**
  - Start them with `(setsid nohup bash job.sh > job.log 2>&1 < /dev/null &)`. Without the subshell and
    redirections, `pi_remote.py run` keeps the SSH channel open until the job ends.
  - `pi_remote.py run` has no timeout. Logins took minutes while the Pi sat at its thermal limit (before the
    cooling upgrade). Wrap polls in `timeout 90` (Git Bash) and don't add load with extra SSH sessions.
  - The journal is not persistent (`journalctl -b -1` finds nothing after a reboot), so a crash leaves no log.
- The Pi now holds `models/yolo26n_320_scale0.9`, `models/yolo26n_416`, `models/yolo26n_640`, `images/compare/` and
  `results/` (`phase4/` = 2026-10-02 runs, `phase4b/` = 2026-10-03 reruns, `psu51/` = 2026-10-06 adapter + live tests, `guardtest/` = stop tests; ~36 MB in total); 2.5 GB free. Both
  result folders are also in `runs/pi/` on the PC.
