# Phase 5 — Live stream web app on the Pi

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-03 |
| **Depends on** | Phase 4 |

## Goal

Watch the camera in the browser with live detection boxes: `http://<pi-ip>:8000`.
Make it work correctly first; speed is phase 7.

## Design (as built: [pi/app.py](../../pi/app.py))

A slow detector must not freeze the video, so capture, detection and streaming share only the latest state (one
`threading.Condition`) and never wait for each other:

1. **Capture thread**: picamera2 with the full-field-of-view sensor mode (1640×1232), ISP-scaled `RGB888` main
   stream (640×480; BGR in memory, which OpenCV and the detector expect). The camera runs at `--stream-fps`
   (`FrameDurationLimits`), so the stream shows every captured frame. Keeps the latest frame.
2. **Detector process + detection thread**: ncnn holds Python's GIL while it runs (~210 ms on the Pi), which would
   stall every thread of the app. So `pi/detector.py` runs in a **child process** (`multiprocessing`, spawn). The
   detection thread sends it the newest frame over a pipe, waits for `(detections, stage times)` and keeps the
   latest detections. The child ignores SIGINT and exits when the pipe closes, so it can't outlive the app.
3. **Render thread**: only while someone watches the stream, it draws the latest boxes, `Name 0.87` labels and
   two status lines (camera/detection FPS, inference ms, °C, MHz, plus a red `UNDER-VOLTAGE` / `ARM FREQUENCY
   CAPPED` warning) on each new frame, then JPEG-encodes it. Each frame is encoded **once** for all viewers.
4. **Monitor thread**: every 2 s it reads the CPU temperature, ARM clock, `get_throttled` flags (helpers from
   `pi/benchmark.py`) and the RSS of both processes; it logs a status line every minute. It stops the app with
   exit code 2 when the last `--undervoltage-window` s (60) hold `--stop-on-undervoltage` s of under-voltage in
   total (10) or `--stop-on-dips` separate dips (3). The rule is `UndervoltageGuard` in `pi/benchmark.py`. Until
   2026-10-06 the app only stopped after 30 s of *continuous* under-voltage, which missed a reset that came after
   short dips (phase 4).
5. **HTTP**: standard library `ThreadingHTTPServer`:
   - `/`: page with the stream and a stats table that polls `/stats` every second.
   - `/stream.mjpg`: `multipart/x-mixed-replace` MJPEG. A slow viewer always gets the newest JPEG and skips the
     rest; a viewer that stops reading is dropped after 10 s.
   - `/snapshot.jpg`: one annotated frame.
   - `/stats`: JSON with `config`; `capture` (FPS, frames); `detection` (FPS, latest and 10-s mean ms of
     preprocess / infer / postprocess / roundtrip / latency); `stream` (viewers, FPS, render ms); `detections`
     (`class`, `score`, `box` in frame pixels); `system` (temperature, ARM MHz, throttled raw / now / since boot,
     `undervoltage_window` = seconds and dips of under-voltage in the last 60 s and the stop rule, under-voltage
     and capped sample counts); `memory` (RSS / peak of both processes,
     MemAvailable); `pids`.

Options: `--model` (default `models/yolo26n_320_scale0.9`), `--conf` (0.5), `--threads` (2), `--port` (8000),
`--width/--height` (640×480), `--stream-fps` (15), `--jpeg-quality` (80), `--stop-on-undervoltage` (10 s),
`--stop-on-dips` (3), `--undervoltage-window` (60 s).
Ctrl+C or SIGTERM: stop the server, join the threads, send the child a stop message (terminate it if it hangs), stop
and close the camera. Exit codes: 0 normal stop, 1 a part failed, 2 too much under-voltage.

[pi/soak.py](../../pi/soak.py) samples a running app every 5 s (its `/stats` plus its own `/proc` readings: RSS and CPU
of every process of the app, system CPU, MemAvailable) into a JSON-lines file that is fsync'ed, so it survives a
reset. It then prints a summary that includes the memory trend in MiB/h. `--summarize` works on the PC too.

## Steps

- [x] `pi/app.py` with CLI arguments: `--model`, `--conf`, `--threads` (default 2, see phase 4), `--port` (default 8000),
      `--width/--height`, `--stream-fps`, `--jpeg-quality`, plus `--stop-on-undervoltage`.
- [x] Check the colour order from picamera2 (`RGB888` is BGR in memory) so boxes and colours are right and the
      detector gets the channel order it expects. Checked on stream frames: the wooden door is orange, the
      blanket blue.
- [x] Clean shutdown on Ctrl+C (stop camera, close server). Tested with SIGINT on the PC (fake camera, 0.5 s) and
      with SIGTERM on the Pi (1 s, exit code 0), plus the under-voltage stop (exit code 2).
- [x] Run it through `scripts/deploy.py` and test from the PC browser with both cats: both detected and labelled
      correctly when the camera points at them from close range; Blacky far away on the sofa was not detected.
- [x] Soak test, 35 min: no crash, no memory leak (+0.25 MiB/h).
- [ ] Optional: a `systemd` service so the app starts on boot. **Skipped for now** (user decision 2026-10-03:
      phases 6/7 need the camera and port 8000, and `--threads 1` is a temporary power workaround). A ready
      unit is in the handover notes.

## Done when

- The browser shows a smooth live stream with boxes that follow Blacky and Niche and are labelled correctly.
  **Yes at close range** (15 fps stream, boxes ~0.4 s behind the video). Far-away cats are phase 6.
- The app runs for at least 30 minutes without crashing or leaking memory. **Yes**: 35 min, RSS flat.

## Results (details in [results.md](../results.md#phase-5--live-stream-web-app-on-the-pi-2026-10-03))

- **Power decides the settings.** With 2 ncnn threads, as soon as someone watches the stream the Pi goes into
  continuous under-voltage (600 MHz), even with a 10 fps stream: ~2.5–2.7 cores busy, and this supply holds ~2.2.
  The app's 30-s stop caught it every time, with no reset. **On the current supply run `--threads 1`.** At 2.9
  detections/s it is faster than 2 threads at 600 MHz (2.6/s).
- **Soak, 35 min at `--threads 1 --stream-fps 15`, 2 viewers:**
  - camera 15.00 fps, stream 14.97 fps, 2.81 detections/s;
  - inference 335 ms, boxes 389 ms behind the video;
  - drawing + JPEG 11.5 ms per frame;
  - RSS app 176 + detector 181 + resource tracker 11 MiB, trend +0.25 MiB/h;
  - max 64 °C;
  - two 6-s under-voltage dips in the first minute, then 34 min clean at 1200 MHz.
- **Wi-Fi limits delivery.** The Pi rendered 15 fps the whole time, but the PC sometimes got 4–10 fps (~39 KB per
  frame = ~4.7 Mbit/s per viewer on the Pi's 2.4 GHz Wi-Fi).

## Handover notes

- **Run the app** (on the Pi, from `~/cats-localization-v2`, with the current supply):
  - Interactive: `.venv/bin/python pi/app.py --threads 1`. Ctrl+C stops it.
  - Detached: `(setsid nohup .venv/bin/python pi/app.py --threads 1 > results/app.log 2>&1 < /dev/null &)`.
    Stop it with `pkill -TERM -f "[p]i/app.py"`. (Corrected 2026-10-04: through `pi_remote.py run`, a plain
    `pkill -f pi/app.py` also matches the remote shell's own command line and kills it.)
  - Then open `http://192.168.2.112:8000/`. The app logs a status line every minute.
  - It takes ~13 s to start: spawning Python and loading ncnn and the model in the child.
- **Only one process can open the camera.** While the app runs, `pi/camera_test.py`, `pi/benchmark.py --camera` or a
  separate capture tool fail. Stop the app first, or (better for phase 6) build the capture into the app:
  - `/snapshot.jpg` is **annotated**; phase 6 needs **raw** frames for labelling. Add e.g. `/frame.jpg` (raw latest
    frame) or a "save frame" action that writes the full-quality frame to `captures/` on the Pi.
  - The phase 6 doc already plans "a snapshot button or endpoint in the web app".
- **Model gap seen in this phase (input for phase 6):**
  - Both cats are found from close range.
  - Blacky lying on the sofa back across the room (~90 px wide in the 640 px frame) got no box at confidence 0.5.
  - One sample had two Niche boxes (the known partial-Niche duplicate).
  - The user re-aimed the camera at the living room (sofa) during this session; it no longer points at the ceiling.
- **Power** (the supply is unchanged: USB-C charger + micro-USB adapter):
  - The limit is ~2.2 busy cores.
  - The app costs ~0.3 core without viewers (capture at 15 fps, pipe, monitor) and ~0.4 core with a 15 fps
    stream at full clock (drawing + JPEG 11.5 ms per frame).
  - **After a power fix**, probe the defaults (`--threads 2 --stream-fps 15`) for 3 min first, then soak them.
- **Guarded probes / soaks** (helpers in `runs/pi/phase5/` on the PC, not in git):
  - `probe.sh <name> <seconds> <app args...>`:
    - starts the app on the Pi, runs `pi/soak.py` for `<seconds>`, then stops the app with SIGTERM;
    - writes `results/phase5/<name>.{run.log,app.log,jsonl,soak.log}` on the Pi;
    - copy it to `~/cats-localization-v2/results/phase5/` first.
  - `stream_client.py <url> <seconds> <out.jsonl>` reads the MJPEG stream from the PC and logs the received FPS
    per 10 s.
  - `timeline.py <jsonl> [every]` prints a soak log compactly.
  - Watch the app with `curl http://192.168.2.112:8000/stats` from the PC rather than repeated SSH logins: every
    SSH login costs the Pi CPU, which matters near the power limit.
  - All phase 5 logs: `results/phase5/` on the Pi, copied to `runs/pi/phase5/pi/`.
  - Pi log times: the Pi's clock was correct and NTP-synchronised, but its time zone was Europe/London (UTC+1) while
    the PC is on UTC+2, so the phase 4/5 logs read one hour behind the PC. (Corrected 2026-10-04; since then the Pi is
    on Europe/Madrid, like the PC.)
- **Memory**:
  - app ~176 MiB, detector ~181 MiB, multiprocessing's resource tracker ~11 MiB, ~575 MiB still available.
  - Both Python processes import OpenCV, numpy and ncnn: spawn re-runs `app.py`'s imports in the child, and the
    parent imports ncnn through `benchmark.py` / `detector.py`. Only picamera2 is imported lazily (in the parent).
  - Trimming that is a phase 7 option, not needed now.
- **For phase 7**: cost of the app at 1.2 GHz:
  - ~19% of a core without viewers (capture at 15 fps + pipe + monitor);
  - drawing + JPEG 11.5 ms;
  - the pipe ~11 ms per detection (pickled 0.9 MB frame; shared memory would remove it);
  - Wi-Fi bandwidth (~4.7 Mbit/s per viewer at 15 fps, quality 80).
  - The hardware JPEG encoder + boxes drawn in the browser (phase 7 plan) would remove the drawing/JPEG cost.
- **systemd unit** (not installed; untested). `/etc/systemd/system/cats-app.service`, then
  `sudo systemctl daemon-reload && sudo systemctl enable --now cats-app`:

  ```ini
  [Unit]
  Description=Cats live stream (pi/app.py)
  After=network-online.target
  Wants=network-online.target

  [Service]
  User=pi
  WorkingDirectory=/home/pi/cats-localization-v2
  ExecStart=/home/pi/cats-localization-v2/.venv/bin/python pi/app.py --threads 1
  Restart=on-failure
  RestartSec=10
  # exit code 2 = stopped on sustained under-voltage: do not restart into the same brownout
  RestartPreventExitStatus=2
  # SIGTERM to the app only; it stops its detector process itself
  KillMode=mixed
  TimeoutStopSec=20

  [Install]
  WantedBy=multi-user.target
  ```
- **Next: phase 6** (real camera data). Start with the capture endpoint in the app, then collect frames at the
  real camera position(s), including far-away cats, both cats together and empty scenes.
