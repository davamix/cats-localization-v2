# Phase 6 — Real camera data and retraining

| | |
|---|---|
| **Status** | In progress: capture tooling (autostart, button, LED), download and labelling tools done; collection and labelling next |
| **Last updated** | 2026-10-08 |
| **Depends on** | Phase 5 |

## Goal

Improve accuracy where it matters: images from the Pi camera, at its real position and lighting. The 2020
dataset was taken with a different camera, has one cat per image and no empty scenes.

## Steps

- [x] **Push button** on GPIO 25 (BCM numbering = physical pin 22) to GND, no external resistor; internal pull-up,
      50 ms debounce. Tested with [pi/button_test.py](../../pi/button_test.py) (2026-10-04, see Results).
- [x] Capture tool on the Pi, inside the live app (it owns the camera): button, `POST /capture` / Capture button on
      the page, optional timer. Saves the raw frame + the model's detections as pre-labels
      ([pi/app.py](../../pi/app.py), design below). Tested on the Pi: 19 captures from all three sources.
- [x] **Collection mode** (2026-10-08): the app starts at boot (systemd service `cats-app`,
      [pi/system/](../../pi/system/)); a red **status LED** on GPIO 24 (physical pin 18, 220 Ω to GND) shows
      starting / ready / saved / refused; **holding the button 5 s powers the Pi off** cleanly; captures wait until the
      clock is NTP-synchronised; each capture records its session (one per power-on = camera spot). Timer off: all
      pictures by hand (user decision). Tested on the Pi including a power-off and re-plug.
- [x] Download the captures to the PC: [scripts/pull_captures.py](../../scripts/pull_captures.py) →
      `data/pi-camera/captures/<day>/` (not in git; the user decides what goes into the Drive zip).
- [x] Annotation tool: **Label Studio** (user decision 2026-10-04; 1.23 running locally on the PC at
      `http://localhost:8080`). Pre-labels import as editable predictions
      ([tools/captures_to_labelstudio.py](../../tools/captures_to_labelstudio.py), config
      [tools/labelstudio_config.xml](../../tools/labelstudio_config.xml)); the export converts to a VIA-format file
      that [tools/via_to_yolo.py](../../tools/via_to_yolo.py) reads ([tools/labelstudio_to_via.py](../../tools/labelstudio_to_via.py)).
      Tested with a simulated export; **not yet with a real one**.
- [x] Collection plan (below), including the held-out Pi-camera test set.
- [ ] Collect the images (plan below): **both cats** together, each alone near and far, **empty scenes** as
      negatives (YOLO uses images with empty label files as background), day / evening / lamp light, 2–3 camera
      positions.
- [ ] Label them in Label Studio (import, review, export, convert, commit `data/pi-camera/captures/cats-annotations.json`).
- [ ] Keep a held-out **Pi-camera test set** that is never used for training (split by whole days, see below).
- [ ] Build the combined dataset (2020 + Pi-camera train days; Pi-camera validation and test days).
- [ ] Retrain (phase 2 scripts), re-export and verify (phase 3), redeploy (phase 4), and compare against `v0.1.0`
      on the Pi-camera test set.
- [ ] Publish a new Release when the new model is better.

## Done when

- The Pi-camera test set exists and the new model beats `v0.1.0` on it.

## Capture design (as built 2026-10-04; collection mode 2026-10-08)

- **What is saved**: the 640×480 frame the detector sees (full-field-of-view sensor mode scaled by the ISP), as a
  JPEG at quality 95 (user decision; 29–107 KB). No higher-resolution still: that needs a camera mode switch, which
  stops the stream, and the detector only ever sees 640×480 (scaled to 320).
- **Which frame**: the frame of the latest *finished* detection, not the newest camera frame, so the image and its
  pre-labels match even when a cat moves. At `--threads 1` it is 0.36–0.74 s older than the live picture. No extra
  inference.
- **Pre-labels**: the detector process keeps every box with a score ≥ `--prelabel-conf` (0.25); the app filters
  them at `--conf` (0.5) for the stream and `/stats`. Boxes ≥ 0.5 are the same as before: NMS only removes a box
  for one with a higher score. So far-away cats with low scores still come as boxes to check.
- **Files** (on the Pi, `~/cats-localization-v2/captures/`; `scripts/deploy.py` never touches it):
  `captures/<YYYY-MM-DD>/<YYYYMMDD-HHMMSS-mmm>_<source>.jpg` + `.json` with the same name. Names are in Pi local
  time (Europe/Madrid, like the PC) and stay unique when flattened. The JSON holds `time` (ISO 8601 with UTC offset),
  `source` (`button` / `web` / `timer`), `width`/`height`, `frame_age_ms`, `camera` (ExposureTime µs,
  AnalogueGain, DigitalGain, Lux, ColourTemperature K, ColourGains: to sort by lighting), `model` (name, imgsz,
  threads, conf, prelabel_conf) and `detections` (`class`, `class_id`, `score`, `box` x1 y1 x2 y2 in frame pixels).
  Both files are written to a hidden `.<name>.part` file, synced to disk and renamed; the JSON after the image.
- **Requests**: the button callback (gpiozero's thread) only queues a request; a new **save thread** takes requests
  and the timer, applies the limits and writes the files. `POST /capture` waits for the result: 201 saved,
  429 too soon, 503 not ready (starting, or the clock not synchronised yet) / stopping, 507 disk limit, 504 no answer
  in 5 s.
- **Button**: with `--button-hold-s 5` (default) a short press takes the capture **on release**, so a long press
  doesn't also take one; holding it 5 s runs `sudo -n /usr/bin/systemctl poweroff` (allowed without a password by
  [pi/system/cats-poweroff.sudoers](../../pi/system/cats-poweroff.sudoers), nothing else), and systemd then stops the
  app with SIGTERM like any service. `--button-hold-s 0` = capture on press, no power-off.
- **Clock**: the Pi has no clock battery. After a power-on, systemd-timesyncd restores the time saved at the last
  shutdown, so the clock is behind by however long the Pi was off, until the first NTP sync over Wi-Fi (~75 s after
  boot in the 2026-10-08 test). A capture in that window would get a wrong date and land in the wrong day (and
  split). So captures start only when `/run/systemd/timesync/synchronized` exists (`--wait-for-clock`, default on;
  `--no-wait-for-clock` to skip). Without Wi-Fi the app never becomes ready.
- **Ready** = the first detection is done and the clock is synchronised (checked every 2 s by the monitor thread).
  At that moment the app starts the session: `session` in each capture's JSON is that time (ISO 8601), so captures
  from one power-on (one camera spot) can be grouped.
- **LED** (`--led-pin 24`, gpiozero `LED`, 0 = none): **slow blink** = starting or waiting for the clock; **off** =
  ready; **on 1 s** = capture saved (button or page); **3 quick blinks** = capture refused (too soon, not ready, disk
  limit, or a failed power-off); **fast blink** = powering off, or the app stopping on an error (5 s before it exits
  with code 1 or 2). It goes off when the app exits. Timer captures don't flash it.
- **Limits**: at most one capture per `--capture-min-interval` (1 s, any source); none when `captures/` holds more than
  `--capture-max-mb` (1000) or the card has less than `--capture-min-free-mb` (500 MB) free ("paused" on the page).
- **Timer**: `--capture-every MIN` (default 0 = off; starts when the app is ready). At 10 min that is 144 images/day,
  ~10 MB/day. Not used: the service runs without it (user decision 2026-10-08: all pictures by hand).
- **Page / stats**: a Capture button, "N this run; M on the Pi (MB), free MB", the last capture (time, source,
  pre-labels) with a thumbnail (`/captures/last.jpg`), rejected requests, button and LED state, timer, and a Ready
  row (session, or what it waits for). `/stats` has the same in `captures` (`ready`, `clock`, `session`, ...). The
  status line in the log ends with the capture count; log lines carry the date.
- **GPIO**: `--button-pin 25` (0 = no button), `--button-pull up` (button to GND), `--button-bounce-ms 50`,
  `--button-hold-s 5`, `--led-pin 24`. gpiozero is imported only in the app process (like picamera2). If a pin
  can't be opened, the app logs it and runs without the button / LED. Shutdown closes the button first and the LED
  last.
- **Autostart**: [pi/system/cats-app.service](../../pi/system/cats-app.service) runs
  `.venv/bin/python pi/app.py` (2 threads, no timer) as user `pi` from `~/cats-localization-v2` at boot. It appends
  the output to `results/phase6/service.log`, because the journal is lost at every power-off. It restarts the app
  10 s after a failure, but not after exit code 2 (under-voltage stop: no restart into a brownout), and gives up
  after 5 failed starts in 10 minutes. [pi/system/install.sh](../../pi/system/install.sh) installs the unit and the
  sudoers rule (`visudo` check first), enables and (re)starts the service; `--remove` undoes it.

## Collection workflow (with the button)

1. Put the Pi at the spot and plug it in. After ~30 s the LED **blinks slowly** (the app is starting, then waiting
   for the clock); when it goes **off**, the app is ready (~1–1.5 min after plugging in).
2. **Press** the button for each picture: the LED lights for **1 s** when it's saved. **3 quick blinks** = not saved
   (pressed within 1 s of the last picture, or not ready yet). The page `http://192.168.2.112:8000/` shows the
   live view to aim the camera, the last capture and the Ready state.
3. To move: **hold the button 5 s**. The LED blinks fast, then the Pi shuts down; unplug when its green light has
   stopped flashing (~10–20 s). Plug in at the new spot: a new session starts.
4. Every few days: `python scripts/pull_captures.py` on the PC, then label (below).

## Collection plan

Aim for a first round of **~500 labelled Pi-camera images: ~350 train, ~50 validation, ~100–120 test**. That is 3–4×
the 2020 dataset, enough to teach the model the real scenes without hours of labelling (with pre-labels, reviewing an
image takes seconds when the model is right).

| Category | Train | Test | How to get them |
|---|---|---|---|
| Blacky alone, **near** (box ≥ ~25% of the frame width) | 40 | 10 | button |
| Blacky alone, **far** (box < ~15% of the width, e.g. on the sofa back across the room) | 60 | 15 | button |
| Niche alone, near | 40 | 10 | button |
| Niche alone, far (incl. the cat-tree hammock, the floor across the room) | 60 | 15 | button |
| **Both cats** in the frame | 50 | 15 | button, whenever it happens (e.g. while they sleep together) |
| **Empty scenes** (no cat) | 80 | 25 | button: a few per spot and lighting, plus the hard negatives below |
| *(middle distances come on their own)* | | | |

- **Hard negatives** among the empty scenes: dark objects (the black leather sofa: "Blacky 0.66" on 2026-10-04;
  cushions with cat prints, cat beds, bags, coats, shoes), the beige blanket close-up ("Blacky 0.60"), and people
  without cats.
- **Lighting**: roughly a third each of **day** (daylight through the windows), **evening/twilight** (low lux, high
  gain), and **lamp light** (warm, colour temperature ~2700–3300 K). The JSON files record Lux, gain and colour
  temperature, so the mix can be checked before labelling. Don't bother with lights-off night: the camera has no IR.
- **Camera positions**: most images (≥ 60%) from the position(s) where the camera will live, plus 2–3 other
  positions/rooms so the model doesn't learn one background. Keep the camera roughly level, as when deployed. The
  2026-10-04 test captures were hand-held in several rooms after the Pi moved. Each power-on is one session in the
  JSON files, so the spots can be counted.
- **Pace**: all pictures by hand with the button (user decision 2026-10-08), several per session. Label every
  capture. The timer (`--capture-every`) stays available if empty scenes or lighting changes fall short; then label
  at most ~1 empty timer image per hour and spot (consecutive empty frames are near-duplicates).
- **Near-duplicates**: captures seconds apart of a still cat are almost the same image. In train, prefer ≥ 30 s or a
  visible change between images of the same scene.

### Held-out test set (and validation)

- **Split by whole capture days, decided by date before looking at the images**: the 3rd, 6th, 9th… collection day
  is a **test** day; one more day (e.g. the 2nd) is the **validation** day used for choosing `best.pt` and tuning the
  confidence threshold; all other days are **train**. Neighbouring frames therefore never end up in two splits.
- Write the day lists down (the next session puts them in a tracked file, e.g. `data/pi-camera/splits.json`) and
  never move an image between splits after seeing results. Never train on the test days, and never tune anything on
  them.
- Before training, check that the test days cover every category in the table (both cats, far cats, empty scenes,
  each lighting). If one is missing, capture it on a test day (a dedicated session) rather than borrowing from train
  days.
- 2026-10-04 (19 hand-held test captures) and 2026-10-08 (5 captures from the autostart test, plus whatever is
  collected later that day) count as **train** days and are not numbered: the 1st collection day is the next day
  with captures.
- Optional: one camera position that only appears on test days, to measure generalisation to a new view.

## Labelling workflow (Label Studio)

1. **Start Label Studio with local-file serving** (needed once per start; the images stay on disk). In PowerShell:

   ```powershell
   $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
   $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "F:\Development\cats-localization-v2\data"
   label-studio start
   ```

   Keep Label Studio's own data folder out of the repository (or in the git-ignored `/captures/` at the repo root,
   where the 2026-10-04 instance keeps it).
2. **Create one project** for all Pi-camera captures (e.g. "Cats Pi camera"):
   - Settings → Labeling Interface → Code: paste [tools/labelstudio_config.xml](../../tools/labelstudio_config.xml)
     (`1` = Blacky, `2` = Niche).
   - Settings → Cloud Storage → Add Source Storage → **Local files**, absolute path
     `F:\Development\cats-localization-v2\data\pi-camera\captures`. Check the connection, but **don't sync** (sync
     would add the images again as tasks without pre-labels).
   - Settings → Annotation: turn on using predictions to pre-label tasks, so each task opens with the model's boxes.
3. **Each batch**: `python scripts/pull_captures.py`, then
   `python tools/captures_to_labelstudio.py --days <YYYY-MM-DD ...>` and Import the file it writes
   (`data/pi-camera/labelstudio/<days>.json`). Import each day once.
4. **Review**: fix, add or delete boxes; one box per visible cat, tight around the visible part. **Submit every
   task you want in the dataset, including empty scenes** (submitted without boxes = background image). **Skip**
   images that should not be used (blurred, the camera pointing at a wall, privacy).
5. **Export** → JSON, then `python tools/labelstudio_to_via.py <export>.json`. It writes
   `data/pi-camera/captures/cats-annotations.json` (VIA 2.0.8 format with `rect` regions, only reviewed tasks, no user
   names or e-mails) — commit that file. `tools/via_to_yolo.py` reads it like the 2020 polygons; VIA keys images by
   name + size and the day folder is found from that.

## Results (2026-10-04 and 2026-10-08; details in [results.md](../results.md#phase-6--real-camera-data-2026-10-04))

- **Button**: no floating (0 edges in 15–30 s untouched with the pull-up), all 8 presses seen in both runs. Without
  debounce, 3 of 8 releases bounced (2–3 edges within 0.1–0.4 ms); with lgpio's 50 ms debounce, 16 clean edges
  for 8 presses. Presses last 127–381 ms.
- **Captures on the Pi** (`--threads 1 --capture-every 1`, 1 viewer, 8.6 min): 19 saved (button 10, web 2,
  timer 7; 2 refused as too soon), 29–107 KB each, frames 0.36–0.74 s old. Clean SIGTERM stop in 1 s, exit code 0,
  GPIO 25 released.
- **The model on the new scenes**: in the 19 hand-held captures, Niche was missed 3 times even at 0.25 (standing by
  a door, lying far away on the floor, in the cat-tree hammock); the black leather sofa seat next to Blacky (0.98,
  in the cat bed) got a second "Blacky 0.66" box, and a close-up of a beige blanket got "Blacky 0.60" over the whole
  frame. Exactly the gaps this phase targets.
- **Power after the move**: with the Pi in its new room, the same `--threads 1` app + 1 viewer had **repeated
  4–16 s under-voltage dips**: after 5 clean minutes, 14 dips in the last 4 minutes (600 MHz during each), where
  the phase 5 soak had 2 dips in 35 min. No sustained under-voltage, so the 30-s stop never triggered, and no reset.
- **2026-10-08, button + LED** (rewired: LED on GPIO 24 through 220 Ω; the user had first put the LED in series with
  the button, which can't work: ~0.03 mA through the pull-up, and the pin can't fall below the LED's forward
  voltage): 8 of 8 presses, 16 clean edges (shortest gap 102 ms), no edge while untouched, the LED lit on every press.
- **2026-10-08, collection mode on the Pi** (service, 5.1 V adapter, 2 threads, no viewer): the user ran the whole
  cycle and everything worked as designed: 4 presses saved (LED 1 s each), a second quick press refused (3 blinks),
  5-s hold → clean power-off (SIGTERM, app stopped with exit code 0, LED off), re-plug → autostart → slow blink →
  ready → 1 press saved. Boot → ready took ~75 s: the app was up ~48 s after boot, then waited ~27 s for the NTP
  sync; until then the clock was ~38 s behind (the time the Pi was off). The captures carry the right time and one `session` per power-on.
- **Temperature, no viewer, 2 threads**: 59.1 °C after 1 min, 68.8–69.8 °C from minute 6 to 11, 1200 MHz, no
  under-voltage, 4.3 detections/s (inference 213 ms). The detector runs all the time, even when nobody presses.

## Handover notes

- **The app now runs as the `cats-app` service and starts at boot** (installed and enabled 2026-10-08, user OK;
  `/etc/systemd/system/cats-app.service` + `/etc/sudoers.d/cats-poweroff`, both from [pi/system/](../../pi/system/)).
  - Status: `systemctl status cats-app`; log: `results/phase6/service.log` (dated lines, all runs appended).
  - **Before anything else that needs the camera, port 8000 or GPIO 24/25** (benchmarks, `pi/button_test.py`,
    `pi/camera_test.py`, a manual `pi/app.py`): `sudo systemctl stop cats-app`; afterwards `sudo systemctl start
    cats-app`. Through `pi_remote.py run --sudo`.
  - **After `scripts/deploy.py`**, the service keeps the old code until `sudo systemctl restart cats-app` (deploy.py
    prints a reminder); after changing a file in `pi/system/`, run `sudo bash pi/system/install.sh` again.
  - Undo: `sudo bash pi/system/install.sh --remove` (removes both files in /etc, ask the user first).
  - `pi_remote.py run` crashes on non-ASCII output (e.g. the `→` printed by `systemctl enable`) on the Windows
    console: set `PYTHONIOENCODING=utf-8`.
  - A manual run is still possible with the service stopped:
    `(setsid nohup .venv/bin/python pi/app.py > results/phase6/app.log 2>&1 < /dev/null &)`, stopped with
    `pkill -TERM -f "[p]i/app.py"` (the brackets stop the pattern from matching the remote shell itself).
- **The user collects by hand** (no timer) and moves the Pi between spots with the long-press power-off; see the
  collection workflow above. Each power-on is one `session`.
- **Power first**: check the power path in the new room before long timer runs (same charger + USB-C → micro-USB
  adapter? an extension lead or a longer cable now?). The dips started with a viewer connected; the stream costs
  ~0.4 core, so for long collection runs **close the page** when nobody watches. At ≥ 10 s of under-voltage or
  ≥ 3 dips within 60 s the app stops itself (exit code 2); if the Pi resets, stop and report.
- **2026-10-06: supply changed to a 5.1 V adapter** (phase 4 test, see
  [results.md](../results.md#51-v-adapter-check-2026-10-06)). It is better: the boot was clean, and 2 threads and
  4 threads each ran 60 s with no dip. But a 3-minute 4-thread run reset the Pi after short 2–6 s dips that the
  old 30-s stop did not catch.
  - The stop now trips at ≥ 10 s of under-voltage or ≥ 3 dips within 60 s (app, benchmark, `pi/watch.sh`).
  - **Live test, same day:** `--threads 2` + 1 viewer for 3 + 10 min had **no under-voltage at all**, 1200 MHz,
    4.0 detections/s (vs 2.8 at 1 thread), max 74.7 °C
    ([results](../results.md#51-v-adapter-check-2026-10-06)).
  - **Decided (user, 2026-10-06):** the collection runs with `--threads 2` (the code default).
- **Watch the temperature on long runs** (the user's open point for 2 threads).
  - The 10-minute live test reached 74.7 °C and was still rising slowly (+0.3 °C/min at the end), so it's not known
    yet where it settles. The firmware caps the clock at 80 °C.
  - 2026-10-08, no viewer: 68.8–69.8 °C after 6–11 min, flat at 1200 MHz. With a viewer (2026-10-06) 74.7 °C.
  - The app logs °C and MHz every minute in `results/phase6/service.log`, and `/stats` has them live. For a log that
    survives a reset, run `pi/soak.py` next to the app (sampling the app) or `pi/watch.sh`.
  - In the first long runs, check the peak temperature and any `ARM frequency capped` / clock < 1200 MHz samples.
  - If it gets close to 80 °C: close the stream page when nobody watches (~0.4 core), lower `--stream-fps`, improve
    the cooling, or go back to `--threads 1`.
  - Bigger lever (not built): the detector runs nonstop although captures need it only at a press. Detecting only
    while someone watches, plus once on demand at a press (~0.25 s), would leave the Pi almost idle between presses.
- **Disk**: 2.5 GB free on the Pi; a button capture is ~60–120 KB, so space is no issue without the timer. Nothing
  deletes captures on the Pi (neither the app nor `pull_captures.py`); delete old days by hand after pulling, if
  needed, inside `~/cats-localization-v2/captures/`.
- **Captures may show people** (the user's hand is in one 2026-10-04 image). They stay out of git (`data/**` is
  ignored except the `cats-annotations.json` files); the user decides what goes into the Drive zip.
- **lgpio** leaves a 0-byte FIFO `.lgd-nfy0` in the working directory (`~/cats-localization-v2`); harmless, it is
  reused at every start.
- **GPIO wiring** (2026-10-08): button GPIO 25 (pin 22) → GND (pin 20), no resistor; red LED GPIO 24 (pin 18) →
  220 Ω → LED → GND (pin 20). GPIO 24 is the only output.
- **Pi time zone** is Europe/Madrid since 2026-10-04 (user decision; it was Europe/London), so Pi logs and capture
  names match the PC's clock. The Pi was unplugged and moved to another room at ~10:26 that day (the boot time
  changed; not a fault).
- **Label Studio**: 1.23.0 on the PC (port 8080; the app is on port 8000 on the Pi, no conflict). The running instance
  was started without local-file serving, so restart it with the two environment variables above before importing.
  The import/export converters were tested with a simulated export only: **check the first real export** (box
  positions in `cats-annotations.json` vs the image, e.g. with `tools/via_to_yolo.py` + `tools/visualize_labels.py`).
- **PC tests of the app**: a stub `picamera2` (frames from validation images; can fail after N frames) and a stub
  `gpiozero` (a Button that presses and holds itself on a schedule, an LED that prints its changes) were used from
  the session scratchpad (not in git), with a fake clock-sync file and a harmless power-off command. They covered
  not-ready (clock / first detection), too-soon, disk limit, button failure, `--button-pin 0`, the LED states, the
  long press (success and failure) and the 5-s alarm on an error exit.
- **Next**: the user collects (plan above) with the button over ~1–2 weeks; label in batches, then build the
  combined dataset with the day-based splits, retrain and compare against `v0.1.0` on the Pi-camera test days.
