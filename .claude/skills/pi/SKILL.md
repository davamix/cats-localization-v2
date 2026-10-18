---
name: pi
description: Operate the project's Raspberry Pi 3B (cats-localization-v2) safely - status (reachable, boot time and resets, the cats-app service, temperature, under-voltage, disk, captures), start / stop / restart the capture app, deploy code, read the app's log, run commands or copy files. Use it for anything that touches the Pi - "is the Pi on", "check the raspberry", "restart the app", "deploy", "the LED keeps blinking", "no pictures were saved", and before benchmarks or camera / button / LED tests - even when the user only says "the raspberry".
---

# Raspberry Pi operations

## Facts

- Raspberry Pi 3 Model B (1 GB RAM, Raspberry Pi OS Lite 64-bit) at `192.168.2.112` over Wi-Fi, user `pi`. Access by
  password SSH through `scripts/pi_remote.py`, credentials in the untracked `pi.env` (never print or commit it).
  Project folder `~/cats-localization-v2`: `.venv/`, `pi/` (code), `models/`, `captures/`, `results/`.
- The app `pi/app.py` runs as the systemd service **`cats-app`**, enabled at boot (installed from `pi/system/`). Log:
  `~/cats-localization-v2/results/phase6/service.log` (the journal is lost at every power-off). Live page and JSON:
  `http://192.168.2.112:8000/` and `/stats`.
- Button on GPIO 25 (short press = capture, hold 5 s = clean power-off); red LED on GPIO 24: slow blink = starting or
  waiting for the NTP clock sync, off = ready, on 1 s = picture saved, 3 quick blinks = refused, fast blink =
  powering off or stopping on an error.
- The user powers the Pi off and on to move it between spots, so a new boot time is normal: ask before treating it
  as a crash. Right after a power-on the clock is behind until NTP syncs (the service's start time can show an old
  date); the app takes no pictures until then.

## Commands (Git Bash, repository root; always inside `timeout`, `pi_remote.py` has none)

- **Status** (one SSH login + one HTTP request): `timeout 90 .venv/Scripts/python scripts/pi_status.py`
  (`--log 40` for more log lines). It decodes the `get_throttled` flags.
- **Any command**: `PYTHONIOENCODING=utf-8 timeout 90 .venv/Scripts/python scripts/pi_remote.py run '<command>'`;
  `run --sudo` for root (the password goes in on stdin). `PYTHONIOENCODING` keeps non-ASCII output (`°C`, `→`)
  from crashing the helper on the Windows console. Use single quotes around the remote command in Git Bash.
- **Service**: `... run --sudo 'systemctl stop cats-app'` (also `start`, `restart`, `status --no-pager`).
- **Log**: `... run 'grep -v " INFO " ~/cats-localization-v2/results/phase6/service.log | tail -n 40'`
- **Deploy code** (replaces `pi/` on the Pi; models with `scripts/deploy.py <run names or folders>`):
  `timeout 120 .venv/Scripts/python scripts/deploy.py`, then restart the service (the script reminds you).
- **Files**: `scripts/pi_remote.py get|put` with quoted `'~/...'` remote paths; `MSYS_NO_PATHCONV=1` for absolute ones.
- **Captures to the PC**: `scripts/pull_captures.py` (or `/labelstudio-import` to also prepare Label Studio).

## Rules and why

- **Ask the user first** before installing anything (apt, pip), deleting outside `~/cats-localization-v2`, or
  changing system configuration: systemd units (re-running `pi/system/install.sh` counts), sudoers, `config.txt`,
  device-tree overlays, the time zone.
- Nothing on the Pi may import torch or ultralytics: 1 GB RAM, the detector uses ncnn + numpy only.
- The service owns the camera, port 8000 and GPIO 24/25. **Stop it** before `pi/benchmark.py`, `pi/camera_test.py`,
  `pi/button_test.py` or a manual `pi/app.py`, and start it again afterwards. A manual app is stopped with
  `pkill -TERM -f "[p]i/app.py"`: without the brackets the pattern matches the SSH shell running `pkill` and kills it.
- **Power is the weak spot** (a 5.1 V adapter through a USB-C → micro-USB adapter; brownout resets happened under
  load). Load tests ramp up, run detached as `(setsid nohup ... > log 2>&1 < /dev/null &)`, are watched through
  `/stats` instead of repeated SSH logins, and are guarded: the app, `pi/benchmark.py` and `pi/watch.sh` stop at
  ≥ 10 s of under-voltage or ≥ 3 dips within 60 s. **If the Pi resets during a test, stop, report it and don't
  restart the load.** Details: handover notes in `docs/phases/phase-4-deploy-benchmark.md` and phase 5.
- `get_throttled` bits: 0 under-voltage now, 1 ARM frequency capped now, 2 throttled now, 3 soft temperature limit
  now; bits 16-19 the same since boot.

## Troubleshooting

| Symptom | Check |
|---|---|
| Not reachable | Probably switched off or moving: ask the user. |
| LED blinks slowly for minutes | No NTP sync (no Wi-Fi at that spot?) or the detector didn't start: `pi_status.py` (clock / ready) and the log. |
| Service inactive, log ends with exit code 2 | It stopped itself on under-voltage and is not restarted on purpose. Check the power, then `systemctl start cats-app`. |
| Button does nothing | Service stopped or not ready; LED state; the log shows `capture (button)` lines. |
| Camera busy / port 8000 in use | The service is running: stop it first. |
