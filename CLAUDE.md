# cats-localization-v2 — notes for Claude

Detect and identify the user's two cats, **Blacky** and **Niche**, in real time on a Raspberry Pi 3B with its camera
(YOLO26n exported to NCNN; the Pi runs ncnn + numpy only) and watch it as a live stream in the browser. Trained on the
PC (Windows, RTX 2080 Ti). Phase 6 (current) collects and labels Pi-camera pictures to retrain.

## Every session

- Start: read `docs/PLAN.md` (overview, decisions, environment, phase table) and the current phase document in
  `docs/phases/` (status, steps, **handover notes**). Don't re-derive what is written there.
- Before significant new work, show the user a short plan and wait for the OK.
- End: tick steps, set status and "last updated", write handover notes in the phase document, update the phase table
  in `docs/PLAN.md` and `docs/results.md` with anything measured. Commit and push only when the user asks.

## Skills (repeated procedures, so the docs don't have to be re-read)

- `/pi` — Raspberry Pi status, the `cats-app` service, deploy, logs, safe remote commands.
- `/labelstudio-import` — new Pi pictures → Label Studio batch with pre-labels.
- `/labelstudio-export` — Label Studio export → `data/pi-camera/captures/cats-annotations.json` + checks.
- Label Studio in Docker (setup, project, troubleshooting): `docs/label-studio.md`.

## Environment

- Python: `.venv/Scripts/python` from the repository root (Python 3.12; torch, Ultralytics, ncnn).
- Git Bash rewrites arguments that start with `/` (`MSYS_NO_PATHCONV=1`) and expands an unquoted `~` on the PC;
  PowerShell mangles nested quotes in `scripts/pi_remote.py run`; set `PYTHONIOENCODING=utf-8` for scripts that print
  remote output.
- The repository uses LF (`.gitattributes`): Python's `write_text` on Windows writes CRLF unless `newline="\n"`.
- Long inline Python in a shell heredoc can lose backslashes: put it in a scratch file and run that.

## Rules

- Never commit images, weights, exported models, run outputs or Label Studio exports: `data/**` is ignored except
  `README.md` and the `cats-annotations.json` files; `runs/`, `models/`, `datasets/`, `*.pt`, `*_ncnn_model/` too.
  `pi.env` holds the Pi password. The repository is public: no personal paths or account names in tracked files.
- Pi: ask before installing, deleting outside `~/cats-localization-v2` or changing system configuration; nothing on
  the Pi imports torch or ultralytics; load tests are guarded and stop on a reset (see `/pi`).
- Pi-camera data is split by whole capture day (`tools/pi_camera_splits.py`): never train or tune on test days.
