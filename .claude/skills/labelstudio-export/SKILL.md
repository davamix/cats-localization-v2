---
name: labelstudio-export
description: Turn a Label Studio export of the cat pictures into the project's annotation file and check it (cats-localization-v2). Converts the JSON export in data/annotations/ to data/pi-camera/captures/cats-annotations.json (VIA format, read by tools/via_to_yolo.py), reports counts per day and train/val/test split, how the review changed the pre-labels and how the deployed model scores against the labels, and draws contact sheets to look at. Use it whenever the user finished labelling, exported annotations, saved a Label Studio JSON, or asks to convert, check or update the labels, even if VIA or the scripts aren't named.
---

# Label Studio export → annotation file + checks

## What you need to know

- The user labels Pi-camera pictures in Label Studio (Docker, project "cats-localization-v2"; reference
  `docs/label-studio.md`) and exports the **whole project** as JSON into `data/annotations/`. Exports are
  git-ignored because they hold user ids: never commit or quote them beyond counts.
- `tools/labelstudio_to_via.py` rebuilds `data/pi-camera/captures/cats-annotations.json` from the export: only
  submitted tasks (the newest annotation of each), `rect` regions in pixels, an empty region list = a reviewed empty
  scene (a YOLO background image). Skipped and unreviewed tasks are left out on purpose, and so is a box drawn on a
  skipped task. This file is tracked in git, like the 2020 VIA files.
- Splits are by whole capture day (`tools/pi_camera_splits.py`): never train or tune on test days.
- The pre-labels are the deployed model's (v0.1.0) detections on exactly these frames, made on the Pi, so comparing
  them with the labels measures the deployed model on Pi-camera images. The first batch (2026-10-18): 91 images,
  86 boxes; the model found 21% of the cats at 0.5 (6% on the test day) — see `docs/results.md`.

## Steps (Git Bash, repository root)

1. **Pick the export**: the file the user names, otherwise the newest `data/annotations/*.json` (`ls -t`).
2. **Convert**: `.venv/Scripts/python tools/labelstudio_to_via.py data/annotations/<export>.json`
   It stops on a rotated box, an unknown label or an image missing on the PC (then run `scripts/pull_captures.py`
   first, or `/labelstudio-import`). Report such errors to the user; don't patch the export.
3. **Report**: `.venv/Scripts/python tools/label_report.py --export data/annotations/<export>.json`
   Read it: there must be no `MISMATCH` line; note what val/test still lack (the `has no ... yet` lines) and the
   model's numbers per split.
4. **Look at every contact sheet** it lists (`runs/labels/<date>/sheet_*.jpg`, read them as images). Check that each
   box sits on a cat, the name fits (Blacky = black cat, orange box; Niche = white with black patches, blue box),
   and no visible cat lacks a box. Dark things are often mistaken for Blacky: the black leather sofa, the black
   cat-face cushion on the sofa back, the grey-and-black cat tunnel, bags. Before calling something wrong, look at
   the image at full size (crop and enlarge it with OpenCV). Don't change labels yourself: list the doubtful images
   by name so the user can fix them in Label Studio and export again.
5. **Summarise for the user**: images, empty scenes, both-cats images and boxes per split; what val/test lack; the
   model's numbers (especially the test split at 0.5) and how they compare with the earlier batch; doubtful images.
6. **Docs and git**: add the new counts to `docs/phases/phase-6-real-data.md` (step "Label them" and the handover
   notes) and, if the numbers matter, `docs/results.md`. `cats-annotations.json` is the file to commit; commit and push
   only when the user asks.
