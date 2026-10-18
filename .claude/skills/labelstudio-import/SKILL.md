---
name: labelstudio-import
description: Bring new pictures from the Raspberry Pi into Label Studio for labelling (cats-localization-v2). Pulls the new captures from the Pi, picks the ones that are not in Label Studio yet, writes the import file with the Pi model's pre-labels, copies the images into Label Studio's Docker data folder and checks that the container sees them. Use it whenever the user took pictures, did a capture session, wants to label new pictures, asks to copy or bring Pi captures/photos into Label Studio, or asks for the next labelling batch, even if Label Studio isn't named.
---

# New Pi pictures → Label Studio

## What you need to know (no other document needed)

- The Pi app (`pi/app.py`, systemd service `cats-app`) saves each capture on the Pi as
  `~/cats-localization-v2/captures/<YYYY-MM-DD>/<YYYYMMDD-HHMMSS-mmm>_<source>.jpg` plus a `.json` with the time,
  the session (one per power-on) and the model's detections ≥ 0.25 (the pre-labels).
- `scripts/pull_captures.py` copies new captures to the PC copy `data/pi-camera/captures/<day>/` (git-ignored).
- Label Studio runs in Docker, container `label-studio`. Its data folder on the PC is mounted as `/label-studio/data`
  and is also the local-files document root, so images must sit inside it under `pi-camera/captures/<day>/` — the
  same relative path as in `data/`. Tasks reference them as `/data/local-files/?d=pi-camera/captures/<day>/<name>.jpg`.
- The project "cats-localization-v2" is already set up (labelling config, Local files storage, pre-labelling with
  the Pi model). Setup, re-creating the container and troubleshooting: `docs/label-studio.md`.

## Steps (Git Bash, repository root)

1. **Pull the new captures from the Pi:**
   `PYTHONIOENCODING=utf-8 timeout 300 .venv/Scripts/python scripts/pull_captures.py`
   It prints only at the end and normally takes a few seconds (plus ~1 s per new picture). If the Pi can't be
   reached, it is probably switched off (the user powers it off to move it between spots): say so and offer to
   continue with the pictures already on the PC. A stalled connection ends with a timeout error (once seen under
   load): just run it again.
2. **Find Label Studio's data folder** from the container (don't assume a path):
   `docker inspect label-studio --format '{{range .Mounts}}{{if eq .Destination "/label-studio/data"}}{{.Source}}{{end}}{{end}}'`
   If the container doesn't exist or isn't running (`docker ps`), ask the user to start it (`docker start
   label-studio`); never create or re-create containers on your own.
3. **Write the batch and copy the images** (`<folder>` = the path from step 2):
   `.venv/Scripts/python tools/captures_to_labelstudio.py --new-only "<folder>/label_studio.sqlite3" --copy-to "<folder>"`
   `--new-only` opens Label Studio's database read-only and leaves out every picture that is already a task, so a
   batch holds exactly the new pictures, including new ones from a day imported earlier. Never write to that
   database. If it prints "nothing new to import", tell the user and stop.
4. **Check the container sees every image:** the two counts must be equal.
   `MSYS_NO_PATHCONV=1 docker exec label-studio sh -c 'find /label-studio/data/pi-camera/captures -name "*.jpg" | wc -l'`
   `find data/pi-camera/captures -name "*.jpg" | wc -l`
5. **Tell the user**, briefly:
   - how many new pictures, from which days, how many came with pre-labels;
   - the split of the new days (`.venv/Scripts/python tools/pi_camera_splits.py`): if a new day is a val/test day,
     its pictures are held out from training;
   - the file to import, `data/pi-camera/labelstudio/<...>.json`: in the project, **Import → Upload Files →** choose
     the file **→ click Import** (top right; uploading alone creates no tasks). Import only the file this run wrote:
     older files in that folder are records of earlier batches;
   - labelling rules: one tight box per visible cat (1 = Blacky, the black cat; 2 = Niche, white with black
     patches); **submit** every picture wanted in the dataset, including empty scenes; **skip** unusable ones;
   - afterwards: export the whole project as JSON into `data/annotations/` and use `/labelstudio-export`.

## Don'ts and why

- Don't sync the Local files storage: Sync turns every image into a second task without pre-labels.
- Don't import a day's file and then the same pictures again: duplicates have to be deleted by hand in the Data
  Manager. `--new-only` prevents this as long as only the newest file is imported.
- Don't commit anything from this step: pictures, task files and exports are git-ignored on purpose (people may
  appear in pictures; exports hold user ids).
