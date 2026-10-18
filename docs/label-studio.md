# Label Studio with Docker — quick reference

How to run [Label Studio](https://labelstud.io/) in Docker on Windows, serve images from disk, and set up a labelling
project with model pre-labels. Sections 1–3 and 6–7 work for any project; sections 4–5 are this project's routine (Pi
captures → Label Studio → VIA → YOLO). Set up and checked on 2026-10-18 with Label Studio 1.23.0.

## 1. Run the container

One folder on the PC holds everything Label Studio keeps: its database (`label_studio.sqlite3`, with the account,
projects, tasks and annotations), uploads, exports, and the images it serves. It is mounted into the container as
`/label-studio/data`. This project uses `%USERPROFILE%\captures`; another project can use its own folder (and its own
container name and port).

```powershell
docker run -d --name label-studio -p 8080:8080 `
  -v ${env:USERPROFILE}\captures:/label-studio/data `
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true `
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data `
  heartexlabs/label-studio:latest
```

Then open `http://localhost:8080` and sign up (a local account, stored in the data folder).

- **The document root must be a path inside the container** (`/label-studio/data`). A Windows path such as
  `F:\...\data` doesn't exist in the Linux container: images don't load and a local-files storage can't be added.
  The container also can't see any folder that isn't mounted, so the images must live inside the data folder.
- Day to day: `docker stop label-studio` / `docker start label-studio`. The container has no restart policy, so after a
  PC reboot run `docker start label-studio` (or add `--restart unless-stopped` to the `docker run` above).
- **Changing a setting** (an environment variable, the port, the mount) means re-creating the container:
  `docker rm -f label-studio`, then the `docker run` above. Nothing is lost: all data is in the mounted folder.
- Update: `docker pull heartexlabs/label-studio:latest`, then re-create. To stay on one version, use a tag such as
  `heartexlabs/label-studio:1.23.0` instead of `latest`.
- Backup: copy the data folder while the container is stopped.

## 2. Serve images from disk (local files)

1. Put the images inside the data folder, e.g. `%USERPROFILE%\captures\pi-camera\captures\<day>\<name>.jpg`.
2. In a task, an image is referenced by its path relative to the document root:
   `/data/local-files/?d=pi-camera/captures/<day>/<name>.jpg`.
3. The project needs a **Local files** source storage that covers those files (section 3). It only grants Label Studio
   permission to serve them.
4. Check what the container sees: `docker exec label-studio ls /label-studio/data/pi-camera/captures`.

## 3. Create and configure a project

1. **Create Project**, give it a name, skip the data import.
2. **Settings → Labeling Interface → Code**: paste the labelling config, before any import (pre-labels must match its
   names). This project: [tools/labelstudio_config.xml](../tools/labelstudio_config.xml), one `Image` named `image`
   and `RectangleLabels` named `label` with Blacky (hotkey 1) and Niche (hotkey 2).
3. **Settings → Cloud Storage → Add Source Storage → Local files**: absolute path inside the container, e.g.
   `/label-studio/data/pi-camera/captures`. *Check Connection*, *Add Storage*, but **don't Sync** when the tasks come
   from a JSON file: Sync adds every image again as a task without pre-labels. (Without pre-labels, Sync is the easy
   way to create one task per image.)
4. **Import → Upload Files**: choose the task file (JSON), then **click Import** at the top right. Uploading alone
   doesn't create the tasks.
5. **Settings → Annotation**: turn on *Use predictions to prelabel tasks* and select the model version. The list only
   shows model versions of predictions already in the project, so it stays empty ("No model or predictions
   available") until step 4 is done.
6. Optional: in the Data Manager, show the task fields as columns and filter on them (this project: `day`, `session`,
   `source`, `lux`).

### Task file format (with pre-labels)

A JSON list, one task per image. Coordinates are percentages of the image size (x, y = top-left corner);
`from_name` / `to_name` must match the labelling config. One task of this project's first batch (other `data` fields
left out):

```json
[{
  "data": {"image": "/data/local-files/?d=pi-camera/captures/2026-10-15/20261015-163424-653_button.jpg",
           "day": "2026-10-15"},
  "predictions": [{
    "model_version": "yolo26n_320_scale0.9 on the Pi (score >= 0.25)",
    "score": 0.963,
    "result": [{
      "id": "20261015-163424-653_button-0", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
      "original_width": 640, "original_height": 480, "image_rotation": 0,
      "value": {"x": 46.8, "y": 32.4, "width": 24.6, "height": 28.4, "rotation": 0, "rectanglelabels": ["Blacky"]},
      "score": 0.963
    }]
  }]
}]
```

## 4. This project: a new batch of Pi captures

In Claude Code: **`/labelstudio-import`** does all of this. By hand, from the repository root on the PC (the Pi must
be on; the app on it is the `cats-app` service):

```powershell
python scripts/pull_captures.py                    # new captures from the Pi -> data/pi-camera/captures/<day>/
python tools/captures_to_labelstudio.py --new-only $env:USERPROFILE/captures/label_studio.sqlite3 `
    --copy-to $env:USERPROFILE/captures
```

The second command reads Label Studio's database (read-only) to leave out every picture that is already a task, writes
the task file `data/pi-camera/labelstudio/<days>.json` with the Pi's detections as pre-labels, and copies the images
into the Label Studio folder under the same relative path. Import that file (section 3, step 4), and only that one:
older files in the folder are earlier batches. (`--days <YYYY-MM-DD ...>` instead of `--new-only` selects whole days.)

**Labelling rules**: one tight box per visible cat (also partly hidden or far away ones); **submit every image you
want in the dataset, including empty scenes** (submitted without boxes = a background image for YOLO); **skip** images
that should not be used (blurred, pointing at a wall, people).

## 5. This project: export and convert

In Claude Code: **`/labelstudio-export`** does steps 2–3 and the checks. By hand:

1. **Export → JSON** (the whole project each time) and save it in `data/annotations/` (git-ignored: the export holds
   user ids).
2. `python tools/labelstudio_to_via.py data/annotations/<export>.json` writes
   `data/pi-camera/captures/cats-annotations.json` (VIA 2.0.8 layout, `rect` regions in pixels, only submitted tasks,
   no user data). Commit that file.
3. Check: `python tools/label_report.py --export data/annotations/<export>.json` prints the counts per day and
   train/val/test split, how the review changed the pre-labels and how the deployed model scores against the labels,
   and draws every labelled image with its boxes on contact sheets in `runs/labels/<date>/` to look at.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Images don't load (broken image icon) | Document root is not a container path, the storage is missing, or the file isn't in the data folder. Check with `docker exec label-studio ls /label-studio/data/<path>`. |
| "Absolute local path must start with LOCAL_FILES_DOCUMENT_ROOT" when adding the storage | The storage path is outside the document root, or the root is a Windows path. Re-create the container (section 1). |
| "No model or predictions available" under Settings → Annotation | No predictions in the project yet: import the task file and click **Import**. |
| Pre-labels missing on tasks | `from_name` / `to_name` in the task file don't match the labelling config, or the config was set after the import. |
| Every image twice | The storage was synced after importing the task file, or a task file was imported twice. Delete the extra tasks in the Data Manager. |
| Label Studio not reachable after a reboot | `docker start label-studio`. |

## 7. Reusing this in another project

- Pick a data folder (and, if Label Studio should run next to this one, another container name and port such as
  `-p 8081:8080`). One container can also hold several projects.
- Put the images in a subfolder of the data folder, write the labelling config, and either Sync a Local files storage
  (no pre-labels) or generate a task file with predictions as in section 3.
- `tools/captures_to_labelstudio.py` and `tools/labelstudio_to_via.py` show both directions (task file with
  pre-labels, export → another format) and are small enough to adapt.
