"""Turn Pi-camera captures and their pre-labels into a Label Studio task file (tasks with predictions).

    python tools/captures_to_labelstudio.py                                   # every day in data/pi-camera/captures
    python tools/captures_to_labelstudio.py --days 2026-10-04 2026-10-05      # only these days
    python tools/captures_to_labelstudio.py --copy-to C:/Users/<user>/captures   # + copy the images (Docker, below)
    python tools/captures_to_labelstudio.py --new-only C:/Users/<user>/captures/label_studio.sqlite3 \
        --copy-to C:/Users/<user>/captures    # only captures not in Label Studio yet (the usual batch)

Each capture from pi/app.py (data/pi-camera/captures/<day>/<name>.jpg + <name>.json, downloaded with
scripts/pull_captures.py) becomes one task:
    data         image (a Label Studio local-files URL), capture name, day, session, source (button / web / timer),
                 time, lux
    predictions  the pre-labels: the detections of the model on the Pi with a score >= --min-score, as editable boxes
                 (captures without detections get no prediction). Captures of test days (tools/pi_camera_splits.py)
                 never get pre-labels: they would flatter the model under test (phase 6 decision 2026-10-22).
The labeling config is tools/labelstudio_config.xml. Output (git-ignored): data/pi-camera/labelstudio/<name>.json, to
import in the project with Import > Upload files.

Label Studio reads the images from disk: the image URL is /data/local-files/?d=pi-camera/captures/<day>/<name>.jpg,
i.e. the image's path relative to --document-root (this repository's data/). Label Studio must run with
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<a folder holding pi-camera/captures/<day>/<name>.jpg>
and the project needs a "Local files" source storage for <that folder>/pi-camera/captures (no sync needed).
- Label Studio installed on the PC: the document root can be this repository's data/ folder itself.
- Label Studio in Docker (our setup since 2026-10-18) can't see the repository: --copy-to <the folder mounted as
  /label-studio/data> also copies the images there, under the same relative path (only missing files), and the
  container runs with LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data.
tools/labelstudio_to_via.py maps the URLs back to this repository's data/.

--new-only <label_studio.sqlite3> reads Label Studio's database (read-only) and leaves out every capture whose image is
already a task in any project, so a batch holds exactly the new pictures, also new ones from a day imported before.
Full steps: docs/label-studio.md.
"""
import argparse
import json
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from pi_camera_splits import CAPTURES, REPO_ROOT, capture_days, split_days

FROM_NAME, TO_NAME = "label", "image"  # as in tools/labelstudio_config.xml


def imported_images(database: Path) -> set[str]:
    """Image paths (the d= of local-files URLs) of every task in a Label Studio database, opened read-only."""
    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    try:
        paths = set()
        for (data,) in connection.execute("SELECT data FROM task"):
            image = json.loads(data).get("image", "")
            paths.update(parse_qs(urlparse(image).query).get("d", []))
        return paths
    finally:
        connection.close()


def capture_task(sidecar: Path, document_root: Path, min_score: float, prelabels: bool = True) -> dict:
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    image = sidecar.with_name(info["image"])
    if not image.is_file():
        raise FileNotFoundError(f"{sidecar}: image {image.name} is missing")
    width, height = info["width"], info["height"]
    relative = image.resolve().relative_to(document_root.resolve()).as_posix()
    task = {"data": {"image": f"/data/local-files/?d={quote(relative)}", "capture": image.stem,
                     "day": image.parent.name, "session": info.get("session"),  # no session before 2026-10-08
                     "source": info["source"], "time": info["time"],
                     "lux": round(info["camera"]["Lux"]) if "Lux" in info["camera"] else None}}
    results = []
    detections = info["detections"] if prelabels else []
    for i, detection in enumerate(d for d in detections if d["score"] >= min_score):
        x1, y1, x2, y2 = detection["box"]
        x1, x2 = max(x1, 0), min(x2, width)
        y1, y2 = max(y1, 0), min(y2, height)
        results.append({
            "id": f"{image.stem}-{i}", "from_name": FROM_NAME, "to_name": TO_NAME, "type": "rectanglelabels",
            "original_width": width, "original_height": height, "image_rotation": 0,
            "value": {"x": x1 / width * 100, "y": y1 / height * 100, "width": (x2 - x1) / width * 100,
                      "height": (y2 - y1) / height * 100, "rotation": 0, "rectanglelabels": [detection["class"]]},
            "score": detection["score"],
        })
    if results:
        model = info["model"]
        task["predictions"] = [{"model_version": f"{model['name']} on the Pi (score >= {min_score:g})",
                                "score": max(r["score"] for r in results), "result": results}]
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=CAPTURES, help="captures folder (day folders inside)")
    parser.add_argument("--days", nargs="+", metavar="YYYY-MM-DD", help="only these day folders (default: all)")
    parser.add_argument("--min-score", type=float, default=0.25, help="lowest pre-label score to import as a box")
    parser.add_argument("--document-root", type=Path, default=REPO_ROOT / "data",
                        help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT of the Label Studio server")
    parser.add_argument("--out", type=Path, help="task file (default: data/pi-camera/labelstudio/<days>.json)")
    parser.add_argument("--new-only", type=Path, metavar="DB",
                        help="leave out captures already in Label Studio; DB = its label_studio.sqlite3 (read-only)")
    parser.add_argument("--copy-to", type=Path, metavar="DIR",
                        help="also copy the images to DIR under their path relative to --document-root (missing "
                             "files only), for a Label Studio that can't read this repository, e.g. in Docker")
    args = parser.parse_args()

    days = args.days or sorted(p.name for p in args.captures.iterdir() if p.is_dir())
    sidecars = []
    for day in days:
        folder = args.captures / day
        if not folder.is_dir():
            parser.error(f"no day folder {folder}")
        sidecars += sorted(folder.glob("*.json"))
    if not sidecars:
        parser.error(f"no captures in {args.captures} for {', '.join(days) or 'any day'}")
    if args.new_only:
        if not args.new_only.is_file():
            parser.error(f"no Label Studio database at {args.new_only}")
        imported = imported_images(args.new_only)
        root = args.document_root.resolve()
        found = len(sidecars)
        sidecars = [s for s in sidecars if s.with_suffix(".jpg").resolve().relative_to(root).as_posix() not in imported]
        print(f"{found - len(sidecars)} of {found} captures are already in Label Studio")
        if not sidecars:
            print("nothing new to import")
            return
        days = sorted({s.parent.name for s in sidecars})

    test_days = {day for day, (split, _) in split_days(capture_days(args.captures)).items() if split == "test"}
    tasks = [capture_task(s, args.document_root, args.min_score, prelabels=s.parent.name not in test_days)
             for s in sidecars]
    if args.copy_to:
        copied = 0
        for sidecar in sidecars:
            image = sidecar.with_suffix(".jpg")
            target = args.copy_to / image.resolve().relative_to(args.document_root.resolve())
            if target.exists() and target.stat().st_size == image.stat().st_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            copied += 1
        print(f"copied {copied} of {len(sidecars)} images to {args.copy_to} (the others were there already)")
    name = days[0] if len(days) == 1 else f"{days[0]}_to_{days[-1]}"
    out = args.out or REPO_ROOT / "data" / "pi-camera" / "labelstudio" / f"{name}.json"
    if not args.out and out.exists():  # an earlier batch of the same days: keep it
        out = out.with_name(f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, indent=1) + "\n", encoding="utf-8", newline="\n")

    boxes = Counter(r["value"]["rectanglelabels"][0] for t in tasks for p in t.get("predictions", [])
                    for r in p["result"])
    with_boxes = sum("predictions" in t for t in tasks)
    on_test_days = sum(s.parent.name in test_days for s in sidecars)
    print(f"{len(tasks)} tasks from {len(days)} day(s), {with_boxes} with pre-labels "
          f"({', '.join(f'{k} {v}' for k, v in sorted(boxes.items())) or 'no boxes'})"
          + (f", {on_test_days} from test days without pre-labels" if on_test_days else "") + f" -> {out}")


if __name__ == "__main__":
    main()
