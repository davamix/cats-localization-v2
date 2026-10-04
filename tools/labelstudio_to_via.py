"""Convert a Label Studio JSON export of the Pi-camera captures into a VIA 2.0.8-style annotation file.

    python tools/labelstudio_to_via.py path/to/export.json
    -> data/pi-camera/captures/cats-annotations.json    (tracked in git, like data/{train,validation}/)

Export with Export > JSON in the Label Studio project (tasks imported with tools/captures_to_labelstudio.py, labeling
config tools/labelstudio_config.xml). The output is rebuilt from the export every time, so export the whole project.

- Only **reviewed** tasks are kept: tasks with a submitted, not skipped, annotation (the newest one if there are
  several). Tasks nobody has submitted yet are left out, so a half-labelled project gives a smaller file, never a wrong
  one.
- A reviewed task without boxes is an empty scene: an image with no regions, i.e. a background image for YOLO.
- Boxes become VIA "rect" regions in image pixels with region_attributes.Class, which tools/via_to_yolo.py reads like the
  2020 polygons. VIA keys images by file name + size; the day folder is found again by that pair.
- Nothing else from the export is copied: it also holds user names, e-mail addresses and Label Studio ids, and the repo
  is public.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["Blacky", "Niche"]  # as in tools/via_to_yolo.py and tools/labelstudio_config.xml
FROM_NAME = "label"
VIA_ATTRIBUTES = {"region": {"Class": {"type": "radio", "description": "", "options": {c: "" for c in CLASSES},
                                       "default_options": {}}},
                  "file": {}}


def image_path(task: dict, document_root: Path) -> Path:
    """Local path of a task's image from its local-files URL (/data/local-files/?d=<path under the document root>)."""
    url = task["data"]["image"]
    relative = parse_qs(urlparse(url).query).get("d")
    if not relative:
        raise ValueError(f"task {task.get('id')}: not a local-files image URL: {url}")
    return document_root / relative[0]


def latest_submitted(task: dict) -> dict | None:
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    return max(annotations, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0)) if annotations else None


def rect_region(result: dict) -> dict:
    value = result["value"]
    if value.get("rotation"):
        raise ValueError(f"rotated box ({value['rotation']} degrees): rotation is not supported")
    labels = value["rectanglelabels"]
    if len(labels) != 1 or labels[0] not in CLASSES:
        raise ValueError(f"box with labels {labels}: expected exactly one of {CLASSES}")
    width, height = result["original_width"], result["original_height"]
    x1, y1 = max(value["x"], 0) / 100 * width, max(value["y"], 0) / 100 * height
    x2 = min(value["x"] + value["width"], 100) / 100 * width
    y2 = min(value["y"] + value["height"], 100) / 100 * height
    return {"shape_attributes": {"name": "rect", "x": round(x1, 1), "y": round(y1, 1),
                                 "width": round(x2 - x1, 1), "height": round(y2 - y1, 1)},
            "region_attributes": {"Class": labels[0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="Label Studio JSON export")
    parser.add_argument("--document-root", type=Path, default=REPO_ROOT / "data",
                        help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT of the Label Studio server")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "pi-camera" / "captures" / "cats-annotations.json")
    args = parser.parse_args()

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    via, counts = {}, Counter()
    for task in tasks:
        annotation = latest_submitted(task)
        if annotation is None:
            counts["skipped" if task.get("annotations") else "not reviewed"] += 1
            continue
        path = image_path(task, args.document_root)
        if not path.is_file():
            raise FileNotFoundError(f"task {task.get('id')}: {path} is missing (run scripts/pull_captures.py)")
        try:
            regions = [rect_region(r) for r in annotation["result"]
                       if r.get("type") == "rectanglelabels" and r.get("from_name") == FROM_NAME]
        except ValueError as error:
            raise SystemExit(f"task {task.get('id')} ({path.name}): {error}")
        size = path.stat().st_size
        key = f"{path.name}{size}"
        if key in via:
            raise SystemExit(f"{path.name} is in the export twice (two tasks for one image)")
        via[key] = {"filename": path.name, "size": size, "regions": regions, "file_attributes": {}}
        counts["images"] += 1
        counts["empty"] += not regions
        counts.update(r["region_attributes"]["Class"] for r in regions)

    out = {"_via_attributes": VIA_ATTRIBUTES, **dict(sorted(via.items()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{counts['images']} reviewed images ({counts['empty']} empty scenes), "
          f"{', '.join(f'{c} {counts[c]}' for c in CLASSES)} boxes; left out: {counts['not reviewed']} not reviewed, "
          f"{counts['skipped']} skipped -> {args.out}")


if __name__ == "__main__":
    main()
