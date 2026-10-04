"""Turn Pi-camera captures and their pre-labels into a Label Studio task file (tasks with predictions).

    python tools/captures_to_labelstudio.py                                   # every day in data/pi-camera/captures
    python tools/captures_to_labelstudio.py --days 2026-10-04 2026-10-05      # only these days

Each capture from pi/app.py (data/pi-camera/captures/<day>/<name>.jpg + <name>.json, downloaded with
scripts/pull_captures.py) becomes one task:
    data         image (a Label Studio local-files URL), capture name, day, source (button / web / timer), time, lux
    predictions  the pre-labels: the detections of the model on the Pi with a score >= --min-score, as editable boxes
                 (captures without detections get no prediction)
The labeling config is tools/labelstudio_config.xml. Output (git-ignored): data/pi-camera/labelstudio/<name>.json, to
import in the project with Import > Upload files.

Label Studio reads the images from disk, so it must run with
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<this repository>/data      (= --document-root)
and the project needs a "Local files" source storage whose absolute path is the captures folder (no sync needed). Full
steps: docs/phases/phase-6-real-data.md.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURES = REPO_ROOT / "data" / "pi-camera" / "captures"
FROM_NAME, TO_NAME = "label", "image"  # as in tools/labelstudio_config.xml


def capture_task(sidecar: Path, document_root: Path, min_score: float) -> dict:
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    image = sidecar.with_name(info["image"])
    if not image.is_file():
        raise FileNotFoundError(f"{sidecar}: image {image.name} is missing")
    width, height = info["width"], info["height"]
    relative = image.resolve().relative_to(document_root.resolve()).as_posix()
    task = {"data": {"image": f"/data/local-files/?d={quote(relative)}", "capture": image.stem,
                     "day": image.parent.name, "source": info["source"], "time": info["time"],
                     "lux": round(info["camera"]["Lux"]) if "Lux" in info["camera"] else None}}
    results = []
    for i, detection in enumerate(d for d in info["detections"] if d["score"] >= min_score):
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

    tasks = [capture_task(s, args.document_root, args.min_score) for s in sidecars]
    name = days[0] if len(days) == 1 else f"{days[0]}_to_{days[-1]}"
    out = args.out or REPO_ROOT / "data" / "pi-camera" / "labelstudio" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, indent=1) + "\n", encoding="utf-8", newline="\n")

    boxes = Counter(r["value"]["rectanglelabels"][0] for t in tasks for p in t.get("predictions", [])
                    for r in p["result"])
    with_boxes = sum("predictions" in t for t in tasks)
    print(f"{len(tasks)} tasks from {len(days)} day(s), {with_boxes} with pre-labels "
          f"({', '.join(f'{k} {v}' for k, v in sorted(boxes.items())) or 'no boxes'}) -> {out}")


if __name__ == "__main__":
    main()
