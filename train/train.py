"""Fine-tune a YOLO detection model on the cats dataset (thin wrapper around Ultralytics training).

Pretrained weights are read from (and downloaded to) models/, training output goes to runs/train/<name>/:
    runs/train/<name>/weights/best.pt, last.pt     checkpoints
    runs/train/<name>/args.yaml, results.csv       full training arguments and per-epoch metrics

    python train/train.py --imgsz 320
    python train/train.py --imgsz 320 --scale 0.9 --name yolo26n_320_scale0.9
    python train/train.py --imgsz 416 mosaic=0.5 hsv_v=0.6      # any other Ultralytics train argument as KEY=VALUE
"""
import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RUNS_DIR = REPO_ROOT / "runs" / "train"


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ["mosaic=0.5", "cos_lr=true"] into {"mosaic": 0.5, "cos_lr": True}."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo26n.pt", help="weights file name in models/ (downloaded if missing) or a path")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "cats" / "cats.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30, help="early stopping: epochs without val improvement")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--scale", type=float, default=0.5, help="scale augmentation gain: random zoom in [1-s, 1+s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", help="run name (default: <model>_<imgsz>[_scale<s>])")
    parser.add_argument("--exist-ok", action="store_true", help="overwrite an existing run with the same name")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE", help="extra Ultralytics train arguments")
    args = parser.parse_args()
    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as error:
        parser.error(str(error))

    # Imported here so --help stays fast.
    import ultralytics.utils
    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = Path(args.model)
    if not model_path.is_file():
        model_path = MODELS_DIR / args.model
    # The AMP self-check loads yolo26n.pt from WEIGHTS_DIR (default: a relative "weights/" folder); reuse models/.
    ultralytics.utils.WEIGHTS_DIR = MODELS_DIR

    name = args.name or f"{model_path.stem}_{args.imgsz}" + (f"_scale{args.scale:g}" if args.scale != 0.5 else "")
    model = YOLO(model_path)  # Ultralytics downloads official weights to this path if they are missing
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        scale=args.scale,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        device=args.device,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=args.exist_ok,
        **overrides,
    )
    print(f"best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
