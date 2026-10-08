"""Deploy the Pi code, exported models and test images to the Raspberry Pi.

Always uploads pi/ and requirements-pi.txt to the project folder on the Pi (PI_PROJECT_DIR in pi.env, default
~/cats-localization-v2); model folders and test images are optional:

    ~/cats-localization-v2/
        .venv/                Python environment (set up once by hand, see requirements-pi.txt; not touched)
        requirements-pi.txt
        pi/                   code
        models/<name>/        model.ncnn.param, model.ncnn.bin, model.json
        images/<set>/         test images (--images)
        results/              outputs of pi/benchmark.py and pi/detector.py --json (not touched)
        captures/             training images saved by pi/app.py (not touched; scripts/pull_captures.py downloads them)

    python scripts/deploy.py                                            # code only
    python scripts/deploy.py yolo26n_320_scale0.9 yolo26n_416           # code + models (run names in runs/train/)
    python scripts/deploy.py downloads/yolo26n_320_scale0.9_ncnn_model  # a model folder, e.g. from a GitHub Release
    python scripts/deploy.py --images runs/pi/compare --set compare     # code + test images

A run name is read from runs/train/<run>/weights/best_ncnn_model and deployed as models/<run>. A folder is deployed
under its own name without the `_ncnn_model` suffix (`.../<run>/weights/best_ncnn_model` also becomes models/<run>).
Only the files pi/detector.py needs are uploaded. pi/, each models/<name> and images/<set> are replaced as a whole, so
files deleted locally do not linger on the Pi. Nothing is installed. If the app runs as the cats-app service
(pi/system/install.sh), it keeps running the old code until `sudo systemctl restart cats-app`; the script says so.
"""
import argparse
import posixpath
import shlex
import sys
from pathlib import Path

from pi_remote import REPO_ROOT, Pi

MODEL_FILES = ("model.ncnn.param", "model.ncnn.bin", "model.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_model(arg: str) -> tuple[Path, str]:
    """Model argument (run name or folder) -> (local model folder, name on the Pi)."""
    folder = Path(arg)
    if folder.is_dir():
        folder = folder.resolve()
        if folder.name == "best_ncnn_model" and folder.parent.name == "weights":
            name = folder.parent.parent.name
        else:
            name = folder.name.removesuffix("_ncnn_model")
    else:
        folder, name = REPO_ROOT / "runs" / "train" / arg / "weights" / "best_ncnn_model", arg
    missing = [f for f in MODEL_FILES if not (folder / f).is_file()]
    if missing:
        sys.exit(f"{arg}: {folder} is missing {', '.join(missing)} (export it with train/export.py)")
    return folder, name


def collect_images(paths: list[str]) -> list[Path]:
    images = []
    for arg in paths:
        path = Path(arg)
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            images.append(path)
        else:
            sys.exit(f"no such image or folder: {arg}")
    if len({p.name for p in images}) != len(images):
        sys.exit("image file names must be unique (pi/detector.py --json keys results by file name)")
    return images


def replace_dir(pi: Pi, remote_dir: str):
    """Delete a folder inside the project folder on the Pi (it is re-uploaded right after)."""
    pi.run(f"rm -rf {shlex.quote(remote_dir)}", echo=False)


def size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="run names in runs/train/ or exported model folders")
    parser.add_argument("--images", nargs="+", metavar="PATH", help="test images or folders of images to upload")
    parser.add_argument("--set", default="test", help="name of the image set on the Pi (images/<set>/)")
    args = parser.parse_args()

    models = [resolve_model(m) for m in args.models]  # check everything before connecting
    if len({name for _, name in models}) != len(models):
        sys.exit("two models would be deployed under the same name")
    images = collect_images(args.images) if args.images else []

    with Pi() as pi:
        project = pi.remote_path(pi.config["PI_PROJECT_DIR"])

        replace_dir(pi, posixpath.join(project, "pi"))
        pi.put(REPO_ROOT / "pi", posixpath.join(project, "pi"))
        pi.put(REPO_ROOT / "requirements-pi.txt", posixpath.join(project, "requirements-pi.txt"))
        code = [p for p in (REPO_ROOT / "pi").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        print(f"code:   pi/ ({len(code)} files) and requirements-pi.txt -> {project}/")

        for folder, name in models:
            remote_dir = posixpath.join(project, "models", name)
            replace_dir(pi, remote_dir)
            for file in MODEL_FILES:
                pi.put(folder / file, posixpath.join(remote_dir, file))
            print(f"model:  {folder} ({size_mb([folder / f for f in MODEL_FILES]):.1f} MB) -> models/{name}/")

        if images:
            remote_dir = posixpath.join(project, "images", args.set)
            replace_dir(pi, remote_dir)
            for image in images:
                pi.put(image, posixpath.join(remote_dir, image.name))
            print(f"images: {len(images)} files ({size_mb(images):.1f} MB) -> images/{args.set}/")

        _, listing = pi.run(f"cd {shlex.quote(project)} && ls models 2>/dev/null | tr '\\n' ' '; "
                            "echo; df -h --output=avail . | tail -1", echo=False)
        deployed, free = (listing.splitlines() + ["", ""])[:2]
        print(f"on the Pi: models [{deployed.strip() or 'none'}], {free.strip()} free")
        _, service = pi.run("systemctl is-active cats-app", echo=False)
        if service.strip() == "active":
            print("cats-app service is running the old code: sudo systemctl restart cats-app "
                  "(and sudo bash pi/system/install.sh after changing pi/system/)")


if __name__ == "__main__":
    main()
