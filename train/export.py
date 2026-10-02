"""Export trained weights to NCNN for the Raspberry Pi and print the exported model's input/output layout.

Uses Ultralytics' NCNN export (PyTorch -> PNNX -> NCNN). The model folder is written next to the weights:
    runs/train/<run>/weights/best_ncnn_model/
        model.ncnn.param, model.ncnn.bin    the network
        metadata.yaml                       Ultralytics metadata (lets Ultralytics load the folder, e.g. evaluate.py)
        model.json                          class names and input size for pi/detector.py (stdlib json: the Pi
                                            needs no PyYAML)

Ultralytics disables YOLO26's end-to-end (NMS-free) branch for NCNN, so the output is the raw one-to-many head,
the same head Ultralytics' PyTorch predict/val use by default: one row per value, one column per anchor,
[cx, cy, w, h, score_0, ..., score_nc-1] in input pixels with sigmoid scores. pi/detector.py applies the NMS.

    python train/export.py yolo26n_320_scale0.9            # run name in runs/train/; imgsz from its args.yaml
    python train/export.py path/to/best.pt --imgsz 416
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from evaluate import resolve_weights

GREY = 114 / 255  # letterbox padding value after scaling to [0, 1]


def inspect(model_dir: Path):
    """Print the blob names and the output shape for a grey input, as pi/detector.py will see them."""
    import ncnn

    info = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    height, width = info["imgsz"]
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(str(model_dir / "model.ncnn.param"))
    net.load_model(str(model_dir / "model.ncnn.bin"))
    print(f"\nncnn {ncnn.__version__}: {len(net.layers())} layers")
    print(f"  inputs  {net.input_names()}  (C, H, W) = (3, {height}, {width}), RGB in [0, 1]")
    print(f"  outputs {net.output_names()}")

    # ncnn.Mat(array) shares the array's memory: keep both referenced until the extraction is done, or ncnn reads
    # freed memory (garbage outputs or a crash).
    grey = np.full((3, height, width), GREY, dtype=np.float32)
    grey_mat = ncnn.Mat(grey)
    with net.create_extractor() as ex:
        ex.input(net.input_names()[0], grey_mat)
        for name in net.output_names():
            out = np.array(ex.extract(name)[1])
            print(f"  {name}: shape {out.shape} (grey input)")
            labels = ["cx", "cy", "w", "h", *(f"score {n}" for n in info["names"])]
            for row, label in zip(out, labels + [""] * (len(out) - len(labels))):
                print(f"    {label or '?':14} min {row.min():9.4f}  max {row.max():9.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", help="run name in runs/train/ (uses weights/best.pt) or a weights file")
    parser.add_argument("--imgsz", type=int, help="input size (default: the run's training imgsz, else 640)")
    parser.add_argument("--half", action="store_true", help="store FP16 weights (default FP32)")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help stays fast

    weights, _, train_imgsz = resolve_weights(args.weights)
    if weights.suffix != ".pt":
        parser.error(f"expected a .pt weights file, got {weights}")
    imgsz = args.imgsz or train_imgsz or 640
    model_dir = Path(
        YOLO(weights).export(format="ncnn", imgsz=imgsz, quantize=16 if args.half else None, batch=1, device="cpu")
    )
    # PNNX leftovers: a sample script that imports torch, and the bytecode of its intermediate model_pnnx.py
    (model_dir / "model_ncnn.py").unlink(missing_ok=True)
    shutil.rmtree(model_dir / "__pycache__", ignore_errors=True)

    metadata = yaml.safe_load((model_dir / "metadata.yaml").read_text(encoding="utf-8"))
    names = metadata["names"]
    info = {"names": [names[i] for i in range(len(names))], "imgsz": metadata["imgsz"]}
    (model_dir / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    inspect(model_dir)
    print(f"\nmodel folder: {model_dir}")
    for file in sorted(model_dir.iterdir()):
        print(f"  {file.name:20} {file.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
