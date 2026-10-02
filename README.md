# cats-localization-v2

Real-time detection of my two cats, **Blacky** and **Niche**, on a Raspberry Pi 3 Model B with the Camera
Module v2.1, viewed as a live stream in the browser.

This is the 2026 upgrade of [cats-localization](https://github.com/davamix/cats-localization)
([blog post](https://davamix.net/2020/01/24/detecting-my-cats-with-detectron2/)), which used Detectron2's
Mask R-CNN. Detectron2 is no longer maintained and is far too heavy for a Pi 3B, so v2 trains a **YOLO26n**
model on a PC GPU and runs it on the Pi with **NCNN**.

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the plan, the decisions and the status of each phase.

## How it works

1. **Data** — the 2020 VIA polygon annotations are converted to YOLO bounding boxes (`tools/`).
2. **Training** — YOLO26n is fine-tuned on the PC with Ultralytics and CUDA (`train/`).
3. **Export** — the model is exported to NCNN; trained weights are published as
   [GitHub Releases](https://github.com/davamix/cats-localization-v2/releases).
4. **Inference on the Pi** — picamera2 captures frames, an `ncnn` + `numpy` detector finds the cats, and a small
   web server streams the annotated video (`pi/`).

## Setup

### Data

Images are not in the repository. Download the zip and extract it into `data/` as described in
[data/README.md](data/README.md). The annotation files are already in the repo.

### PC (training)

Windows with an NVIDIA GPU, Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

### Raspberry Pi (inference)

Raspberry Pi OS (64-bit) with the camera connected:

```bash
sudo apt install --no-install-recommends python3-picamera2 python3-opencv python3-venv python3-pip
python3 -m venv --system-site-packages ~/cats-localization-v2/.venv
~/cats-localization-v2/.venv/bin/pip install --no-deps -r requirements-pi.txt
```

### Remote access from the PC

Copy `pi.env.example` to `pi.env` (ignored by git) and fill in the Pi's address and credentials. Then:

```powershell
.\.venv\Scripts\python.exe scripts\pi_remote.py run "hostname"
```

### Deploy and benchmark

Upload the Pi code and one or more exported models (run names in `runs/train/`, or model folders from a release) to
`~/cats-localization-v2/` on the Pi, then benchmark on the Pi:

```powershell
.\.venv\Scripts\python.exe scripts\deploy.py yolo26n_320_scale0.9
.\.venv\Scripts\python.exe scripts\pi_remote.py run "cd ~/cats-localization-v2 && .venv/bin/python pi/benchmark.py models/yolo26n_320_scale0.9 --camera --threads 2"
```

The Pi 3B needs a solid 5.1 V / 2.5 A supply: under-voltage makes the firmware halve the CPU clock, and a heatsink
helps under sustained load (see [docs/results.md](docs/results.md)).

## License

[AGPL-3.0](LICENSE), as required by [Ultralytics](https://github.com/ultralytics/ultralytics), which this
project uses for training.
