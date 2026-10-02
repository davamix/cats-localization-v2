# Phase 0 — Environment and repository setup

| | |
|---|---|
| **Status** | Done |
| **Last updated** | 2026-10-02 |
| **Depends on** | — |

## Goal

Have everything needed to start working: the Git repository on GitHub, a PC environment that trains on the GPU,
remote access to the Pi from the PC, and a Pi with the camera and inference runtime ready.

## Steps

- [x] Inspect the PC (GPU, Python, GitHub CLI) and the Pi (model, OS, RAM, disk, camera). Results are in the
      Environment section of [PLAN.md](../PLAN.md).
- [x] Repository scaffold: `.gitignore` (no images, datasets, runs, weights, venvs or `pi.env`), `.gitattributes`
      (LF line endings), `README.md`, `LICENSE` (AGPL-3.0), `data/README.md` (Google Drive link placeholder),
      `pi.env.example`, requirements files.
- [x] Create the public GitHub repo [davamix/cats-localization-v2](https://github.com/davamix/cats-localization-v2)
      and push the first commit.
- [x] PC venv `.venv` (Python 3.12): `torch` + `torchvision` (CUDA 12.6 build), `ultralytics`, `paramiko`.
      Verify `torch.cuda.is_available()` and that `sm_75` is in `torch.cuda.get_arch_list()`.
- [x] Pi remote helper [scripts/pi_remote.py](../../scripts/pi_remote.py): run commands (also with `--sudo`),
      upload and download files over SSH with the credentials from `pi.env`.
- [x] Pi system packages (apt): `python3-picamera2`, `python3-opencv`, `python3-venv`, `python3-pip`
      (with `--no-install-recommends`, it is OS Lite).
- [x] Pi venv `~/cats-localization-v2/.venv` created with `--system-site-packages` (so it sees apt's picamera2,
      numpy and OpenCV) and `pip install --no-deps ncnn`.
- [x] Camera smoke test with `rpicam-still` and with [pi/camera_test.py](../../pi/camera_test.py) (picamera2);
      both images downloaded and checked.

## Done when

- [x] The repo exists on GitHub and contains no images.
- [x] `python -c "import torch; print(torch.cuda.is_available())"` prints `True` in `.venv`.
- [x] `python scripts/pi_remote.py run "hostname"` works from the PC.
- [x] On the Pi, `~/cats-localization-v2/.venv/bin/python -c "import ncnn, picamera2, cv2, numpy"` succeeds and a
      picamera2 capture produces a correct image.

## Results

| Where | Component | Version |
|---|---|---|
| PC | torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 (CUDA available, RTX 2080 Ti, `sm_75` in arch list) |
| PC | ultralytics | 8.4.171 |
| PC | paramiko | 5.0.0 |
| Pi | python3-picamera2 (apt) | 0.3.37 |
| Pi | OpenCV (apt) | 4.10.0 |
| Pi | numpy (apt) | 2.2.4 |
| Pi | ncnn (pip, venv) | 1.0.20260526 |

Camera test (`pi/camera_test.py`, 640×480 output): sensor mode 1640×1232 (10-bit), ScalerCrop
`(0, 2, 3280, 2460)` = full field of view, `capture_array` 13.6 ms. Colours are correct when the frame is saved
with `cv2.imwrite` directly, which confirms picamera2 `RGB888` frames are **BGR** in memory.

## Handover notes

- **SSH:** the Pi user `pi` has the default password and `sudo` requires it; `pi_remote.py run --sudo` passes it on
  stdin. Key-based SSH login was not configured: an attempt to check the local SSH key/agent was blocked by the
  agent's permission rules, so all remote access goes through paramiko with password auth. Before that, the PC's
  `~/.ssh/id_ed25519.pub` had been appended to `/home/pi/.ssh/authorized_keys` on the Pi; it is unused. The user
  decides whether to keep it (and set up key login themselves) or remove it.
- **Shell gotchas on the PC:**
  - Git Bash rewrites arguments that start with `/` (e.g. `/tmp/x.jpg`) into Windows paths. Use
    `MSYS_NO_PATHCONV=1` or `~/...` paths when passing remote paths to `pi_remote.py get/put`.
  - PowerShell mangles nested quotes in `pi_remote.py run "..."`; for commands with quotes inside, use Git Bash
    with single quotes around the remote command.
- **ncnn on the Pi:** PyPI has `cp313` `manylinux aarch64` wheels. The wheel declares `opencv-python`, `tqdm`,
  `requests` and `portalocker` as dependencies, which are only needed by its model-zoo helpers; install with
  `--no-deps` so pip does not pull a second OpenCV/numpy over the apt ones.
- **Camera:** the 640×480 sensor mode is a centre crop of the sensor. For the full field of view use the 1640×1232
  mode and let the ISP scale the output stream down (as `pi/camera_test.py` does). The camera is currently
  pointing at the ceiling; it needs a real position before phase 5/6 tests.
- **Disk:** the Pi's SD card has 2.5 GB free (6.8 GB root) after the apt install. Phase 6 captures should be moved
  to the PC regularly.
- **Next:** phase 1 (dataset conversion). The images are already in `data/` on this PC.
