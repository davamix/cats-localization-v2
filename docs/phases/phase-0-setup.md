# Phase 0 — Environment and repository setup

| | |
|---|---|
| **Status** | In progress |
| **Last updated** | 2026-10-02 |
| **Depends on** | — |

## Goal

Have everything needed to start working: the Git repository on GitHub, a PC environment that trains on the GPU,
remote access to the Pi from the PC, and a Pi with the camera and inference runtime ready.

## Steps

- [x] Inspect the PC (GPU, Python, GitHub CLI) and the Pi (model, OS, RAM, disk, camera). Results are in the
      Environment section of [PLAN.md](../PLAN.md).
- [ ] Repository scaffold: `.gitignore` (no images, datasets, runs, weights, venvs or `pi.env`), `README.md`,
      `LICENSE` (AGPL-3.0), `data/README.md` (Google Drive link placeholder), `pi.env.example`, requirements files.
- [ ] Create the public GitHub repo `davamix/cats-localization-v2` and push the first commit.
- [ ] PC venv `.venv` (Python 3.12): `torch` + `torchvision` (CUDA 12.6 build), `ultralytics`, `paramiko`.
      Verify `torch.cuda.is_available()` and that `sm_75` is in `torch.cuda.get_arch_list()`.
- [ ] Pi remote helper [scripts/pi_remote.py](../../scripts/pi_remote.py): run commands, upload and download
      files over SSH with the credentials from `pi.env`.
- [ ] Pi system packages (apt): `python3-picamera2`, `python3-opencv`, `python3-venv`, `python3-pip`
      (with `--no-install-recommends`, it is OS Lite).
- [ ] Pi venv `~/cats-localization-v2/.venv` created with `--system-site-packages` (so it sees apt's picamera2,
      numpy and OpenCV) and `pip install --no-deps ncnn`.
- [ ] Camera smoke test: capture a still with `rpicam-still` and another one with picamera2, download both and
      check them.

## Done when

- The repo exists on GitHub and contains no images.
- `python -c "import torch; print(torch.cuda.is_available())"` prints `True` in `.venv`.
- `python scripts/pi_remote.py run "hostname"` works from the PC.
- On the Pi, `~/cats-localization-v2/.venv/bin/python -c "import ncnn, picamera2, cv2, numpy"` succeeds and a
  picamera2 capture produces a correct image.

## Handover notes

- **SSH:** the Pi user `pi` has the default password and `sudo` requires it (`sudo -S` reads it from stdin).
  Key-based SSH login was not configured: an attempt to check the local SSH key/agent was blocked by the agent's
  permission rules, so all remote access goes through paramiko with password auth. Before that, the PC's
  `~/.ssh/id_ed25519.pub` had been appended to `/home/pi/.ssh/authorized_keys` on the Pi; it is unused. The user
  decides whether to keep it (and set up key login themselves) or remove it.
- **ncnn on the Pi:** PyPI has `cp313` `manylinux aarch64` wheels. The wheel declares `opencv-python`, `tqdm`,
  `requests` and `portalocker` as dependencies, which are only needed by its model-zoo helpers; install with
  `--no-deps` so pip does not pull a second OpenCV/numpy over the apt ones.
- **Camera:** the 640×480 sensor mode is a centre crop of the sensor. For the full field of view, use the
  1640×1232 mode and let the ISP scale the output stream down.
