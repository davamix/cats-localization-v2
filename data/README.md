# Data

The images are not stored in this repository. Download the zip from Google Drive and extract it here:

**Download:** _link coming soon_

Expected layout after extracting:

```
data/
├── train/
│   ├── cats-annotations.json   # tracked in git
│   ├── Blacky/                 # b_1.jpg … b_51.jpg
│   └── Niche/                  # n_1.jpg … n_47.jpg
└── validation/
    ├── cats-annotations.json   # tracked in git
    ├── Blacky/                 # frameN.jpg (21 video frames)
    └── Niche/                  # frameN.jpg (22 video frames)
```

## Annotation format

The annotations were made in 2020 with [VIA 2.0.8](https://www.robots.ox.ac.uk/~vgg/software/via/) and are the
JSON export of a VIA project:

- `_via_attributes` defines the region attribute `Class` with the options `Blacky` and `Niche`.
- Every other key is one image: `filename`, `size` and `regions`. Each region is a `polygon`
  (`all_points_x`, `all_points_y`) with `region_attributes.Class`.
- All images are 1920×1080 and have exactly one region.

`tools/via_to_yolo.py` (phase 1) converts them to the YOLO format used for training.

## Pi-camera captures (phase 6)

Taken with the Raspberry Pi camera by `pi/app.py` (push button, the page's Capture button or a timer) and downloaded
with `python scripts/pull_captures.py`:

```
data/pi-camera/
├── captures/
│   ├── cats-annotations.json        # tracked in git once labelled (tools/labelstudio_to_via.py)
│   └── <YYYY-MM-DD>/                # one folder per capture day
│       ├── <YYYYMMDD-HHMMSS-mmm>_<source>.jpg    # 640x480, JPEG quality 95; source = button / web / timer
│       └── <YYYYMMDD-HHMMSS-mmm>_<source>.json   # time, camera metadata, the model's detections (pre-labels)
└── labelstudio/                     # task files for Label Studio (tools/captures_to_labelstudio.py)
```

`cats-annotations.json` has the VIA 2.0.8 layout described above, with `rect` regions (x, y, width, height in
pixels) instead of polygons; an image with no regions is a reviewed empty scene. Only the annotation file is tracked;
images, the per-capture JSON files and the Label Studio files are not. The captures can show people: they go into
the Drive zip only after a check. Labelling workflow: [phase 6](../docs/phases/phase-6-real-data.md#labelling-workflow-label-studio).
