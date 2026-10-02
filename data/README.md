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
