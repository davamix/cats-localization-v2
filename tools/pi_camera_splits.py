"""Train / validation / test split of the Pi-camera captures, by whole capture day (phase 6 collection plan).

The rule was fixed before looking at the images (docs/phases/phase-6-real-data.md, "Held-out test set"):
- 2026-10-04 and 2026-10-08 (test captures of the capture tooling) are train days and are not numbered;
- the other days with captures are numbered 1, 2, 3, ... in date order;
- every 3rd day (3, 6, 9, ...) is a test day, day 2 is the validation day, and all other days are train days.
Neighbouring frames are near-duplicates, so a whole day always stays in one split. Never move a day to another split
after its results have been seen, and never train or tune anything on the test days.

    python tools/pi_camera_splits.py      # the split of every capture day in data/pi-camera/captures
"""
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURES = REPO_ROOT / "data" / "pi-camera" / "captures"

UNNUMBERED_TRAIN_DAYS = ("2026-10-04", "2026-10-08")
VALIDATION_DAY_NUMBERS = (2,)
TEST_EVERY = 3
SPLITS = ("train", "val", "test")


def capture_days(captures: Path = CAPTURES) -> list[str]:
    """Day folders (YYYY-MM-DD) in the captures folder, in date order."""
    return sorted(p.name for p in captures.iterdir() if p.is_dir())


def split_days(days) -> dict[str, tuple[str, int | None]]:
    """Capture days -> {day: (split, collection day number or None for the unnumbered days)}."""
    splits, number = {}, 0
    for day in sorted(set(days)):
        if day in UNNUMBERED_TRAIN_DAYS:
            splits[day] = ("train", None)
            continue
        number += 1
        if number % TEST_EVERY == 0:
            splits[day] = ("test", number)
        elif number in VALIDATION_DAY_NUMBERS:
            splits[day] = ("val", number)
        else:
            splits[day] = ("train", number)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=CAPTURES, help="captures folder (day folders inside)")
    args = parser.parse_args()
    for day, (split, number) in split_days(capture_days(args.captures)).items():
        images = len(list((args.captures / day).glob("*.jpg")))
        print(f"{day}  {'day ' + str(number) if number else 'unnumbered':>10}  {split:5}  {images} images")


if __name__ == "__main__":
    main()
