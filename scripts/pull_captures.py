"""Download new captures (training images from pi/app.py) from the Raspberry Pi to the PC.

    python scripts/pull_captures.py              # copy what the PC does not have yet
    python scripts/pull_captures.py --dry-run    # only list it

Mirrors <PI_PROJECT_DIR>/captures/<YYYY-MM-DD>/... on the Pi into data/pi-camera/captures/<YYYY-MM-DD>/... (git-ignored,
like all images under data/). Copies only files that are not on the PC yet, through a temporary file, and keeps their
modification times. A file that exists on both with a different size is reported and left alone. Skips the app's
temporary files (.<name>.part). Never deletes or changes anything on the Pi.
"""
import argparse
import os
import posixpath
import stat
from pathlib import Path

from pi_remote import REPO_ROOT, Pi

DEFAULT_DEST = REPO_ROOT / "data" / "pi-camera" / "captures"


def remote_files(pi: Pi, folder: str, prefix: str = "") -> list[tuple[str, int, int]]:
    """Files under a remote folder -> [(relative POSIX path, size, mtime)], sorted."""
    files = []
    for entry in pi.sftp.listdir_attr(folder):
        relative = posixpath.join(prefix, entry.filename) if prefix else entry.filename
        if stat.S_ISDIR(entry.st_mode):
            files += remote_files(pi, posixpath.join(folder, entry.filename), relative)
        elif not entry.filename.startswith("."):  # .<name>.part = still being written
            files.append((relative, entry.st_size, entry.st_mtime))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="local captures folder")
    parser.add_argument("--dry-run", action="store_true", help="list the files to copy, copy nothing")
    args = parser.parse_args()

    with Pi() as pi:
        source = posixpath.join(pi.remote_path(pi.config["PI_PROJECT_DIR"]), "captures")
        try:
            files = remote_files(pi, source)
        except FileNotFoundError:
            raise SystemExit(f"no captures on the Pi yet ({source} does not exist)")
        new, conflicts = [], []
        for relative, size, mtime in files:
            local = args.dest / relative
            if not local.exists():
                new.append((relative, size, mtime))
            elif local.stat().st_size != size:
                conflicts.append((relative, size, local.stat().st_size))

        for i, (relative, size, mtime) in enumerate(new, 1):
            local = args.dest / relative
            if args.dry_run:
                print(f"would copy {relative} ({size / 1024:.0f} KB)")
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            temp = local.with_name(f".{local.name}.part")
            try:
                pi.sftp.get(posixpath.join(source, relative), str(temp))
                os.utime(temp, (mtime, mtime))
                os.replace(temp, local)
            finally:
                temp.unlink(missing_ok=True)
            if i % 50 == 0:
                print(f"  {i} / {len(new)} files", flush=True)

        _, free = pi.run(f"df -h --output=avail {source} | tail -1", echo=False)

    for relative, remote_size, local_size in conflicts:
        print(f"CONFLICT {relative}: {remote_size} bytes on the Pi, {local_size} on the PC (left alone)")
    images = sum(relative.endswith(".jpg") for relative, _, _ in new)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {len(new)} new files ({images} images, {sum(s for _, s, _ in new) / 1e6:.1f} MB) to {args.dest}")
    pi_images = sum(relative.endswith(".jpg") for relative, _, _ in files)
    print(f"on the Pi: {pi_images} images, {sum(s for _, s, _ in files) / 1e6:.1f} MB in captures/, "
          f"{free.strip()} free on the SD card")


if __name__ == "__main__":
    main()
