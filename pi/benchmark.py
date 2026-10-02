"""Benchmark the detector on the Raspberry Pi: latency per stage, FPS, memory, temperature and throttling.

Runs pi/detector.py on many frames, from test images or from the camera, and reports:
  - mean / p50 / p95 / max time of each stage: capture (camera only), preprocess, inference, postprocess;
  - FPS: frames per second of the whole loop (capture + detection), and of the detector alone;
  - memory: RSS of this process after the imports, after loading the model, and its peak (VmHWM);
  - CPU temperature, ARM clock and `vcgencmd get_throttled` flags before and after the run, sampled every
    --sample-every seconds during it (between frames, outside the stage timers), and the latency per 30 s window,
    to see whether the Pi heats up and slows down.

    python pi/benchmark.py models/yolo26n_320_scale0.9 --camera --duration 600 --json results/bench_320.json
    python pi/benchmark.py models/yolo26n_320_scale0.9 --images images/compare --frames 300 --threads 2

Images are decoded once and kept in memory, so JPEG decoding and SD-card reads are not timed; they are used in a
loop. Camera frames come from picamera2 as in camera_test.py and the live app: full-field-of-view sensor mode
(1640x1232) scaled by the ISP to --width x --height (640x480), RGB888 (BGR in memory). The first --warmup frames are
not counted. --cool-to waits until the CPU is at or below that temperature before starting (camera not yet running),
so runs start from the same thermal state. Needs ncnn, numpy and OpenCV, plus picamera2 for --camera.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np

from detector import Detector

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
# `vcgencmd get_throttled` bits: 0-3 = the condition now, 16-19 = it has happened since boot
THROTTLE_FLAGS = {0: "under-voltage", 1: "ARM frequency capped", 2: "throttled", 3: "soft temperature limit"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
STAGES = ("capture", "preprocess", "infer", "postprocess")
WINDOW_S = 30


# --- system readings (None when not available, e.g. when testing on a PC) ------------------------------------------

def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cpu_temp() -> float | None:
    value = read_text(THERMAL_ZONE)
    return int(value) / 1000 if value else None


def vcgencmd(*args: str) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def arm_clock_mhz() -> float | None:
    out = vcgencmd("measure_clock", "arm")  # frequency(48)=1200126000
    return round(int(out.split("=")[1]) / 1e6) if out else None


def throttled() -> dict:
    out = vcgencmd("get_throttled")  # throttled=0x50005
    if out is None:
        return {"raw": None, "now": [], "since_boot": []}
    value = int(out.split("=")[1], 16)
    return {"raw": hex(value),
            "now": [name for bit, name in THROTTLE_FLAGS.items() if value >> bit & 1],
            "since_boot": [name for bit, name in THROTTLE_FLAGS.items() if value >> (bit + 16) & 1]}


def proc_kib(path: str, *keys: str) -> dict[str, float | None]:
    """Values in MiB from a /proc file with `Key:   1234 kB` lines."""
    text = read_text(Path(path)) or ""
    values = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    return {key: round(int(values[key].split()[0]) / 1024, 1) if key in values else None for key in keys}


def memory() -> dict[str, float | None]:
    """RSS and peak RSS of this process, and the system's available memory, in MiB."""
    return {**proc_kib("/proc/self/status", "VmRSS", "VmHWM"), **proc_kib("/proc/meminfo", "MemAvailable")}


def snapshot() -> dict:
    return {"temp_c": cpu_temp(), "arm_mhz": arm_clock_mhz(), "throttled": throttled(), "memory": memory()}


def wait_until_cool(limit_c: float, timeout_s: float):
    start, temp = time.monotonic(), cpu_temp()
    if temp is None:
        print("--cool-to: CPU temperature not available, starting now")
        return
    next_print = 0.0
    while temp > limit_c:
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            print(f"cooling: still {temp:.1f} °C after {timeout_s:.0f} s, starting anyway")
            return
        if elapsed >= next_print:
            print(f"cooling: {temp:.1f} °C, waiting for <= {limit_c:.1f} °C", flush=True)
            next_print = elapsed + 30
        time.sleep(5)
        temp = cpu_temp()
    print(f"cooling: {temp:.1f} °C after {time.monotonic() - start:.0f} s")


# --- frame sources ---------------------------------------------------------------------------------------------------

class ImageSource:
    """Decoded images kept in memory and returned in a loop."""

    def __init__(self, paths: list[str]):
        files = []
        for arg in paths:
            path = Path(arg)
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if path.is_dir() else [path]
        self.frames = []
        for file in files:
            frame = cv2.imread(str(file))
            if frame is None:
                sys.exit(f"cannot read image {file}")
            self.frames.append(frame)
        if not self.frames:
            sys.exit("no images found")
        sizes = sorted({f"{f.shape[1]}x{f.shape[0]}" for f in self.frames})
        self.description = f"{len(self.frames)} images ({', '.join(sizes)})"
        self.index = 0

    def read(self) -> np.ndarray:
        frame = self.frames[self.index % len(self.frames)]
        self.index += 1
        return frame

    def close(self):
        pass


class CameraSource:
    """picamera2 frames, configured like camera_test.py."""

    def __init__(self, width: int, height: int, sensor_size: tuple[int, int]):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}, sensor={"output_size": sensor_size}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # let auto exposure / white balance settle
        sensor = self.picam2.camera_configuration()["sensor"]
        self.description = f"camera {width}x{height} (sensor {sensor['output_size'][0]}x{sensor['output_size'][1]})"

    def read(self) -> np.ndarray:
        return self.picam2.capture_array("main")  # RGB888 = BGR in memory, as the detector expects

    def close(self):
        self.picam2.stop()
        self.picam2.close()


# --- statistics ------------------------------------------------------------------------------------------------------

def stats_ms(values: np.ndarray) -> dict[str, float]:
    return {"mean": round(float(values.mean()), 2), "p50": round(float(np.percentile(values, 50)), 2),
            "p95": round(float(np.percentile(values, 95)), 2), "max": round(float(values.max()), 2)}


def windows(t: np.ndarray, total: np.ndarray, infer: np.ndarray, timeline: list[dict]) -> list[dict]:
    """Mean latency, temperature and clock per WINDOW_S seconds of the run."""
    rows = []
    for start in np.arange(0, t[-1] + 1e-9, WINDOW_S):
        in_window = (t >= start) & (t < start + WINDOW_S)
        if not in_window.any():
            continue
        samples = [s for s in timeline if start <= s["t"] < start + WINDOW_S]
        temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
        clocks = [s["arm_mhz"] for s in samples if s["arm_mhz"] is not None]
        rows.append({"t": int(start), "frames": int(in_window.sum()),
                     "total_ms": round(float(total[in_window].mean()), 1),
                     "infer_ms": round(float(infer[in_window].mean()), 1),
                     "temp_max_c": max(temps, default=None), "arm_min_mhz": min(clocks, default=None)})
    return rows


def describe_throttled(state: dict) -> str:
    if state["raw"] is None:
        return "n/a"
    now = ", ".join(state["now"]) or "none"
    since = ", ".join(state["since_boot"]) or "none"
    return f"{state['raw']} (now: {now}; since boot: {since})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="exported model folder (models/<name> on the Pi)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", nargs="+", metavar="PATH", help="test images or folders of images")
    inputs.add_argument("--camera", action="store_true", help="use picamera2 frames")
    parser.add_argument("--width", type=int, default=640, help="camera frame width")
    parser.add_argument("--height", type=int, default=480, help="camera frame height")
    parser.add_argument("--sensor-size", type=int, nargs=2, default=(1640, 1232), metavar=("W", "H"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.5)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--frames", type=int, help="number of timed frames (default 300)")
    length.add_argument("--duration", type=float, help="run for this many seconds instead of a number of frames")
    parser.add_argument("--warmup", type=int, default=10, help="untimed frames before the measurement")
    parser.add_argument("--cool-to", type=float, help="wait until the CPU is at or below this temperature (°C)")
    parser.add_argument("--cool-timeout", type=float, default=1200, help="give up waiting after this many seconds")
    parser.add_argument("--sample-every", type=float, default=5, help="seconds between temperature/clock samples")
    parser.add_argument("--json", type=Path, help="write the summary, per-frame times and samples to this file")
    args = parser.parse_args()
    frames_target = None if args.duration else (args.frames or 300)

    mem_imports = memory()
    detector = Detector(args.model_dir, conf=args.conf, threads=args.threads)
    mem_model = memory()
    if args.images:
        source = ImageSource(args.images)
    if args.cool_to is not None:
        wait_until_cool(args.cool_to, args.cool_timeout)
    if args.camera:
        source = CameraSource(args.width, args.height, tuple(args.sensor_size))

    model_name = Path(args.model_dir).resolve().name
    print(f"model:   {model_name} ({detector.input_width}x{detector.input_height}), {args.threads} threads, FP32, "
          f"conf {args.conf}")
    print(f"source:  {source.description}")
    print(f"length:  {f'{frames_target} frames' if frames_target else f'{args.duration:.0f} s'} "
          f"(+{args.warmup} warm-up)", flush=True)

    try:
        for _ in range(args.warmup):
            detector.detect(source.read())

        before = snapshot()
        times = {stage: [] for stage in STAGES}
        frame_t, detections_per_frame, timeline = [], [], []
        start = next_sample = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now >= next_sample:
                timeline.append({"t": round(now - start, 2), "temp_c": cpu_temp(), "arm_mhz": arm_clock_mhz(),
                                 "throttled": throttled()["raw"]})
                next_sample += args.sample_every
            if (frames_target and len(frame_t) >= frames_target) or (args.duration and now - start >= args.duration):
                break
            t0 = time.perf_counter()
            frame = source.read()
            t1 = time.perf_counter()
            mat, scale, pad = detector.preprocess(frame)
            t2 = time.perf_counter()
            output = detector.infer(mat)
            t3 = time.perf_counter()
            detections = detector.postprocess(output, scale, pad, frame.shape[:2])
            t4 = time.perf_counter()
            for stage, (a, b) in zip(STAGES, [(t0, t1), (t1, t2), (t2, t3), (t3, t4)]):
                times[stage].append((b - a) * 1000)
            frame_t.append(t0 - start)
            detections_per_frame.append(len(detections))
        wall_s = time.perf_counter() - start
        after = snapshot()
    finally:
        source.close()

    ms = {stage: np.array(values) for stage, values in times.items()}
    ms["detect"] = ms["preprocess"] + ms["infer"] + ms["postprocess"]
    ms["total"] = ms["capture"] + ms["detect"]
    summary = {stage: stats_ms(values) for stage, values in ms.items()}
    fps = 1000 / summary["total"]["mean"]
    detector_fps = 1000 / summary["detect"]["mean"]
    t = np.array(frame_t)
    per_window = windows(t, ms["total"], ms["infer"], timeline)
    temps = [s["temp_c"] for s in timeline if s["temp_c"] is not None]
    clocks = [s["arm_mhz"] for s in timeline if s["arm_mhz"] is not None]

    print(f"\n{len(t)} frames in {wall_s:.1f} s, {np.mean(detections_per_frame):.2f} detections per frame")
    print(f"{'stage (ms)':12} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}")
    for stage in (*STAGES, "detect", "total"):
        if stage == "capture" and not args.camera:
            continue
        print(f"{stage:12} " + " ".join(f"{summary[stage][k]:8.1f}" for k in ("mean", "p50", "p95", "max")))
    print(f"FPS:         {fps:.2f} (capture + detection), {detector_fps:.2f} detector only")
    print(f"memory:      RSS {mem_imports['VmRSS']} MiB after imports, {mem_model['VmRSS']} MiB with the model, "
          f"peak {after['memory']['VmHWM']} MiB; available {before['memory']['MemAvailable']} -> "
          f"{after['memory']['MemAvailable']} MiB")
    print(f"temperature: {before['temp_c']} -> {after['temp_c']} °C (max {max(temps, default=None)})")
    print(f"ARM clock:   {before['arm_mhz']} -> {after['arm_mhz']} MHz (min {min(clocks, default=None)})")
    print(f"throttled:   before {describe_throttled(before['throttled'])}")
    print(f"             after  {describe_throttled(after['throttled'])}")
    if len(per_window) > 1:
        print(f"\nper {WINDOW_S} s:  {'t (s)':>6} {'frames':>7} {'total ms':>9} {'infer ms':>9} {'max °C':>7} "
              f"{'min MHz':>8}")
        for row in per_window:
            print(f"            {row['t']:6} {row['frames']:7} {row['total_ms']:9.1f} {row['infer_ms']:9.1f} "
                  f"{row['temp_max_c'] or float('nan'):7.1f} {row['arm_min_mhz'] or float('nan'):8.0f}")

    if args.json:
        result = {
            "config": {"model": model_name, "imgsz": [detector.input_height, detector.input_width],
                       "threads": args.threads, "conf": args.conf, "fp16": False, "source": source.description,
                       "frames": len(t), "warmup": args.warmup, "duration_s": round(wall_s, 1),
                       "cool_to": args.cool_to},
            "system": {"python": platform.python_version(), "ncnn": ncnn.__version__, "numpy": np.__version__,
                       "opencv": cv2.__version__, "kernel": platform.release(), "machine": platform.machine(),
                       "governor": read_text(GOVERNOR)},
            "summary_ms": summary, "fps": round(fps, 2), "detector_fps": round(detector_fps, 2),
            "detections_per_frame": round(float(np.mean(detections_per_frame)), 3),
            "memory_mib": {"after_imports": mem_imports, "after_model": mem_model, "after_run": after["memory"]},
            "before": before, "after": after, "temp_max_c": max(temps, default=None),
            "arm_min_mhz": min(clocks, default=None), "windows": per_window, "timeline": timeline,
            "frames_ms": {"t_s": t.round(3).tolist(), **{s: ms[s].round(2).tolist() for s in STAGES}},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nresults: {args.json}")


if __name__ == "__main__":
    main()
