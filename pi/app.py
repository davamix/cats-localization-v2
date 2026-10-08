"""Live stream web app for the Raspberry Pi: camera + cat detector, watched in the browser, plus captures for training.

    python pi/app.py --model models/yolo26n_320_scale0.9

then open http://<pi-ip>:8000/ :
    /                   page with the live stream, stats and a Capture button
    /stream.mjpg        MJPEG stream (multipart/x-mixed-replace) with the boxes, labels, scores and FPS drawn
    /snapshot.jpg       one annotated frame
    /stats              JSON: capture / detection / stream FPS, stage times, the current detections, CPU temperature,
                        ARM clock, `vcgencmd get_throttled` flags, memory and the captures
    POST /capture       save a capture (see below); answers with JSON
    /captures/last.jpg  the last saved capture

Captures (training images, phase 6): a press of the push button on --button-pin (BCM numbering, button to GND with the
internal pull-up by default), POST /capture, or a timer (--capture-every minutes) saves the raw camera frame (no boxes)
as a JPEG (--capture-quality) under --captures/<YYYY-MM-DD>/<YYYYMMDD-HHMMSS-mmm>_<source>.jpg, with a JSON file of the
same name: time, source, camera metadata (exposure, gain, lux, colour temperature), the model and its detections with
a score >= --prelabel-conf as pre-labels. The saved frame is the one the latest finished detection ran on, so image and
pre-labels match even when a cat moves (it is up to ~0.5 s older than the live picture). At most one capture per
--capture-min-interval seconds; none when the captures use more than --capture-max-mb or the SD card has less than
--capture-min-free-mb free, and none before the system clock is NTP-synchronised (--wait-for-clock): the Pi has no
clock battery, so after a power-on its clock is behind until it syncs, and a capture would get a wrong date.

Button and LED (phase 6 collection, the app starts at boot with pi/system/cats-app.service): with --button-hold-s
(5 s), a short press takes a capture when the button is released, and holding it powers the Pi off cleanly
(`sudo -n systemctl poweroff`, allowed by pi/system/cats-poweroff.sudoers). The LED on --led-pin (24, through a
220 ohm resistor to GND) shows the state: slow blink = starting or waiting for the clock, off = ready, on for 1 s =
capture saved, 3 quick blinks = capture refused, fast blink = powering off, or the app stopped itself.

How it works (a slow detector must not freeze the video):
  - capture thread: picamera2 with the full-field-of-view sensor mode (1640x1232) scaled by the ISP to
    --width x --height, RGB888 (BGR in memory, which OpenCV and the detector expect), as in camera_test.py. The
    camera runs at --stream-fps; the thread keeps the latest frame.
  - detector process: ncnn holds Python's GIL while it runs (~210 ms per frame on the Pi), which would stall every
    other thread of the process, so pi/detector.py runs in a child process. The detection thread sends it the
    latest frame over a pipe and keeps the latest detections.
  - render thread: while someone watches the stream, draws the latest detections on each new frame and JPEG-encodes
    it, once for all viewers.
  - monitor thread: every 2 s reads the CPU temperature, ARM clock, throttle flags and memory. Under-voltage caps
    the CPU at 600 MHz and can end in a brownout reset, even as short dips (phase 4), so when the last
    --undervoltage-window seconds (60) hold --stop-on-undervoltage seconds of it in total (10) or --stop-on-dips
    separate dips (3), the app stops with exit code 2 (0 = no limit).
  - save thread: takes capture requests from a queue (the button callback, which runs in gpiozero's thread, only
    queues one) and the timer, checks the limits and writes the files. The monitor thread enables captures once
    the first detection is done and the clock is synchronised.
  - HTTP: the standard library's ThreadingHTTPServer.

Ctrl+C or SIGTERM shuts down cleanly and releases the GPIO pins. Exit code: 0 after a normal stop, 1 when a part
fails, 2 after too much under-voltage (the LED blinks fast for 5 s before the app exits with 1 or 2). Needs ncnn,
numpy, OpenCV, picamera2 and gpiozero + lgpio for the button and LED (no torch).
"""
import argparse
import json
import multiprocessing
import os
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from benchmark import UndervoltageGuard, arm_clock_mhz, cpu_temp, proc_kib, throttled, undervoltage_now
from detector import Detector

SENSOR_SIZE = (1640, 1232)  # IMX219 full field of view (the 640x480 sensor mode is a crop)
MONITOR_S = 2  # seconds between temperature / clock / throttle / memory samples
LOG_EVERY_S = 60  # seconds between status lines in the log
WINDOW_S = 10  # FPS and mean times are computed over this many seconds
WAIT_S = 0.5  # longest a thread waits before it checks for shutdown
DETECTOR_START_S = 120  # spawning Python and loading ncnn + the model takes a few seconds on the Pi
BOUNDARY = b"frame"
COLORS = [(0, 190, 255), (255, 190, 0), (80, 220, 80), (200, 80, 255)]  # box colour per class id (BGR)
WARNING_COLOR = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# camera metadata saved with each capture (to sort the images by lighting)
CAMERA_KEYS = ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "ColourTemperature", "ColourGains")
CAPTURE_WAIT_S = 5  # longest POST /capture waits for the save thread
# capture result status -> HTTP status of POST /capture
CAPTURE_HTTP = {"saved": 201, "too_soon": 429, "not_ready": 503, "disk_limit": 507, "stopping": 503, "timeout": 504}
CLOCK_SYNCED = Path("/run/systemd/timesync/synchronized")  # created by systemd-timesyncd at the first NTP sync
POWEROFF = ["sudo", "-n", "/usr/bin/systemctl", "poweroff"]  # allowed without a password by cats-poweroff.sudoers
ALARM_S = 5  # the LED blinks fast this long before the app exits with an error


def log(message: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)  # with the date: the service log spans days


def fmt(value, spec: str = ".0f") -> str:
    return "n/a" if value is None else format(value, spec)


# --- detector process ------------------------------------------------------------------------------------------------

def detector_process(conn, model_dir: str, conf: float, threads: int):
    """Child process: detect on each frame that arrives on `conn`, reply (detections, stage times in ms).

    Ends on None or when the parent's end of the pipe closes, so it never outlives the app.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C reaches the whole process group; the parent stops us
    detector = Detector(model_dir, conf=conf, threads=threads)
    conn.send({"names": detector.names, "imgsz": [detector.input_height, detector.input_width]})
    try:
        while (frame := conn.recv()) is not None:
            t0 = time.perf_counter()
            mat, scale, pad = detector.preprocess(frame)  # copies the pixels into a new Mat
            t1 = time.perf_counter()
            output = detector.infer(mat)
            t2 = time.perf_counter()
            detections = detector.postprocess(output, scale, pad, frame.shape[:2])
            t3 = time.perf_counter()
            conn.send((detections, {"preprocess": (t1 - t0) * 1000, "infer": (t2 - t1) * 1000,
                                    "postprocess": (t3 - t2) * 1000}))
    except (EOFError, BrokenPipeError):
        pass  # the app has gone


# --- shared state ----------------------------------------------------------------------------------------------------

class Window:
    """Events of the last WINDOW_S seconds, each with optional times in ms: event rate and mean times."""

    def __init__(self):
        self.events = deque()  # (perf_counter time, {name: ms})

    def add(self, now: float, times: dict | None = None):
        self.events.append((now, times or {}))
        self.prune(now)

    def prune(self, now: float):
        while self.events and self.events[0][0] < now - WINDOW_S:
            self.events.popleft()

    def rate(self, now: float) -> float:
        self.prune(now)
        if len(self.events) < 2:
            return 0.0
        return (len(self.events) - 1) / max(self.events[-1][0] - self.events[0][0], 1e-6)

    def means(self) -> dict[str, float]:
        if not self.events:
            return {}
        return {key: round(sum(times[key] for _, times in self.events) / len(self.events), 1)
                for key in self.events[-1][1]}

    def last(self) -> dict[str, float]:
        return {key: round(value, 1) for key, value in self.events[-1][1].items()} if self.events else {}


class CaptureRequest:
    """One request to save a capture; the save thread sets `result` (a dict with "status") and `done`."""

    def __init__(self, source: str):
        self.source, self.result, self.done = source, None, threading.Event()

    def finish(self, result: dict):
        self.result = result
        self.done.set()


class Indicator:
    """The status LED: a base state plus short patterns on top of it. Does nothing without an LED.

    Base states: "starting" = slow blink (starting, or waiting for the clock), "ready" = off, "alarm" = fast blink
    (powering off, or stopping on an error). Patterns: "saved" = on for 1 s, "refused" = 3 quick blinks; then back to
    the base state. Called from several threads (save, monitor, gpiozero's, main), hence the lock.
    """

    def __init__(self, led=None):
        self.led, self.lock, self.base = led, threading.Lock(), "ready"
        self.pattern_id, self.pattern_running = 0, False

    def set_base(self, base: str):
        with self.lock:
            self.base = base
            if base == "alarm" or not self.pattern_running:  # a running pattern shows the new base when it ends
                self.pattern_running = False
                self.show_base()

    def show_base(self):  # under self.lock
        if self.led is None:
            return
        if self.base == "starting":
            self.led.blink(0.5, 0.5)
        elif self.base == "alarm":
            self.led.blink(0.1, 0.1)
        else:
            self.led.off()

    def pattern(self, name: str):
        with self.lock:
            if self.led is None or self.base == "alarm":
                return
            self.pattern_id += 1
            self.pattern_running = True
            if name == "saved":
                self.led.on()
                seconds = 1.0
            else:
                self.led.blink(0.15, 0.15, n=3)
                seconds = 0.9
            timer = threading.Timer(seconds, self.end_pattern, args=(self.pattern_id,))
            timer.daemon = True
            timer.start()

    def end_pattern(self, pattern_id: int):
        with self.lock:
            if self.pattern_running and pattern_id == self.pattern_id:  # not replaced or cancelled in the meantime
                self.pattern_running = False
                self.show_base()

    def close(self):
        with self.lock:
            if self.led is not None:
                self.led.close()  # off; the pin goes back to an input
                self.led = None


def folder_usage(folder: Path) -> tuple[int, int, Path | None]:
    """Captures already on disk -> (number of images, bytes of all files, newest image or None)."""
    images, size, newest = 0, 0, None
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            size += path.stat().st_size
            if path.suffix == ".jpg":
                images += 1
                newest = path if newest is None or path.name > newest.name else newest  # names sort by time
    return images, size, newest


def write_file(path: Path, data: bytes):
    """Write via a hidden temporary file + rename, synced to disk: no half-written capture after a reset, and
    scripts/pull_captures.py never copies a file that is still being written."""
    temp = path.with_name(f".{path.name}.part")
    with open(temp, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


class App:
    def __init__(self, args):
        self.args = args
        self.cond = threading.Condition()  # guards the state below; notified on every new frame and JPEG
        self.stopping = threading.Event()
        self.exit_code, self.stop_reason = 0, None
        self.start_time = time.perf_counter()
        self.config = {}
        self.names = []
        # latest camera frame (never modified: the render thread draws on a copy) and its camera metadata
        self.frame, self.frame_id, self.frame_time, self.frame_metadata = None, 0, 0.0, {}
        self.captured = Window()
        # latest detections with a score >= --conf (drawn and in /stats)
        self.detections = np.zeros((0, 6), np.float32)
        self.detected_time, self.detected_count = None, 0
        self.detected = Window()
        # the frame of the latest finished detection with all its detections >= the detector's threshold (captures)
        self.detected_frame = None  # (frame, frame_time, metadata, detections)
        # captures
        self.capture_requests = queue.Queue()
        self.capture_dir = Path(args.captures)
        self.capture_count, self.last_capture_time = 0, None
        self.capture_sources, self.capture_rejected = Counter(), Counter()
        self.capture_files, self.capture_bytes, self.last_capture_path = 0, 0, None
        self.last_capture = None  # summary of the last capture for /stats
        self.next_timed = None  # perf_counter time of the next timed capture
        self.disk_free = None  # bytes free on the captures' file system
        self.button, self.button_state, self.button_held = None, "off", False
        self.indicator, self.led_state = Indicator(), "off"
        # captures start once the first detection is done and the clock is synchronised
        self.ready, self.clock_synced, self.session = False, False, None
        # latest annotated JPEG for the stream
        self.jpeg, self.jpeg_id, self.viewers = None, 0, 0
        self.rendered = Window()
        # latest system sample
        self.system, self.memory = {}, {}
        self.samples = self.undervoltage_samples = self.capped_samples = 0
        self.guard = UndervoltageGuard(args.stop_on_undervoltage, args.stop_on_dips, args.undervoltage_window)

        self.server = self.server_thread = self.process = self.conn = self.picam2 = None
        self.threads = []

    # --- start / stop ---

    def start(self):
        args = self.args
        self.server = ThreadingHTTPServer(("", args.port), Handler)  # bind first: fail fast if the port is taken
        self.server.app = self
        self.start_led()  # first: it blinks while the detector and the camera start
        self.start_detector()
        self.start_camera()
        self.start_captures()
        self.config = {"model": Path(args.model).name, "imgsz": self.imgsz, "threads": args.threads, "conf": args.conf,
                       "prelabel_conf": args.prelabel_conf, "frame": [args.width, args.height],
                       "sensor": list(SENSOR_SIZE), "stream_fps": args.stream_fps, "jpeg_quality": args.jpeg_quality,
                       "undervoltage_stop": {"seconds": args.stop_on_undervoltage, "dips": args.stop_on_dips,
                                             "window_s": args.undervoltage_window}}
        for loop in (self.capture_loop, self.detection_loop, self.render_loop, self.monitor_loop, self.save_loop):
            self.threads.append(self.start_thread(loop))
        self.start_button()  # last: a press only queues a request, which the save thread handles
        self.server_thread = threading.Thread(target=self.server.serve_forever, args=(WAIT_S,), name="http",
                                              daemon=True)
        self.server_thread.start()
        log(f"serving on http://{socket.gethostname()}:{args.port}/ (Ctrl+C to stop)")

    def start_detector(self):
        args = self.args
        context = multiprocessing.get_context("spawn")  # a fresh interpreter: no camera or threads inherited
        self.conn, child_conn = context.Pipe()
        # The detector keeps everything above the lower of the two thresholds; the app filters at --conf for the
        # stream. Boxes >= --conf are the same either way: NMS only removes a box for one with a higher score.
        self.process = context.Process(target=detector_process, name="detector", daemon=True,
                                       args=(child_conn, args.model, min(args.conf, args.prelabel_conf),
                                             args.threads))
        self.process.start()
        child_conn.close()  # keep only the child's copy, so a dead child shows up as EOFError
        try:
            if not self.conn.poll(DETECTOR_START_S):
                raise TimeoutError
            info = self.conn.recv()
        except (EOFError, TimeoutError):
            raise RuntimeError("the detector process did not start (see its error above)") from None
        self.names, self.imgsz = info["names"], info["imgsz"]
        log(f"detector: {args.model} ({self.imgsz[1]}x{self.imgsz[0]}), {args.threads} threads, FP32, "
            f"conf {args.conf} (pre-labels {args.prelabel_conf}), classes {', '.join(self.names)} "
            f"(pid {self.process.pid})")

    def start_camera(self):
        from picamera2 import Picamera2  # imported here so the detector process does not load it

        args = self.args
        frame_us = round(1e6 / args.stream_fps)
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            sensor={"output_size": SENSOR_SIZE},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
        )
        self.picam2.configure(config)
        self.picam2.start()
        sensor = self.picam2.camera_configuration()["sensor"]
        log(f"camera: {args.width}x{args.height} RGB888 at {args.stream_fps} fps, sensor mode {sensor}")

    def start_captures(self):
        args = self.args
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_files, self.capture_bytes, self.last_capture_path = folder_usage(self.capture_dir)
        if self.last_capture_path:  # show the newest capture from an earlier run until the first new one
            try:
                info = json.loads(self.last_capture_path.with_suffix(".json").read_text(encoding="utf-8"))
                self.last_capture = self.capture_summary(self.last_capture_path, info)
            except (OSError, ValueError, KeyError):
                pass
        self.disk_free = shutil.disk_usage(self.capture_dir).free
        log(f"captures: {self.capture_dir.resolve()} ({self.capture_files} images, {self.capture_bytes / 1e6:.1f} MB; "
            f"{self.disk_free / 1e6:.0f} MB free), JPEG quality {args.capture_quality}, pre-labels >= "
            f"{args.prelabel_conf}, at most one per {args.capture_min_interval:g} s, timed "
            f"{f'every {args.capture_every:g} min' if args.capture_every else 'off'}; limits {args.capture_max_mb:g} MB "
            f"of captures, {args.capture_min_free_mb:g} MB free")

    def start_led(self):
        pin = self.args.led_pin
        if not pin:
            return
        try:
            from gpiozero import LED  # imported here so the detector process does not load it

            self.indicator = Indicator(LED(pin))
        except Exception as error:  # no GPIO (e.g. pin busy, or not a Pi): run without the LED
            self.led_state = f"unavailable: {error}"
            log(f"LED: GPIO {pin} {self.led_state}")
            return
        self.indicator.set_base("starting")
        self.led_state = f"GPIO {pin}"
        log(f"LED: GPIO {pin} (slow blink = starting / waiting for the clock, off = ready, on 1 s = saved, "
            f"3 blinks = refused, fast blink = powering off / stopped on an error)")

    def start_button(self):
        args = self.args
        if not args.button_pin:
            return
        try:
            from gpiozero import Button, Device  # imported here so the detector process does not load it

            self.button = Button(args.button_pin, pull_up=args.button_pull == "up",
                                 bounce_time=args.button_bounce_ms / 1000 or None, hold_time=args.button_hold_s or 1)
            if args.button_hold_s:  # capture on release, so a long press only powers off
                self.button.when_released = self.on_release
                self.button.when_held = self.on_hold
            else:
                self.button.when_pressed = self.on_press
            factory = type(Device.pin_factory).__name__
        except Exception as error:  # no GPIO (e.g. pin busy, or not a Pi): run without the button
            self.button, self.button_state = None, f"unavailable: {error}"
            log(f"button: GPIO {args.button_pin} {self.button_state}; captures only from the page and the timer")
            return
        self.button_state = (f"GPIO {args.button_pin}, pull-{args.button_pull}, debounce {args.button_bounce_ms:g} ms, "
                             + (f"hold {args.button_hold_s:g} s = power off" if args.button_hold_s else "no power-off")
                             + f" ({factory})")
        log(f"button: {self.button_state}")

    # gpiozero calls these in its own thread: only queue a request, or start a thread

    def on_press(self):
        self.capture_requests.put(CaptureRequest("button"))

    def on_release(self):
        if self.button_held:  # the end of a long press: no capture
            self.button_held = False
        else:
            self.capture_requests.put(CaptureRequest("button"))

    def on_hold(self):
        self.button_held = True
        threading.Thread(target=self.power_off, name="poweroff", daemon=True).start()

    def power_off(self):
        """Long press: power the Pi off. systemd then stops this app with SIGTERM like any other service."""
        log(f"button held for {self.args.button_hold_s:g} s: powering off ({' '.join(POWEROFF)})")
        self.indicator.set_base("alarm")
        try:
            result = subprocess.run(POWEROFF, capture_output=True, text=True, timeout=30)
            error = None if result.returncode == 0 else (result.stderr.strip() or f"exit code {result.returncode}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
        if error:  # e.g. pi/system/cats-poweroff.sudoers is not installed
            log(f"power-off failed: {error}")
            self.indicator.set_base("ready" if self.ready else "starting")
            self.indicator.pattern("refused")

    def start_thread(self, loop) -> threading.Thread:
        def run():
            try:
                loop()
            except Exception:
                if not self.stopping.is_set():  # errors while shutting down (e.g. a closed pipe) are expected
                    log(f"{thread.name} thread failed:\n{traceback.format_exc()}")
                    self.stop(1, f"the {thread.name} thread failed")

        thread = threading.Thread(target=run, name=loop.__name__.removesuffix("_loop"), daemon=True)
        thread.start()
        return thread

    def stop(self, exit_code: int = 0, reason: str = "stop requested"):
        """Ask the app to stop; the main thread then calls shutdown(). Safe to call from a signal handler."""
        if not self.stopping.is_set():
            self.exit_code, self.stop_reason = exit_code, reason
            self.stopping.set()

    def shutdown(self):
        self.stopping.set()
        log(f"stopping: {self.stop_reason or 'startup failed'}")
        if self.button:
            self.button.close()  # no more presses; releases the pin (it stays an input)
        with self.cond:
            self.cond.notify_all()
        if self.server_thread:
            self.server.shutdown()  # stream handlers see `stopping` and return
        for thread in self.threads:
            thread.join(timeout=5)  # every loop checks `stopping` at least every WAIT_S (detection: after a frame)
        if self.process:
            try:
                self.conn.send(None)
            except OSError:
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            self.conn.close()
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        if self.server:
            self.server.server_close()
        if self.exit_code and self.indicator.led is not None:  # stopped on an error: show it before the LED goes off
            self.indicator.set_base("alarm")
            time.sleep(ALARM_S)
        self.indicator.close()
        log(f"stopped (exit code {self.exit_code})")

    # --- worker threads ---

    def capture_loop(self):
        while not self.stopping.is_set():
            request = self.picam2.capture_request()  # what capture_array() does, plus the metadata
            try:
                frame = request.make_array("main")  # a new array per frame, BGR in memory
                metadata = request.get_metadata()
            finally:
                request.release()
            now = time.perf_counter()
            with self.cond:
                self.frame, self.frame_time, self.frame_metadata = frame, now, metadata
                self.frame_id += 1
                self.captured.add(now)
                self.cond.notify_all()

    def wait_for(self, predicate) -> bool:
        """Under self.cond: wait until predicate() is true. False if the app is stopping or the wait timed out."""
        ready = self.cond.wait_for(lambda: self.stopping.is_set() or predicate(), timeout=WAIT_S)
        return ready and not self.stopping.is_set()

    def detection_loop(self):
        last_id = 0
        while not self.stopping.is_set():
            with self.cond:
                if not self.wait_for(lambda: self.frame_id > last_id):
                    continue
                frame, last_id, frame_time, metadata = self.frame, self.frame_id, self.frame_time, self.frame_metadata
            sent = time.perf_counter()
            self.conn.send(frame)
            detections, stage_ms = self.conn.recv()
            now = time.perf_counter()
            with self.cond:
                self.detections, self.detected_time = detections[detections[:, 4] >= self.args.conf], now
                self.detected_frame = (frame, frame_time, metadata, detections)
                self.detected_count += 1
                self.detected.add(now, {**stage_ms, "roundtrip": (now - sent) * 1000,
                                        "latency": (now - frame_time) * 1000})

    def render_loop(self):
        last_id = 0
        while not self.stopping.is_set():
            with self.cond:
                if not self.wait_for(lambda: self.viewers > 0 and self.frame_id > last_id):
                    continue
                frame, last_id, overlay = self.frame, self.frame_id, self.overlay()
            start = time.perf_counter()
            jpeg = self.encode(self.annotate(frame, *overlay))
            render_ms = (time.perf_counter() - start) * 1000
            with self.cond:
                self.jpeg = jpeg
                self.jpeg_id += 1
                self.rendered.add(time.perf_counter(), {"render": render_ms})
                self.cond.notify_all()

    def monitor_loop(self):
        detector_status = f"/proc/{self.process.pid}/status"
        next_log = time.perf_counter() + LOG_EVERY_S
        while True:
            state = throttled()
            system = {"temp_c": cpu_temp(), "arm_mhz": arm_clock_mhz(), "throttled": state}
            memory = {"app": proc_kib("/proc/self/status", "VmRSS", "VmHWM"),
                      "detector": proc_kib(detector_status, "VmRSS", "VmHWM"),
                      **proc_kib("/proc/meminfo", "MemAvailable")}
            disk_free = shutil.disk_usage(self.capture_dir).free
            now = time.perf_counter()
            undervoltage = undervoltage_now(state["raw"])
            with self.cond:
                self.system, self.memory, self.disk_free = system, memory, disk_free
                self.samples += 1
                self.undervoltage_samples += undervoltage
                self.capped_samples += "ARM frequency capped" in state["now"]
                stop_reason = self.guard.add(now, undervoltage)
            if stop_reason:
                log(f"{stop_reason}: the CPU drops to 600 MHz in each dip and a brownout reset may follow")
                self.stop(2, stop_reason)
                return
            if not self.ready:
                self.check_ready()
            if now >= next_log:
                log(self.status_line())
                next_log += LOG_EVERY_S
            if self.stopping.wait(MONITOR_S):
                return

    def check_ready(self):
        """Monitor thread: enable captures once the first detection is done and the clock is synchronised."""
        clock_synced = not self.args.wait_for_clock or CLOCK_SYNCED.exists()
        with self.cond:
            if clock_synced and not self.clock_synced:
                log("clock synchronised" if self.args.wait_for_clock else "clock: not checked (--no-wait-for-clock)")
            self.clock_synced = clock_synced
            if not clock_synced or self.detected_frame is None:
                return
            self.ready = True
            self.session = datetime.now().astimezone().isoformat(timespec="seconds")  # groups this run's captures
            if self.args.capture_every:
                self.next_timed = time.perf_counter() + self.args.capture_every * 60
        self.indicator.set_base("ready")
        log(f"ready: captures enabled, session {self.session}")

    def save_loop(self):
        while not self.stopping.is_set():
            timeout = WAIT_S if self.next_timed is None else min(WAIT_S, max(self.next_timed - time.perf_counter(), 0))
            try:
                request = self.capture_requests.get(timeout=timeout)
            except queue.Empty:
                if self.next_timed is None or time.perf_counter() < self.next_timed:
                    continue
                self.next_timed = time.perf_counter() + self.args.capture_every * 60
                request = CaptureRequest("timer")
            result = self.save_capture(request.source)
            if request.source != "timer":  # the LED answers presses (and page clicks), not the timer
                self.indicator.pattern("saved" if result["status"] == "saved" else "refused")
            request.finish(result)
        while True:  # answer requests that came in while stopping, so no web request waits for nothing
            try:
                self.capture_requests.get_nowait().finish({"status": "stopping", "message": "the app is stopping"})
            except queue.Empty:
                return

    def capture_blocked(self) -> str | None:
        """Under self.cond: why captures are paused (disk limits), or None."""
        args = self.args
        if self.capture_bytes > args.capture_max_mb * 1e6:
            return f"captures use {self.capture_bytes / 1e6:.1f} MB (limit {args.capture_max_mb:g} MB)"
        if self.disk_free is not None and self.disk_free < args.capture_min_free_mb * 1e6:
            return f"only {self.disk_free / 1e6:.0f} MB free on the SD card (limit {args.capture_min_free_mb:g} MB)"
        return None

    def save_capture(self, source: str) -> dict:
        """Save the frame of the latest detection with its detections as pre-labels -> result for the requester."""
        args = self.args
        now = time.perf_counter()
        with self.cond:
            self.capture_sources[source] += 1
            self.disk_free = shutil.disk_usage(self.capture_dir).free  # fresh: a capture must not fill the card
            blocked = self.capture_blocked()
            if self.last_capture_time is not None and now - self.last_capture_time < args.capture_min_interval:
                status, message = "too_soon", f"at most one capture per {args.capture_min_interval:g} s"
            elif blocked:
                status, message = "disk_limit", f"captures paused: {blocked}"
            elif not self.ready:
                status, message = "not_ready", ("waiting for the clock to synchronise" if not self.clock_synced
                                                else "no detection yet")
            else:
                status, message = "saved", None
                frame, frame_time, metadata, detections = self.detected_frame
                self.last_capture_time = now
            if status != "saved":
                self.capture_rejected[status] += 1
        if status != "saved":
            log(f"capture ({source}) rejected: {message}")
            return {"status": status, "message": message}

        taken = datetime.fromtimestamp(time.time() - (now - frame_time)).astimezone()  # when the frame was captured
        stem = f"{taken:%Y%m%d-%H%M%S}-{taken.microsecond // 1000:03d}_{source}"
        folder = self.capture_dir / f"{taken:%Y-%m-%d}"
        folder.mkdir(exist_ok=True)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.capture_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        prelabels = [{"class": self.names[int(class_id)], "class_id": int(class_id), "score": round(float(score), 3),
                      "box": [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                     for x1, y1, x2, y2, score, class_id in detections if score >= args.prelabel_conf]
        info = {
            "image": f"{stem}.jpg",
            "time": taken.isoformat(timespec="milliseconds"),
            "source": source,
            "session": self.session,  # when this run of the app enabled captures (one per power-on / camera spot)
            "width": frame.shape[1], "height": frame.shape[0],
            "frame_age_ms": round((now - frame_time) * 1000),  # frame captured -> capture requested
            "camera": {key: metadata[key] for key in CAMERA_KEYS if key in metadata},
            "model": {"name": Path(args.model).name, "imgsz": self.imgsz, "threads": args.threads,
                      "conf": args.conf, "prelabel_conf": args.prelabel_conf},
            "detections": prelabels,
        }
        image_path = folder / f"{stem}.jpg"
        write_file(image_path, jpeg.tobytes())
        sidecar = (json.dumps(info, indent=1) + "\n").encode()
        write_file(image_path.with_suffix(".json"), sidecar)

        summary = self.capture_summary(image_path, info)
        relative = summary["file"]
        with self.cond:
            self.capture_count += 1
            self.capture_files += 1
            self.capture_bytes += len(jpeg) + len(sidecar)
            self.last_capture_path, self.last_capture = image_path, summary
        found = ", ".join(f"{d['class']} {d['score']:.2f}" for d in prelabels) or "no cats"
        log(f"capture ({source}): {relative} ({len(jpeg) / 1024:.0f} KB, frame {info['frame_age_ms']} ms old; {found})")
        return {"status": "saved", **summary}

    def capture_summary(self, image_path: Path, info: dict) -> dict:
        """A capture as shown in /stats: file (relative to the captures folder), time, source, pre-labels."""
        return {"file": image_path.relative_to(self.capture_dir).as_posix(), "time": info["time"],
                "source": info["source"], "detections": info["detections"]}

    def request_capture(self, source: str) -> dict:
        """For the HTTP handler: queue a request and wait for the save thread's answer."""
        request = CaptureRequest(source)
        self.capture_requests.put(request)
        if not request.done.wait(CAPTURE_WAIT_S):
            return {"status": "timeout", "message": f"no answer from the save thread in {CAPTURE_WAIT_S} s"}
        return request.result

    # --- drawing ---

    def overlay(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Under self.cond: the latest detections, the status lines and the throttle warnings to draw."""
        now = time.perf_counter()
        infer_ms = self.detected.means().get("infer")
        status = [f"cam {self.captured.rate(now):.1f} fps  det {self.detected.rate(now):.1f} fps "
                  f"({fmt(infer_ms)} ms)",
                  f"{fmt(self.system.get('temp_c'))} C  {fmt(self.system.get('arm_mhz'))} MHz"]
        warnings = (self.system.get("throttled") or {}).get("now", [])
        return self.detections, status, warnings

    def annotate(self, frame: np.ndarray, detections: np.ndarray, status: list[str], warnings: list[str]
                 ) -> np.ndarray:
        image = frame.copy()
        for x1, y1, x2, y2, score, class_id in detections:
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{self.names[class_id]} {score:.2f}"
            (width, height), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            top = max(y1 - height - 8, 0)  # above the box, or inside it at the top of the frame
            cv2.rectangle(image, (x1, top), (x1 + width + 6, top + height + 8), color, cv2.FILLED)
            cv2.putText(image, label, (x1 + 3, top + height + 3), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        # status lines in the bottom-left corner, where they do not cover the labels (those sit on top of the boxes)
        lines = [(line, (255, 255, 255)) for line in status]
        if warnings:
            lines.append((", ".join(warnings).upper(), WARNING_COLOR))
        y = image.shape[0]
        for text, color in reversed(lines):
            (width, height), _ = cv2.getTextSize(text, FONT, 0.5, 1)
            cv2.rectangle(image, (0, y - height - 10), (width + 10, y), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, text, (5, y - 5), FONT, 0.5, color, 1, cv2.LINE_AA)
            y -= height + 10
        return image

    def encode(self, image: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()

    # --- for the HTTP handlers ---

    @contextmanager
    def viewer(self):
        """Count a stream viewer while the block runs (the render thread only works when someone watches)."""
        with self.cond:
            self.viewers += 1
            self.cond.notify_all()
        try:
            yield
        finally:
            with self.cond:
                self.viewers -= 1

    def next_jpeg(self, last_id: int) -> tuple[bytes | None, int]:
        """Wait for a JPEG newer than last_id -> (JPEG, its id); (None, last_id) when the app is stopping."""
        with self.cond:
            while not self.wait_for(lambda: self.jpeg_id > last_id):
                if self.stopping.is_set():
                    return None, last_id
            return self.jpeg, self.jpeg_id

    def snapshot(self) -> bytes | None:
        with self.cond:
            if self.frame is None:
                return None
            frame, overlay = self.frame, self.overlay()
        return self.encode(self.annotate(frame, *overlay))

    def stats(self) -> dict:
        now = time.perf_counter()
        with self.cond:
            undervoltage_s, dips = self.guard.window()
            return {
                "uptime_s": round(now - self.start_time, 1),
                "config": self.config,
                "capture": {"fps": round(self.captured.rate(now), 2), "frames": self.frame_id,
                            "age_ms": round((now - self.frame_time) * 1000) if self.frame_id else None},
                "detection": {"fps": round(self.detected.rate(now), 2), "frames": self.detected_count,
                              "age_ms": round((now - self.detected_time) * 1000) if self.detected_time else None,
                              "last_ms": self.detected.last(), "mean_ms": self.detected.means()},
                "stream": {"viewers": self.viewers, "fps": round(self.rendered.rate(now), 2), "frames": self.jpeg_id,
                           "render_ms": self.rendered.means().get("render")},
                "detections": [{"class": self.names[int(class_id)], "score": round(float(score), 3),
                                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                               for x1, y1, x2, y2, score, class_id in self.detections],
                "system": {**self.system,
                           "undervoltage_window": {"seconds": round(undervoltage_s, 1), "dips": dips,
                                                   "window_s": self.guard.window_s, "stop": self.guard.describe()},
                           "undervoltage_samples": self.undervoltage_samples, "capped_samples": self.capped_samples,
                           "samples": self.samples, "sample_every_s": MONITOR_S},
                "memory": self.memory,
                "captures": {
                    "ready": self.ready,
                    "clock": ("synchronised" if self.clock_synced else "waiting for NTP") if self.args.wait_for_clock
                             else "not checked",
                    "session": self.session,
                    "saved": self.capture_count,  # this run
                    "requests": dict(self.capture_sources), "rejected": dict(self.capture_rejected),
                    "last": self.last_capture,
                    "files": self.capture_files, "mb": round(self.capture_bytes / 1e6, 1),  # on disk, all runs
                    "free_mb": round(self.disk_free / 1e6) if self.disk_free is not None else None,
                    "paused": self.capture_blocked(),
                    "button": self.button_state,
                    "led": self.led_state,
                    "every_min": self.args.capture_every,
                    "next_timed_in_s": round(self.next_timed - now) if self.next_timed is not None else None,
                },
                "pids": {"app": os.getpid(), "detector": self.process.pid},
            }

    def last_capture_jpeg(self) -> bytes | None:
        with self.cond:
            path = self.last_capture_path
        try:
            return path.read_bytes() if path else None
        except OSError:  # deleted by hand
            return None

    def status_line(self) -> str:
        s = self.stats()
        detection, system, memory = s["detection"], s["system"], s["memory"]
        return (f"cam {s['capture']['fps']:.1f} fps, det {detection['fps']:.1f} fps "
                f"(infer {fmt(detection['mean_ms'].get('infer'))} ms, latency "
                f"{fmt(detection['mean_ms'].get('latency'))} ms), stream {s['stream']['fps']:.1f} fps to "
                f"{s['stream']['viewers']} viewer(s); {fmt(system.get('temp_c'), '.1f')} °C, "
                f"{fmt(system.get('arm_mhz'))} MHz, throttled {system['throttled']['raw']}; RSS app "
                f"{fmt(memory['app']['VmRSS'])} + detector {fmt(memory['detector']['VmRSS'])} MiB, "
                f"available {fmt(memory['MemAvailable'])} MiB; captures {s['captures']['saved']}")


# --- HTTP ------------------------------------------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cats live</title>
<style>
  body { margin: 0; background: #15171a; color: #e4e6ea; font: 14px/1.4 system-ui, sans-serif; }
  main { display: flex; flex-wrap: wrap; gap: 16px; padding: 16px; }
  img { max-width: 100%; height: auto; background: #000; }
  h1 { margin: 0 0 8px; font-size: 18px; }
  table { border-collapse: collapse; }
  td { padding: 2px 12px 2px 0; vertical-align: top; }
  td:first-child { color: #9aa0a8; }
  .warn { color: #ff6b6b; font-weight: bold; }
  h2 { margin: 16px 0 8px; font-size: 16px; }
  button { font: inherit; padding: 6px 18px; border: 0; border-radius: 4px; background: #3d7be0; color: #fff;
           cursor: pointer; }
  button:disabled { background: #555; cursor: wait; }
  #last { display: block; width: 320px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <img src="/stream.mjpg" alt="live camera stream">
  <section>
    <h1>Cats live</h1>
    <table id="stats"><tr><td>Loading…</td></tr></table>
    <h2>Captures</h2>
    <p><button id="capture" type="button">Capture</button> <span id="capture-msg"></span></p>
    <table id="captures"></table>
    <img id="last" alt="last capture" hidden>
  </section>
</main>
<script>
const f = (v, d = 0) => v == null ? "n/a" : Number(v).toFixed(d);
const esc = s => String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
const table = rows => rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
const cats = dets => dets.map(x => `${esc(x.class)} ${f(x.score, 2)}`).join(", ") || "no cats";
let lastFile = null;
function showCaptures(c) {
  const rows = [
    ["Ready", c.ready ? `yes, session ${esc(c.session)}`
                      : `<span class="warn">no: ${c.clock === "waiting for NTP" ? "waiting for the clock (NTP)"
                                                                               : "starting"}</span>`],
    ["Saved", `${c.saved} this run; ${c.files} on the Pi (${f(c.mb, 1)} MB), ${f(c.free_mb)} MB free` +
              (c.paused ? `<br><span class="warn">paused: ${esc(c.paused)}</span>` : "")],
    ["Last", c.last ? `${c.last.time.slice(11, 19)} ${esc(c.last.source)}: ${cats(c.last.detections)}` : "none"],
    ["Rejected", Object.entries(c.rejected).map(([k, n]) => `${esc(k)} ${n}`).join(", ") || "none"],
    ["Button", esc(c.button)],
    ["LED", esc(c.led)],
    ["Timer", !c.every_min ? "off" : `every ${c.every_min} min, ` + (c.next_timed_in_s == null ? "starts when ready"
                                     : `next in ${f(c.next_timed_in_s / 60, 1)} min`)],
  ];
  document.getElementById("captures").innerHTML = table(rows);
  const file = c.last ? c.last.file : null;
  if (file !== lastFile) {
    lastFile = file;
    const img = document.getElementById("last");
    img.hidden = !file;
    if (file) img.src = "/captures/last.jpg?f=" + encodeURIComponent(file);
  }
}
document.getElementById("capture").onclick = async e => {
  const msg = document.getElementById("capture-msg");
  e.target.disabled = true;
  try {
    const r = await (await fetch("/capture", {method: "POST"})).json();
    msg.innerHTML = r.status === "saved" ? `saved ${esc(r.file.split("/").pop())} (${cats(r.detections)})`
                                         : `<span class="warn">${esc(r.message || r.status)}</span>`;
  } catch (err) {
    msg.innerHTML = '<span class="warn">no answer from the Pi</span>';
  }
  e.target.disabled = false;
  refresh();
};
async function refresh() {
  let rows;
  try {
    const s = await (await fetch("/stats", {cache: "no-store"})).json();
    const d = s.detection, m = d.mean_ms, sys = s.system, mem = s.memory, t = sys.throttled || {};
    const uv = sys.undervoltage_window;
    const now = (t.now || []).join(", ");
    rows = [
      ["Camera", `${f(s.capture.fps, 1)} fps`],
      ["Detection", `${f(d.fps, 1)} fps, inference ${f(m.infer)} ms, latency ${f(m.latency)} ms`],
      ["Stream", `${f(s.stream.fps, 1)} fps, ${s.stream.viewers} viewer(s), render ${f(s.stream.render_ms)} ms`],
      ["Detections", s.detections.map(x => `${esc(x.class)} ${f(x.score, 2)}`).join(", ") || "none"],
      ["CPU", `${f(sys.temp_c, 1)} °C, ${f(sys.arm_mhz)} MHz`],
      ["Throttled", `${t.raw ?? "n/a"} ` + (now ? `<span class="warn">${esc(now)}</span>` : "(now: none)")],
      ["Since start", `under-voltage ${sys.undervoltage_samples}, capped ${sys.capped_samples} ` +
                      `of ${sys.samples} samples`],
      [`Last ${uv.window_s} s`, `under-voltage ${f(uv.seconds)} s in ${uv.dips} dip(s) (${esc(uv.stop)})`],
      ["Memory", `app ${f(mem.app?.VmRSS)} + detector ${f(mem.detector?.VmRSS)} MiB, ` +
                 `${f(mem.MemAvailable)} MiB available`],
      ["Model", `${esc(s.config.model)}, ${s.config.threads} threads, conf ${s.config.conf}`],
      ["Uptime", `${f(s.uptime_s / 60, 1)} min`],
    ];
    showCaptures(s.captures);
  } catch (e) {
    rows = [["Stats", '<span class="warn">no answer from the Pi</span>']];
  }
  document.getElementById("stats").innerHTML = table(rows);
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    timeout = 10  # seconds: a viewer that stops reading is dropped instead of blocking a thread forever

    def do_GET(self):
        app = self.server.app
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                self.send_body(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/stats":
                self.send_body(json.dumps(app.stats()).encode(), "application/json")
            elif path == "/snapshot.jpg":
                jpeg = app.snapshot()
                if jpeg is None:
                    self.send_error(503, "no camera frame yet")
                else:
                    self.send_body(jpeg, "image/jpeg")
            elif path == "/stream.mjpg":
                self.stream(app)
            elif path == "/captures/last.jpg":
                jpeg = app.last_capture_jpeg()
                if jpeg is None:
                    self.send_error(404, "no capture yet")
                else:
                    self.send_body(jpeg, "image/jpeg")
            elif path == "/favicon.ico":
                self.send_body(b"", "image/x-icon", 204)  # browsers ask for it; no 404 line in the log
            else:
                self.send_error(404)
        except (ConnectionError, TimeoutError):
            pass  # the client went away (e.g. a closed tab, or one that connected before the server was serving)

    def do_POST(self):
        app = self.server.app
        try:
            if self.path.split("?", 1)[0] == "/capture":
                result = app.request_capture("web")
                self.send_body(json.dumps(result).encode(), "application/json", CAPTURE_HTTP[result["status"]])
            else:
                self.send_error(404)
        except (ConnectionError, TimeoutError):
            pass

    def send_body(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream(self, app: App):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        viewer = f"{self.client_address[0]}:{self.client_address[1]}"
        log(f"stream: {viewer} connected")
        try:
            with app.viewer():
                last_id = 0
                while True:
                    jpeg, last_id = app.next_jpeg(last_id)  # always the newest: a slow viewer skips frames
                    if jpeg is None:
                        break
                    self.wfile.write(b"--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                                     % (BOUNDARY, len(jpeg), jpeg))
        finally:
            log(f"stream: {viewer} disconnected")

    def log_request(self, code="-", size="-"):
        pass  # no line per request (the page polls /stats every second); errors are still logged

    def log_message(self, format, *args):
        log(f"http: {self.client_address[0]} {format % args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/yolo26n_320_scale0.9", help="exported model folder")
    parser.add_argument("--conf", type=float, default=0.5, help="minimum detection score")
    parser.add_argument("--threads", type=int, default=2,
                        help="ncnn threads (2: more trigger under-voltage on the current supply, see phase 4)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640, help="camera frame width")
    parser.add_argument("--height", type=int, default=480, help="camera frame height")
    parser.add_argument("--stream-fps", type=float, default=15,
                        help="camera frame rate, and so the highest stream frame rate")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality of the stream (1-100)")
    parser.add_argument("--stop-on-undervoltage", type=float, default=10, metavar="S",
                        help="stop (exit code 2) when the last --undervoltage-window seconds hold S seconds of "
                             "under-voltage in total; 0 = never")
    parser.add_argument("--stop-on-dips", type=int, default=3, metavar="N",
                        help="same, when they hold N separate under-voltage dips; 0 = never")
    parser.add_argument("--undervoltage-window", type=float, default=60, metavar="S",
                        help="seconds looked at by the two limits above")
    captures = parser.add_argument_group("captures (training images)")
    captures.add_argument("--captures", default="captures", help="folder for the captures")
    captures.add_argument("--capture-quality", type=int, default=95, help="JPEG quality of the captures (1-100)")
    captures.add_argument("--prelabel-conf", type=float, default=0.25,
                          help="minimum score of the detections saved with a capture as pre-labels")
    captures.add_argument("--capture-min-interval", type=float, default=1, metavar="S",
                          help="at most one capture per S seconds (any source)")
    captures.add_argument("--capture-every", type=float, default=0, metavar="MIN",
                          help="timed capture every MIN minutes; 0 = off")
    captures.add_argument("--capture-max-mb", type=float, default=1000,
                          help="no more captures once the captures folder holds this many MB")
    captures.add_argument("--capture-min-free-mb", type=float, default=500,
                          help="no more captures when the SD card has less than this many MB free")
    captures.add_argument("--button-pin", type=int, default=25,
                          help="GPIO (BCM numbering) of the capture button; 0 = no button")
    captures.add_argument("--button-pull", choices=["up", "down"], default="up",
                          help="internal pull: up for a button to GND (the wiring on our Pi), down for one to 3.3 V")
    captures.add_argument("--button-bounce-ms", type=float, default=50, help="button debounce time in ms; 0 = off")
    captures.add_argument("--button-hold-s", type=float, default=5, metavar="S",
                          help="holding the button S seconds powers the Pi off (short presses then capture on "
                               "release); 0 = no power-off, capture on press")
    captures.add_argument("--led-pin", type=int, default=24,
                          help="GPIO (BCM numbering) of the status LED (220 ohm to GND); 0 = no LED")
    captures.add_argument("--wait-for-clock", action=argparse.BooleanOptionalAction, default=True,
                          help=f"no captures until the clock is NTP-synchronised ({CLOCK_SYNCED} exists)")
    args = parser.parse_args()
    if not 1 <= args.stream_fps <= 40:
        parser.error("--stream-fps must be between 1 and 40 (the 1640x1232 sensor mode tops out at ~41 fps)")
    if not 1 <= args.jpeg_quality <= 100 or not 1 <= args.capture_quality <= 100:
        parser.error("--jpeg-quality and --capture-quality must be between 1 and 100")

    app = App(args)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda num, _: app.stop(0, f"received {signal.Signals(num).name}"))
    try:
        app.start()
        while not app.stopping.wait(WAIT_S):
            pass
    except Exception:
        log(f"failed:\n{traceback.format_exc()}")
        app.exit_code = 1
    finally:
        app.shutdown()
    raise SystemExit(app.exit_code)


if __name__ == "__main__":
    main()
