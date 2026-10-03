"""Live stream web app for the Raspberry Pi: camera + cat detector, watched in the browser.

    python pi/app.py --model models/yolo26n_320_scale0.9

then open http://<pi-ip>:8000/ :
    /              page with the live stream and stats
    /stream.mjpg   MJPEG stream (multipart/x-mixed-replace) with the boxes, labels, scores and FPS drawn
    /snapshot.jpg  one annotated frame
    /stats         JSON: capture / detection / stream FPS, stage times, the current detections, CPU temperature, ARM
                   clock, `vcgencmd get_throttled` flags and memory

How it works (a slow detector must not freeze the video):
  - capture thread: picamera2 with the full-field-of-view sensor mode (1640x1232) scaled by the ISP to
    --width x --height, RGB888 (BGR in memory, which OpenCV and the detector expect), as in camera_test.py. The
    camera runs at --stream-fps; the thread keeps the latest frame.
  - detector process: ncnn holds Python's GIL while it runs (~210 ms per frame on the Pi), which would stall every
    other thread of the process, so pi/detector.py runs in a child process. The detection thread sends it the
    latest frame over a pipe and keeps the latest detections.
  - render thread: while someone watches the stream, draws the latest detections on each new frame and JPEG-encodes
    it, once for all viewers.
  - monitor thread: every 2 s reads the CPU temperature, ARM clock, throttle flags and memory. Sustained
    under-voltage caps the CPU at 600 MHz and can end in a brownout reset (phase 4), so after
    --stop-on-undervoltage seconds of it (default 30, 0 = never) the app stops with exit code 2.
  - HTTP: the standard library's ThreadingHTTPServer.

Ctrl+C or SIGTERM shuts down cleanly. Exit code: 0 after a normal stop, 1 when a part fails, 2 after sustained
under-voltage. Needs ncnn, numpy, OpenCV and picamera2 (no torch).
"""
import argparse
import json
import multiprocessing
import os
import signal
import socket
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from benchmark import arm_clock_mhz, cpu_temp, proc_kib, throttled, undervoltage_now
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


def log(message: str):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


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


class App:
    def __init__(self, args):
        self.args = args
        self.cond = threading.Condition()  # guards the state below; notified on every new frame and JPEG
        self.stopping = threading.Event()
        self.exit_code, self.stop_reason = 0, None
        self.start_time = time.perf_counter()
        self.config = {}
        self.names = []
        # latest camera frame (never modified: the render thread draws on a copy)
        self.frame, self.frame_id, self.frame_time = None, 0, 0.0
        self.captured = Window()
        # latest detections
        self.detections = np.zeros((0, 6), np.float32)
        self.detected_time, self.detected_count = None, 0
        self.detected = Window()
        # latest annotated JPEG for the stream
        self.jpeg, self.jpeg_id, self.viewers = None, 0, 0
        self.rendered = Window()
        # latest system sample
        self.system, self.memory = {}, {}
        self.samples = self.undervoltage_samples = self.capped_samples = 0
        self.undervoltage_since = None

        self.server = self.server_thread = self.process = self.conn = self.picam2 = None
        self.threads = []

    # --- start / stop ---

    def start(self):
        args = self.args
        self.server = ThreadingHTTPServer(("", args.port), Handler)  # bind first: fail fast if the port is taken
        self.server.app = self
        self.start_detector()
        self.start_camera()
        self.config = {"model": Path(args.model).name, "imgsz": self.imgsz, "threads": args.threads, "conf": args.conf,
                       "frame": [args.width, args.height], "sensor": list(SENSOR_SIZE),
                       "stream_fps": args.stream_fps, "jpeg_quality": args.jpeg_quality,
                       "stop_on_undervoltage_s": args.stop_on_undervoltage}
        for loop in (self.capture_loop, self.detection_loop, self.render_loop, self.monitor_loop):
            self.threads.append(self.start_thread(loop))
        self.server_thread = threading.Thread(target=self.server.serve_forever, args=(WAIT_S,), name="http",
                                              daemon=True)
        self.server_thread.start()
        log(f"serving on http://{socket.gethostname()}:{args.port}/ (Ctrl+C to stop)")

    def start_detector(self):
        args = self.args
        context = multiprocessing.get_context("spawn")  # a fresh interpreter: no camera or threads inherited
        self.conn, child_conn = context.Pipe()
        self.process = context.Process(target=detector_process, name="detector", daemon=True,
                                       args=(child_conn, args.model, args.conf, args.threads))
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
            f"conf {args.conf}, classes {', '.join(self.names)} (pid {self.process.pid})")

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
        log(f"stopped (exit code {self.exit_code})")

    # --- worker threads ---

    def capture_loop(self):
        while not self.stopping.is_set():
            frame = self.picam2.capture_array("main")  # a new array per frame, BGR in memory
            now = time.perf_counter()
            with self.cond:
                self.frame, self.frame_time = frame, now
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
                frame, last_id, frame_time = self.frame, self.frame_id, self.frame_time
            sent = time.perf_counter()
            self.conn.send(frame)
            detections, stage_ms = self.conn.recv()
            now = time.perf_counter()
            with self.cond:
                self.detections, self.detected_time = detections, now
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
            now = time.perf_counter()
            undervoltage = undervoltage_now(state["raw"])
            with self.cond:
                self.system, self.memory = system, memory
                self.samples += 1
                self.undervoltage_samples += undervoltage
                self.capped_samples += "ARM frequency capped" in state["now"]
                if not undervoltage:
                    self.undervoltage_since = None
                elif self.undervoltage_since is None:
                    self.undervoltage_since = now
                lasted = now - self.undervoltage_since if undervoltage else 0.0
            limit = self.args.stop_on_undervoltage
            if limit and undervoltage and lasted >= limit:
                log(f"under-voltage for {lasted:.0f} s: the CPU is capped at 600 MHz and a brownout reset may follow")
                self.stop(2, f"under-voltage for {lasted:.0f} s")
                return
            if now >= next_log:
                log(self.status_line())
                next_log += LOG_EVERY_S
            if self.stopping.wait(MONITOR_S):
                return

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
            undervoltage_for = now - self.undervoltage_since if self.undervoltage_since is not None else 0.0
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
                "system": {**self.system, "undervoltage_for_s": round(undervoltage_for, 1),
                           "undervoltage_samples": self.undervoltage_samples, "capped_samples": self.capped_samples,
                           "samples": self.samples, "sample_every_s": MONITOR_S},
                "memory": self.memory,
                "pids": {"app": os.getpid(), "detector": self.process.pid},
            }

    def status_line(self) -> str:
        s = self.stats()
        detection, system, memory = s["detection"], s["system"], s["memory"]
        return (f"cam {s['capture']['fps']:.1f} fps, det {detection['fps']:.1f} fps "
                f"(infer {fmt(detection['mean_ms'].get('infer'))} ms, latency "
                f"{fmt(detection['mean_ms'].get('latency'))} ms), stream {s['stream']['fps']:.1f} fps to "
                f"{s['stream']['viewers']} viewer(s); {fmt(system.get('temp_c'), '.1f')} °C, "
                f"{fmt(system.get('arm_mhz'))} MHz, throttled {system['throttled']['raw']}; RSS app "
                f"{fmt(memory['app']['VmRSS'])} + detector {fmt(memory['detector']['VmRSS'])} MiB, "
                f"available {fmt(memory['MemAvailable'])} MiB")


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
</style>
</head>
<body>
<main>
  <img src="/stream.mjpg" alt="live camera stream">
  <section>
    <h1>Cats live</h1>
    <table id="stats"><tr><td>Loading…</td></tr></table>
  </section>
</main>
<script>
const f = (v, d = 0) => v == null ? "n/a" : Number(v).toFixed(d);
const esc = s => String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
async function refresh() {
  let rows;
  try {
    const s = await (await fetch("/stats", {cache: "no-store"})).json();
    const d = s.detection, m = d.mean_ms, sys = s.system, mem = s.memory, t = sys.throttled || {};
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
      ["Memory", `app ${f(mem.app?.VmRSS)} + detector ${f(mem.detector?.VmRSS)} MiB, ` +
                 `${f(mem.MemAvailable)} MiB available`],
      ["Model", `${esc(s.config.model)}, ${s.config.threads} threads, conf ${s.config.conf}`],
      ["Uptime", `${f(s.uptime_s / 60, 1)} min`],
    ];
  } catch (e) {
    rows = [["Stats", '<span class="warn">no answer from the Pi</span>']];
  }
  document.getElementById("stats").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
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
            else:
                self.send_error(404)
        except (ConnectionError, TimeoutError):
            pass  # the client went away (e.g. a closed tab, or one that connected before the server was serving)

    def send_body(self, body: bytes, content_type: str):
        self.send_response(200)
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
    parser.add_argument("--stop-on-undervoltage", type=float, default=30, metavar="S",
                        help="stop (exit code 2) after S seconds of continuous under-voltage; 0 = never")
    args = parser.parse_args()
    if not 1 <= args.stream_fps <= 40:
        parser.error("--stream-fps must be between 1 and 40 (the 1640x1232 sensor mode tops out at ~41 fps)")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

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
