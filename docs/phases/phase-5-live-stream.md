# Phase 5 — Live stream web app on the Pi

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 4 |

## Goal

Watch the camera in the browser with live detection boxes: `http://<pi-ip>:8000`.
Make it work correctly first; speed is phase 7.

## Design

Three parts sharing the latest state, so a slow detector does not freeze the video:

1. **Capture** — picamera2 with a full-field-of-view sensor mode (1640×1232), ISP-scaled main stream
   (e.g. 640×480). Keeps the latest frame.
2. **Detection** — runs `pi/detector.py` on the latest frame as fast as it can and keeps the latest detections.
3. **HTTP server** — standard library `ThreadingHTTPServer`:
   - `/` — small HTML page with the stream and live stats.
   - `/stream.mjpg` — `multipart/x-mixed-replace` MJPEG stream; each frame gets the latest boxes, labels, scores
     and FPS drawn with OpenCV, then JPEG-encoded.
   - `/stats` — JSON with capture FPS, detection FPS, inference time and current detections.

## Steps

- [ ] `pi/app.py` with CLI arguments: `--model`, `--conf`, `--threads`, `--port` (default 8000),
      `--width/--height`, `--stream-fps`, `--jpeg-quality`.
- [ ] Check the colour order from picamera2 (`RGB888` is BGR in memory) so boxes and colours are right and the
      detector gets the channel order it expects.
- [ ] Clean shutdown on Ctrl+C (stop camera, close server).
- [ ] Run it through `scripts/deploy.py` and test from the PC browser with both cats.
- [ ] Optional: a `systemd` service so the app starts on boot.

## Done when

- The browser shows a smooth live stream with boxes that follow Blacky and Niche and are labelled correctly.
- The app runs for at least 30 minutes without crashing or leaking memory.

## Handover notes

_None yet._
