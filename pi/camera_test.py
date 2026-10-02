"""Camera smoke test: capture one frame with picamera2 the way the live app will, and save it.

Uses a full-field-of-view sensor mode (1640x1232 on the IMX219) and lets the ISP scale it down to the output size,
so no CPU time is spent resizing. Prints the selected sensor mode and crop so the field of view can be checked.

    python pi/camera_test.py --out camera_test.jpg
"""
import argparse
import time

import cv2
from picamera2 import Picamera2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="camera_test.jpg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sensor-width", type=int, default=1640)
    parser.add_argument("--sensor-height", type=int, default=1232)
    args = parser.parse_args()

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        sensor={"output_size": (args.sensor_width, args.sensor_height)},
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # let auto exposure / white balance settle

    start = time.perf_counter()
    frame = picam2.capture_array("main")
    capture_ms = (time.perf_counter() - start) * 1000
    metadata = picam2.capture_metadata()
    picam2.stop()

    # picamera2 "RGB888" frames are BGR in memory, which is what OpenCV expects.
    cv2.imwrite(args.out, frame)
    print(f"sensor:      {picam2.camera_configuration()['sensor']}")
    print(f"scaler crop: {metadata.get('ScalerCrop')}")
    print(f"frame:       shape={frame.shape} dtype={frame.dtype} capture={capture_ms:.1f} ms")
    print(f"saved:       {args.out}")


if __name__ == "__main__":
    main()
