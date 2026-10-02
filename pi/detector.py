"""Cat detector for the Raspberry Pi: runs the NCNN export of the YOLO model with only ncnn and numpy.

The model folder comes from train/export.py (model.ncnn.param, model.ncnn.bin, model.json). The network takes a
letterboxed RGB image in [0, 1] and returns the raw YOLO head output: one column per anchor with
[cx, cy, w, h, score per class] in input pixels (sigmoid scores, no NMS yet). Pre- and postprocessing follow
Ultralytics (letterbox with grey 114 padding, best class per anchor, class-aware NMS at IoU 0.7), so the detections
match Ultralytics' predict on the same model.

    from detector import Detector
    detector = Detector("models/best_ncnn_model", conf=0.5, threads=4)
    for x1, y1, x2, y2, score, class_id in detector.detect(frame_bgr):
        print(detector.names[int(class_id)], score)

Command-line test (reads the images with OpenCV; the Detector class itself only needs ncnn and numpy):
    python pi/detector.py models/best_ncnn_model image.jpg [image.jpg ...] [--json detections.json]
"""
import json
from pathlib import Path

import ncnn
import numpy as np

GREY = 114  # letterbox padding value, as in Ultralytics
NORM = [1 / 255] * 3  # scale RGB to [0, 1]


class Detector:
    def __init__(self, model_dir, conf: float = 0.5, iou: float = 0.7, threads: int = 4, fp16: bool = False,
                 max_det: int = 300):
        """Load an exported model folder.

        conf: minimum score of a detection. iou: NMS IoU threshold. threads: CPU threads for ncnn.
        fp16: let ncnn store and compute in FP16 where the CPU supports it (ncnn's default on ARM); False keeps FP32.
        """
        model_dir = Path(model_dir)
        info = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        self.names = info["names"]
        self.input_height, self.input_width = info["imgsz"]
        self.conf, self.iou, self.max_det = conf, iou, max_det

        self.net = ncnn.Net()
        opt = self.net.opt  # options must be set before loading the model
        opt.use_vulkan_compute = False
        opt.num_threads = threads
        if not fp16:
            opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = False
            opt.use_bf16_packed = opt.use_bf16_storage = False
        self.net.load_param(str(model_dir / "model.ncnn.param"))
        self.net.load_model(str(model_dir / "model.ncnn.bin"))
        self.input_name = self.net.input_names()[0]
        self.output_name = self.net.output_names()[0]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect in a BGR uint8 frame -> (N, 6) float32 array of x1, y1, x2, y2, score, class_id in frame pixels."""
        mat, scale, pad = self.preprocess(frame)
        return self.postprocess(self.infer(mat), scale, pad, frame.shape[:2])

    def preprocess(self, frame: np.ndarray) -> tuple[ncnn.Mat, float, tuple[int, int]]:
        """Letterbox a BGR uint8 frame to the input size -> (input Mat, scale, (left, top) padding)."""
        height, width = frame.shape[:2]
        scale = min(self.input_height / height, self.input_width / width)
        new_width, new_height = round(width * scale), round(height * scale)
        pad_x, pad_y = (self.input_width - new_width) / 2, (self.input_height - new_height) / 2
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        # Bilinear resize and BGR -> RGB in one pass. from_pixels_resize copies the pixels into a new Mat (unlike
        # ncnn.Mat(array), which shares the array's memory and needs the array kept alive until extraction).
        mat = ncnn.Mat.from_pixels_resize(
            np.ascontiguousarray(frame), ncnn.Mat.PixelType.PIXEL_BGR2RGB, width, height, new_width, new_height
        )
        mat = ncnn.copy_make_border(mat, top, bottom, left, right, ncnn.BorderType.BORDER_CONSTANT, GREY)
        mat.substract_mean_normalize([], NORM)
        return mat, scale, (left, top)

    def infer(self, mat: ncnn.Mat) -> np.ndarray:
        """Run the network -> raw output (4 + classes, anchors)."""
        with self.net.create_extractor() as ex:
            ex.input(self.input_name, mat)
            ret, out = ex.extract(self.output_name)
            if ret != 0:
                raise RuntimeError(f"ncnn extract failed with code {ret}")
            return np.array(out)  # copy before the extractor is released

    def postprocess(self, output: np.ndarray, scale: float, pad: tuple[int, int], frame_shape: tuple[int, int]
                    ) -> np.ndarray:
        """Raw output -> (N, 6) x1, y1, x2, y2, score, class_id in frame pixels, best score first."""
        if output.shape[0] != 4 + len(self.names):
            raise ValueError(f"expected {4 + len(self.names)} output rows (cx, cy, w, h + class scores), "
                             f"got shape {output.shape}")
        scores = output[4:]
        class_ids = scores.argmax(axis=0)
        best = scores.max(axis=0)
        keep = best > self.conf
        cx, cy, w, h = output[:4, keep]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        detections = np.concatenate([boxes, best[keep, None], class_ids[keep, None]], axis=1).astype(np.float32)
        detections = detections[nms(detections, self.iou)[: self.max_det]]

        height, width = frame_shape
        detections[:, [0, 2]] = ((detections[:, [0, 2]] - pad[0]) / scale).clip(0, width)
        detections[:, [1, 3]] = ((detections[:, [1, 3]] - pad[1]) / scale).clip(0, height)
        return detections


def nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Class-aware greedy NMS on (N, 6) x1, y1, x2, y2, score, class_id -> kept indices, best score first.

    A box is dropped when a higher-scoring box of the same class overlaps it with IoU > iou_threshold (the rule of
    torchvision.ops.nms, which Ultralytics uses).
    """
    boxes, classes = detections[:, :4], detections[:, 5]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-detections[:, 4], kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        top_left = np.maximum(boxes[i, :2], boxes[rest, :2])
        bottom_right = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.clip(bottom_right - top_left, 0, None).prod(axis=1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[(iou <= iou_threshold) | (classes[rest] != classes[i])]
    return np.array(keep, dtype=np.int64)


def main():
    import argparse
    import time

    import cv2  # only to read the images in this command-line test

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="exported model folder (from train/export.py)")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="allow FP16 storage/arithmetic (default FP32)")
    parser.add_argument("--json", type=Path, help="also write the detections to this JSON file")
    args = parser.parse_args()

    detector = Detector(args.model_dir, conf=args.conf, threads=args.threads, fp16=args.fp16)
    results = {}
    for path in args.images:
        frame = cv2.imread(path)
        if frame is None:
            parser.error(f"cannot read image {path}")
        start = time.perf_counter()
        detections = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{path}: {len(detections)} detection(s) in {elapsed_ms:.1f} ms")
        for x1, y1, x2, y2, score, class_id in detections:
            print(f"  {detector.names[int(class_id)]:10} {score:.3f}  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        results[Path(path).name] = detections.round(3).tolist()
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"detections: {args.json}")


if __name__ == "__main__":
    main()
