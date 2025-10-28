#!/usr/bin/env python3
"""
YOLO Object Detection CLI

Usage examples:
  python detect.py --source path/to/image.jpg
  python detect.py --source path/to/images_dir --model yolov8s.pt --conf 0.4

This script uses the Ultralytics YOLO models to perform object detection on a
single image file or all images within a directory. Predictions and annotated
images are saved under the specified output directory.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List


def _list_images(path: Path) -> List[Path]:
    """
    Return a list of image files from a file or directory path.
    Supports common image extensions.
    """
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {path}")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in image_exts])


def run_detection(
    source: Path,
    output_dir: Path,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    device: str = "",
) -> None:
    """
    Run YOLO detection on images from `source` and save annotated outputs.
    Also prints a concise summary of detections per image.
    """
    try:
        # Import only when needed, so basic CLI help works without deps
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - import/runtime environment issue
        print(
            "Failed to import ultralytics. Did you install requirements?\n"
            "Try: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise

    output_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(source)
    if not images:
        print(f"No images found at: {source}")
        return

    model = YOLO(model_name)

    total_images = len(images)
    print(f"Running detection on {total_images} image(s) using {model_name}…")
    t0 = time.time()

    for idx, image_path in enumerate(images, start=1):
        results_list = model(
            str(image_path),
            conf=conf,
            device=device if device else None,
            verbose=False,
            save=True,
            project=str(output_dir),
            name="pred",
            exist_ok=True,
        )

        # Ultralytics returns a list of Results; for single image it's len==1
        results = results_list[0]

        names = results.names  # class id -> label
        boxes = results.boxes
        detected = []
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            for cls_id, score in zip(classes, confs):
                label = names.get(int(cls_id), str(int(cls_id)))
                detected.append((label, float(score)))

        print(f"[{idx}/{total_images}] {image_path.name}")
        if detected:
            # Aggregate by label with best score for brevity
            best_by_label = {}
            for label, score in detected:
                best_by_label[label] = max(score, best_by_label.get(label, 0.0))
            summary = ", ".join(f"{lbl} ({score:.2f})" for lbl, score in sorted(best_by_label.items()))
            print(f"  → {summary}")
        else:
            print("  → No objects detected above threshold")

    dt = time.time() - t0
    print(f"Done in {dt:.2f}s. Outputs saved under: {output_dir / 'pred'}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO object detection on images")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an image file or a directory of images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model to use (e.g., yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detect"),
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='Device to run on (e.g., "cpu", "0" for first GPU). Default: auto',
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    try:
        run_detection(
            source=args.source,
            output_dir=args.output,
            model_name=args.model,
            conf=args.conf,
            device=args.device,
        )
    except FileNotFoundError as not_found_err:
        print(str(not_found_err), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - surface unexpected issues
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


