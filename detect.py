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
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def collect_detections(
    source: Path,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    device: str = "",
    save_images: bool = False,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run YOLO on images from `source` and return structured detections.

    Each detection row has keys:
      - image: path to image
      - label: class name
      - confidence: float
      - x_min, y_min, x_max, y_max: bounding box in pixels
      - width, height: original image size
    If `save_images` is True, annotated images are saved under `output_dir`.
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

    if save_images and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(source)
    if not images:
        return []

    model = YOLO(model_name)
    rows: List[Dict[str, Any]] = []

    for image_path in images:
        results_list = model(
            str(image_path),
            conf=conf,
            device=device if device else None,
            verbose=False,
            save=bool(save_images),
            project=str(output_dir) if output_dir is not None else None,
            name="pred",
            exist_ok=True,
        )

        results = results_list[0]
        names = results.names
        boxes = results.boxes
        height, width = results.orig_shape if hasattr(results, "orig_shape") else (0, 0)

        if boxes is None or len(boxes) == 0:
            continue

        cls_list = boxes.cls.tolist()
        conf_list = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()

        for (cls_id, score, bb) in zip(cls_list, conf_list, xyxy):
            x_min, y_min, x_max, y_max = [int(v) for v in bb]
            label = names.get(int(cls_id), str(int(cls_id)))
            rows.append(
                {
                    "image": str(image_path),
                    "label": label,
                    "confidence": float(score),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": int(width),
                    "height": int(height),
                }
            )

    return rows


def write_csv(detections: List[Dict[str, Any]], csv_path: Path) -> None:
    """Write detections to CSV at `csv_path`."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detections:
            writer.writerow(row)


def run_detection(
    source: Path,
    output_dir: Path,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    device: str = "",
    csv_path: Optional[Path] = None,
) -> None:
    """
    CLI entry: run detection, save annotated images, print summary, optional CSV.
    """
    images = _list_images(source)
    if not images:
        print(f"No images found at: {source}")
        return

    total_images = len(images)
    print(f"Running detection on {total_images} image(s) using {model_name}…")
    t0 = time.time()

    detections = collect_detections(
        source=source,
        model_name=model_name,
        conf=conf,
        device=device,
        save_images=True,
        output_dir=output_dir,
    )

    # Print concise per-image summary
    by_image: Dict[str, Dict[str, float]] = {}
    for det in detections:
        image_path = Path(det["image"]).name
        label = det["label"]
        score = float(det["confidence"])
        best = by_image.setdefault(image_path, {})
        best[label] = max(score, best.get(label, 0.0))

    for idx, image_path in enumerate([p.name for p in images], start=1):
        print(f"[{idx}/{total_images}] {image_path}")
        best = by_image.get(image_path, {})
        if best:
            summary = ", ".join(
                f"{lbl} ({score:.2f})" for lbl, score in sorted(best.items())
            )
            print(f"  → {summary}")
        else:
            print("  → No objects detected above threshold")

    if csv_path is not None:
        write_csv(detections, csv_path)
        print(f"CSV saved to: {csv_path}")

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
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path to write detections",
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
            csv_path=args.csv,
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


