#!/usr/bin/env python3
"""Validate YOLO26 Rust WASM model against Python reference using real COCO images.

Downloads 16 diverse COCO val2017 images, runs Python (ultralytics) inference,
and saves fixtures for Rust comparison tests.

Requires: pip install ultralytics Pillow numpy requests

Usage:
    python3 scripts/validate_coco_images.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "coco"
MODEL_PATH = "yolo26n.pt"

# 50 diverse COCO val2017 images covering all major categories
COCO_IMAGES = [
    # --- Original 16 ---
    {"id": "000000000139", "description": "living room with TV"},
    {"id": "000000000285", "description": "giraffes"},
    {"id": "000000000632", "description": "tennis player"},
    {"id": "000000000776", "description": "people on beach"},
    {"id": "000000001000", "description": "plates of food"},
    {"id": "000000001268", "description": "motorcycles"},
    {"id": "000000001503", "description": "dogs"},
    {"id": "000000001761", "description": "people on street"},
    {"id": "000000002006", "description": "cat on couch"},
    {"id": "000000002149", "description": "bus"},
    {"id": "000000002299", "description": "kitchen"},
    {"id": "000000002431", "description": "horse"},
    {"id": "000000002473", "description": "airplane"},
    {"id": "000000002532", "description": "skateboarding"},
    {"id": "000000002587", "description": "dining table"},
    {"id": "000000002685", "description": "bird"},
    # --- Additional 34: more categories and scene diversity ---
    {"id": "000000003156", "description": "train on tracks"},
    {"id": "000000003501", "description": "zebras in field"},
    {"id": "000000003553", "description": "boat on water"},
    {"id": "000000003934", "description": "surfing"},
    {"id": "000000004134", "description": "snowboarding"},
    {"id": "000000004395", "description": "baseball game"},
    {"id": "000000005037", "description": "elephant"},
    {"id": "000000005193", "description": "bear"},
    {"id": "000000005477", "description": "truck on road"},
    {"id": "000000005529", "description": "bicycle parked"},
    {"id": "000000006040", "description": "fire hydrant"},
    {"id": "000000006471", "description": "stop sign"},
    {"id": "000000006763", "description": "parking meter"},
    {"id": "000000006818", "description": "bench in park"},
    {"id": "000000007108", "description": "sheep grazing"},
    {"id": "000000007278", "description": "cow in field"},
    {"id": "000000007386", "description": "umbrella scene"},
    {"id": "000000007574", "description": "backpack"},
    {"id": "000000007784", "description": "kite flying"},
    {"id": "000000007816", "description": "sports ball"},
    {"id": "000000007977", "description": "frisbee"},
    {"id": "000000008021", "description": "skis on snow"},
    {"id": "000000008211", "description": "tennis racket"},
    {"id": "000000008532", "description": "bottles and cups"},
    {"id": "000000008690", "description": "pizza"},
    {"id": "000000008844", "description": "sandwich"},
    {"id": "000000009448", "description": "cake"},
    {"id": "000000009483", "description": "toilet"},
    {"id": "000000009590", "description": "clock on wall"},
    {"id": "000000009769", "description": "potted plant"},
    {"id": "000000009891", "description": "couch with pillows"},
    {"id": "000000009914", "description": "refrigerator"},
    {"id": "000000010092", "description": "vase with flowers"},
    {"id": "000000010977", "description": "traffic light"},
]

COCO_VAL_URL = "http://images.cocodataset.org/val2017/{image_id}.jpg"


def download_image(image_id: str) -> Image.Image:
    """Download a COCO val2017 image by ID."""
    url = COCO_VAL_URL.format(image_id=image_id)
    cache_path = FIXTURES_DIR / f"{image_id}.jpg"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    print(f"  Downloading {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return Image.open(cache_path).convert("RGB")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    summary = {"images": [], "total_python_detections": 0}

    for entry in COCO_IMAGES:
        image_id = entry["id"]
        desc = entry["description"]
        print(f"\n[{image_id}] {desc}")

        # Download and open image
        try:
            img = download_image(image_id)
        except Exception as e:
            print(f"  SKIP: download failed: {e}")
            continue

        width, height = img.size
        print(f"  Size: {width}x{height}")

        # Save RGBA bytes for Rust pipeline test
        rgba = np.array(img.convert("RGBA"))
        rgba_path = FIXTURES_DIR / f"{image_id}_rgba.bin"
        rgba_path.write_bytes(rgba.tobytes())

        # Run Python inference (conf=0.001 to get all detections for flexible threshold testing)
        results = model(img, conf=0.001, verbose=False)
        result = results[0]

        # Extract detections
        python_detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = result.names[cls_id]
                python_detections.append(
                    {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "confidence": round(conf, 6),
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )

        # Sort by confidence descending
        python_detections.sort(key=lambda d: -d["confidence"])

        dets_above_025 = [d for d in python_detections if d["confidence"] >= 0.25]
        print(f"  Detections (conf>=0.25): {len(dets_above_025)}")
        for d in dets_above_025[:5]:
            print(
                f"    {d['class_name']:15s} conf={d['confidence']:.4f} "
                f"box=({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})"
            )
        if len(dets_above_025) > 5:
            print(f"    ... and {len(dets_above_025) - 5} more")

        # Save metadata
        metadata = {
            "image_id": image_id,
            "description": desc,
            "width": width,
            "height": height,
            "python_detections": python_detections,
            "num_detections_above_025": len(dets_above_025),
        }
        metadata_path = FIXTURES_DIR / f"{image_id}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        summary["images"].append(
            {
                "image_id": image_id,
                "description": desc,
                "width": width,
                "height": height,
                "num_detections_above_025": len(dets_above_025),
                "top_class": dets_above_025[0]["class_name"] if dets_above_025 else None,
                "top_confidence": (
                    dets_above_025[0]["confidence"] if dets_above_025 else 0.0
                ),
            }
        )
        summary["total_python_detections"] += len(dets_above_025)

    # Save summary
    summary_path = FIXTURES_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Total images: {len(summary['images'])}")
    print(f"Total detections (conf>=0.25): {summary['total_python_detections']}")
    print(f"Fixtures saved to: {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
