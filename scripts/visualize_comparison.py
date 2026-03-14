#!/usr/bin/env python3
"""Visualize Rust vs Python detection comparison on COCO images.

Draws bounding boxes with class names and confidence scores side by side.
Highlights differences: FAIL (class mismatch), count diff, or PASS.

Prerequisites:
    1. Run: python3 scripts/validate_coco_images.py
    2. Run: cargo test --test coco_validation export_rust_detections -- --ignored

Usage:
    python3 scripts/visualize_comparison.py
    python3 scripts/visualize_comparison.py --all          # all 50 images
    python3 scripts/visualize_comparison.py --ids 007574 008690  # specific images
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "coco"
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "coco" / "visualizations"

# Color palette: distinct colors for different classes (BGR)
PALETTE = [
    (56, 56, 255),    # red
    (151, 157, 255),  # salmon
    (31, 112, 255),   # orange
    (29, 178, 255),   # yellow
    (49, 210, 207),   # gold
    (10, 249, 72),    # green
    (23, 204, 146),   # teal
    (134, 219, 61),   # lime
    (187, 212, 0),    # cyan
    (211, 188, 0),    # sky blue
    (255, 161, 6),    # blue
    (255, 75, 40),    # indigo
    (255, 0, 148),    # magenta
    (227, 76, 237),   # purple
    (160, 100, 255),  # violet
    (130, 150, 200),  # gray-blue
]


def get_color(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


def draw_detections(
    img: np.ndarray,
    detections: list,
    label_prefix: str = "",
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw bounding boxes with class labels and confidence on image."""
    overlay = img.copy()

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        conf = det["confidence"]
        cls_name = det["class_name"]
        cls_id = det["class_id"]
        color = get_color(cls_id)

        # Draw filled rectangle (semi-transparent)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Draw solid border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Label background
        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = max(y1, th + baseline + 4)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)

        # Label text (white on colored background)
        text_y = label_y1 + th + 2
        cv2.putText(img, label, (x1 + 2, text_y), font, font_scale, (255, 255, 255), thickness)

    # Blend overlay for semi-transparent boxes
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img


def create_comparison(
    image_id: str,
) -> tuple:
    """Create side-by-side comparison image for a COCO image."""
    img_path = FIXTURES_DIR / f"{image_id}.jpg"
    metadata_path = FIXTURES_DIR / f"{image_id}_metadata.json"
    rust_path = FIXTURES_DIR / f"{image_id}_rust_detections.json"

    if not all(p.exists() for p in [img_path, metadata_path, rust_path]):
        return None, {}

    img = cv2.imread(str(img_path))
    if img is None:
        return None, {}

    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(rust_path) as f:
        rust_data = json.load(f)

    python_dets = [d for d in metadata["python_detections"] if d["confidence"] >= 0.25]
    rust_dets = rust_data["rust_detections"]
    description = metadata.get("description", "")

    # Determine status
    py_top = python_dets[0]["class_name"] if python_dets else ""
    rs_top = rust_dets[0]["class_name"] if rust_dets else ""
    py_top_conf = python_dets[0]["confidence"] if python_dets else 0.0
    rs_top_conf = rust_dets[0]["confidence"] if rust_dets else 0.0

    if py_top == rs_top or (not python_dets and not rust_dets):
        status = "PASS"
    else:
        status = "FAIL"

    count_diff = len(rust_dets) - len(python_dets)

    info = {
        "image_id": image_id,
        "description": description,
        "status": status,
        "python_count": len(python_dets),
        "rust_count": len(rust_dets),
        "count_diff": count_diff,
        "python_top": f"{py_top} ({py_top_conf:.4f})",
        "rust_top": f"{rs_top} ({rs_top_conf:.4f})",
    }

    h, w = img.shape[:2]

    # Create two copies for drawing
    img_python = img.copy()
    img_rust = img.copy()

    img_python = draw_detections(img_python, python_dets)
    img_rust = draw_detections(img_rust, rust_dets)

    # Create header
    header_h = 60
    canvas_w = w * 2 + 20  # 20px gap
    canvas_h = h + header_h

    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)  # dark background

    # Status color
    if status == "FAIL":
        status_color = (60, 60, 230)  # red
    elif count_diff != 0:
        status_color = (50, 180, 230)  # yellow
    else:
        status_color = (80, 200, 80)  # green

    # Draw header background
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (45, 45, 45), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    title = f"[{image_id}] {description}"
    cv2.putText(canvas, title, (10, 22), font, 0.6, (220, 220, 220), 1)

    # Status badge
    badge_text = f"{status}  |  Python: {len(python_dets)} dets  |  Rust: {len(rust_dets)} dets  |  diff: {count_diff:+d}"
    cv2.putText(canvas, badge_text, (10, 48), font, 0.55, status_color, 1)

    # Python label
    py_label = f"Python (ultralytics)  top: {py_top} ({py_top_conf:.4f})"
    cv2.putText(canvas, py_label, (w // 2 - 100, 48), font, 0.45, (180, 180, 255), 1)

    # Rust label
    rs_label = f"Rust (candle)  top: {rs_top} ({rs_top_conf:.4f})"
    cv2.putText(canvas, rs_label, (w + 20 + w // 2 - 100, 48), font, 0.45, (180, 255, 180), 1)

    # Place images
    canvas[header_h:, :w] = img_python
    canvas[header_h:, w + 20 :] = img_rust

    # Column labels at top of images
    cv2.rectangle(canvas, (0, header_h), (w, header_h + 28), (40, 40, 120), -1)
    cv2.putText(canvas, "PYTHON", (w // 2 - 35, header_h + 20), font, 0.6, (200, 200, 255), 2)
    cv2.rectangle(canvas, (w + 20, header_h), (canvas_w, header_h + 28), (40, 120, 40), -1)
    cv2.putText(
        canvas, "RUST", (w + 20 + w // 2 - 25, header_h + 20), font, 0.6, (200, 255, 200), 2
    )

    return canvas, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Rust vs Python detection comparison")
    parser.add_argument("--all", action="store_true", help="Visualize all 50 images")
    parser.add_argument("--ids", nargs="+", help="Specific image IDs to visualize")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all available image IDs
    all_ids = sorted(
        [
            p.stem.replace("_metadata", "")
            for p in FIXTURES_DIR.glob("*_metadata.json")
            if (FIXTURES_DIR / f"{p.stem.replace('_metadata', '')}_rust_detections.json").exists()
        ]
    )

    if args.ids:
        # Pad to 12-digit format if needed
        target_ids = []
        for id_str in args.ids:
            padded = id_str.zfill(12)
            target_ids.append(padded)
    elif args.all:
        target_ids = all_ids
    else:
        # Default: show FAIL + count diff images
        target_ids = all_ids  # check all, filter interesting ones

    results = []
    interesting_ids = []

    for image_id in target_ids:
        canvas, info = create_comparison(image_id)
        if canvas is None:
            continue

        results.append(info)

        # Determine if this image is "interesting" (FAIL or count diff)
        is_interesting = info["status"] == "FAIL" or info["count_diff"] != 0

        if args.all or args.ids or is_interesting:
            out_path = OUTPUT_DIR / f"{image_id}_comparison.jpg"
            cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tag = f" [{info['status']}]" if info["status"] == "FAIL" else ""
            diff_str = f" (diff={info['count_diff']:+d})" if info["count_diff"] != 0 else ""
            print(f"  Saved: {out_path.name}{tag}{diff_str}")
            interesting_ids.append(image_id)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Total images processed: {len(results)}")
    print(f"Visualizations saved: {len(interesting_ids)}")
    print(f"Output directory: {OUTPUT_DIR}")

    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    diff_count = sum(1 for r in results if r["count_diff"] != 0)
    print(f"FAIL (class mismatch): {fail_count}")
    print(f"Count diff images: {diff_count}")

    if fail_count > 0:
        print("\nFAIL details:")
        for r in results:
            if r["status"] == "FAIL":
                print(
                    f"  [{r['image_id']}] {r['description']:20s} "
                    f"python={r['python_top']:30s} rust={r['rust_top']}"
                )

    if diff_count > 0:
        print("\nCount diff details:")
        for r in results:
            if r["count_diff"] != 0:
                print(
                    f"  [{r['image_id']}] {r['description']:20s} "
                    f"python={r['python_count']} rust={r['rust_count']} "
                    f"diff={r['count_diff']:+d}"
                )


if __name__ == "__main__":
    main()
