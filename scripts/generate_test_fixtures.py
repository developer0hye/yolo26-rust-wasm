#!/usr/bin/env python3
"""Generate reference fixtures for Rust-vs-Python comparison tests.

Runs YOLO26n inference on a test image using the original ultralytics
implementation and saves:
- tests/fixtures/test_input.safetensors        : preprocessed input [1, 3, 640, 640]
- tests/fixtures/reference_output.safetensors   : model output [1, 300, 6]
- tests/fixtures/reference_pre_topk.safetensors : pre-topk decoded output [1, 8400, 84]
- tests/fixtures/test_image_rgba.bin            : raw RGBA pixels for full pipeline test
- tests/fixtures/reference_metadata.json        : image dims + detection list

Requires: pip install ultralytics safetensors torch Pillow numpy
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from ultralytics import YOLO

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")


def create_test_image(width: int = 480, height: int = 640) -> Image.Image:
    """Create a deterministic test image with gradient + colored rectangles."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient background
    xs = np.arange(width, dtype=np.float32) / width * 255
    ys = np.arange(height, dtype=np.float32) / height * 255
    img[:, :, 0] = xs[np.newaxis, :].astype(np.uint8)
    img[:, :, 1] = ys[:, np.newaxis].astype(np.uint8)
    img[:, :, 2] = 128
    # Colored rectangles
    img[100:300, 50:200] = [255, 0, 0]
    img[200:400, 250:450] = [0, 255, 0]
    img[400:550, 100:350] = [0, 0, 255]
    return Image.fromarray(img)


def letterbox_preprocess(
    img_array: np.ndarray, target_size: int = 640
) -> tuple[torch.Tensor, dict]:
    """Replicate the Rust preprocessing exactly: nearest-neighbor letterbox, pad=114/255."""
    h, w = img_array.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    # Nearest-neighbor resize (matches Rust implementation)
    pil_img = Image.fromarray(img_array)
    resized = pil_img.resize((new_w, new_h), Image.NEAREST)
    resized_array = np.array(resized)

    # Padded canvas with value 114 (matches Rust pad_value = 114.0/255.0)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_array

    # HWC -> CHW, normalize to [0, 1]
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # CHW
    tensor = np.expand_dims(tensor, 0)  # BCHW

    letterbox_info = {
        "scale": float(scale),
        "pad_x": pad_x,
        "pad_y": pad_y,
        "new_w": new_w,
        "new_h": new_h,
    }
    # Convert via raw bytes to avoid NumPy version incompatibility with torch
    tensor_bytes = tensor.tobytes()
    t = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.float32).reshape(
        tensor.shape
    )
    return t, letterbox_info


def capture_pre_topk(model_obj: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run model backbone+neck+detect manually to get pre-topk decoded output [B, N, 84].

    Bypasses the topk postprocessing by reconstructing the detect head's
    intermediate computation.
    """
    detect_head = model_obj.model[-1]

    # Run through all layers except detect head to get feature maps
    x = input_tensor
    feature_inputs = []
    save_indices = set(detect_head.f) if hasattr(detect_head, "f") else set()

    intermediates = {}
    for i, m in enumerate(model_obj.model[:-1]):
        # Handle layers that take multiple inputs (concat, etc.)
        if hasattr(m, "f") and isinstance(m.f, list):
            x = m([intermediates.get(j, x) if j != -1 else x for j in m.f])
        elif hasattr(m, "f") and isinstance(m.f, int) and m.f != -1:
            x = m(intermediates[m.f])
        else:
            x = m(x)
        intermediates[i] = x

    # Get detect head's input feature maps
    feat_indices = detect_head.f if hasattr(detect_head, "f") else [-1]
    features = [intermediates[j] if j != -1 else x for j in feat_indices]

    # Run detect head's one2one branches manually
    bs = features[0].shape[0]
    all_boxes = []
    all_scores = []

    for i, feat in enumerate(features):
        # Box branch (one2one_cv2)
        bx = detect_head.one2one_cv2[i](feat)
        bx = bx.view(bs, -1, feat.shape[2] * feat.shape[3])  # [B, 4, H*W]
        all_boxes.append(bx)

        # Cls branch (one2one_cv3)
        cx = detect_head.one2one_cv3[i](feat)
        cx = cx.view(bs, detect_head.nc, feat.shape[2] * feat.shape[3])  # [B, nc, H*W]
        all_scores.append(cx)

    boxes = torch.cat(all_boxes, dim=-1)  # [B, 4, N]
    scores = torch.cat(all_scores, dim=-1)  # [B, nc, N]

    # Generate anchors and strides (reuse detect head's logic)
    feat_sizes = [(f.shape[2], f.shape[3]) for f in features]
    from ultralytics.utils.tal import make_anchors
    anchors, strides = (
        x.transpose(0, 1) for x in make_anchors(features, detect_head.stride, 0.5)
    )

    # dist2bbox xyxy (end2end mode): anchors are in grid coords, multiply by strides ONCE after
    from ultralytics.utils.tal import dist2bbox
    dbox = dist2bbox(detect_head.dfl(boxes), anchors.unsqueeze(0), xywh=False, dim=1)
    dbox = dbox * strides

    # Sigmoid scores
    cls_scores = scores.sigmoid()

    # Concat and permute: [B, 4+nc, N] -> [B, N, 4+nc]
    y = torch.cat([dbox, cls_scores], dim=1)  # [B, 84, N]
    pre_topk = y.permute(0, 2, 1)  # [B, N, 84]

    return pre_topk


def main() -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    # Create test image
    width, height = 480, 640
    img = create_test_image(width, height)
    img.save(os.path.join(FIXTURES_DIR, "test_image.png"))
    print(f"Test image: {width}x{height}")

    # Save RGBA pixels for full pipeline test
    rgba = np.array(img.convert("RGBA"))
    rgba_bytes = rgba.tobytes()
    rgba_path = os.path.join(FIXTURES_DIR, "test_image_rgba.bin")
    with open(rgba_path, "wb") as f:
        f.write(rgba_bytes)
    print(f"RGBA: {len(rgba_bytes)} bytes ({width}x{height}x4)")

    # Preprocess matching Rust implementation
    img_array = np.array(img)
    input_tensor, letterbox_info = letterbox_preprocess(img_array)
    print(f"Input tensor: {input_tensor.shape}, dtype={input_tensor.dtype}")

    # Load model
    model = YOLO("yolo26n.pt")
    model.model.eval()
    print(f"Model loaded: end2end={model.model.model[-1].end2end}")

    # Capture pre-topk output [1, 8400, 84]
    pre_topk = capture_pre_topk(model.model, input_tensor)
    print(f"Pre-topk tensor: {pre_topk.shape}")
    save_file(
        {"pre_topk": pre_topk.float().contiguous()},
        os.path.join(FIXTURES_DIR, "reference_pre_topk.safetensors"),
    )

    # Run inference for final output
    with torch.no_grad():
        result = model.model(input_tensor)

    # Extract output tensor
    if isinstance(result, tuple):
        output_tensor = result[0]  # Post-processed detections [B, 300, 6]
    else:
        output_tensor = result
    print(f"Output tensor: {output_tensor.shape}, dtype={output_tensor.dtype}")

    # Save tensors as SafeTensors
    save_file(
        {"input": input_tensor},
        os.path.join(FIXTURES_DIR, "test_input.safetensors"),
    )
    save_file(
        {"output": output_tensor.float()},
        os.path.join(FIXTURES_DIR, "reference_output.safetensors"),
    )

    # Build detection list from output
    output_t = output_tensor.squeeze(0).detach().cpu()  # [300, 6]
    detections = []
    for i in range(output_t.shape[0]):
        x1, y1, x2, y2, conf, cls_id = [output_t[i][j].item() for j in range(6)]
        if conf > 0.001:
            detections.append(
                {
                    "x1": round(float(x1), 4),
                    "y1": round(float(y1), 4),
                    "x2": round(float(x2), 4),
                    "y2": round(float(y2), 4),
                    "confidence": round(float(conf), 6),
                    "class_id": int(cls_id),
                }
            )

    metadata = {
        "image_width": width,
        "image_height": height,
        "letterbox": letterbox_info,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output_tensor.shape),
        "pre_topk_shape": list(pre_topk.shape),
        "total_output_rows": int(output_t.shape[0]),
        "detections_above_0001": len(detections),
        "detections": detections,
    }
    with open(os.path.join(FIXTURES_DIR, "reference_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    max_conf = max((d["confidence"] for d in detections), default=0.0)
    print(f"Detections (conf>0.001): {len(detections)}")
    print(f"Max confidence: {max_conf:.4f}")
    print("Top-5 detections:")
    for d in sorted(detections, key=lambda d: -d["confidence"])[:5]:
        print(
            f"  class={d['class_id']} conf={d['confidence']:.4f} "
            f"box=({d['x1']:.1f},{d['y1']:.1f},{d['x2']:.1f},{d['y2']:.1f})"
        )


if __name__ == "__main__":
    main()
