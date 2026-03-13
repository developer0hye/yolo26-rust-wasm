# YOLO26 Rust WASM Demo Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based YOLO26 object detection demo where all inference runs inside a Rust WASM module (tract ONNX runtime), served from a single `index.html`.

**Architecture:** Rust WASM module (tract-onnx) exports `init_model` and `detect` functions via wasm-bindgen. JS handles image decoding (Canvas API), WASM handles preprocess/inference/postprocess, JS renders bounding boxes on Canvas.

**Tech Stack:** Rust, wasm-pack, tract-onnx, wasm-bindgen, serde_json, vanilla JS, Canvas API

**Spec:** `docs/superpowers/specs/2026-03-13-yolo26-rust-wasm-design.md`

---

## Chunk 0: Branch Setup

### Task 0: Create Feature Branch via Git Worktree

Per `CLAUDE.md`: all changes go through PRs, use git worktrees for branch work.

- [ ] **Step 1: Create worktree and feature branch**

Run: `git worktree add ../yolo26-rust-wasm-feat-initial-implementation -b feat/initial-implementation`
Run: `cd ../yolo26-rust-wasm-feat-initial-implementation`

All subsequent tasks execute from inside this worktree directory.

---

## Chunk 1: Project Scaffolding & Preprocessing

### Task 1: Project Setup (Cargo.toml, .gitignore, model download)

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs` (minimal placeholder)
- Modify: `.gitignore`
- Create: `scripts/download_model.sh`
- Modify: `FORMATTING.md`

- [ ] **Step 1: Create Cargo.toml with dependencies**

```toml
[package]
name = "yolo26-rust-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tract-onnx = { version = "0.21", features = ["getrandom-js"] }
console_error_panic_hook = "0.1"
web-sys = { version = "0.3", features = ["console"] }
getrandom = { version = "0.2", features = ["js"] }

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
lto = true
opt-level = "z"
strip = true
```

- [ ] **Step 2: Create minimal src/lib.rs**

```rust
use wasm_bindgen::prelude::*;

/// Initialize panic hook for readable WASM error messages.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
```

- [ ] **Step 3: Update .gitignore**

Add these entries to `.gitignore`:
```
/target
/pkg
/weights/*.onnx
/weights/*.safetensors
```

- [ ] **Step 4: Create scripts/download_model.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
MODEL_PATH="${WEIGHTS_DIR}/yolo26n.onnx"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at ${MODEL_PATH}"
    exit 0
fi

mkdir -p "$WEIGHTS_DIR"

echo "Exporting YOLO26n to ONNX via ultralytics..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='onnx', simplify=True, opset=17, imgsz=640)
"

# ultralytics exports next to the .pt file (current directory)
if [ -f "yolo26n.onnx" ]; then
    mv "yolo26n.onnx" "$MODEL_PATH"
fi

echo "Model saved to ${MODEL_PATH}"
echo "Size: $(du -h "$MODEL_PATH" | cut -f1)"
```

- [ ] **Step 5: Make download script executable**

Run: `chmod +x scripts/download_model.sh`

- [ ] **Step 6: Update FORMATTING.md**

Set the primary command to `cargo fmt --all`.

- [ ] **Step 7: Verify project compiles**

Run: `cargo check`
Expected: Success (no errors)

- [ ] **Step 8: Verify WASM build works**

Run: `wasm-pack build --target web --dev`
Expected: Success, `pkg/` directory created with `.wasm` file

**RISK CHECK — tract + wasm-bindgen compatibility:**
tract's `inventory` crate may crash under wasm-bindgen at runtime (see [sonos/tract#2001](https://github.com/sonos/tract/issues/2001)).
If the WASM build succeeds but crashes at runtime in the browser with an indirect call error,
the fallback is to switch to **rten** (lightweight ONNX runtime with native WASM support):
- Replace `tract-onnx` with `rten = "0.14"` in Cargo.toml
- Adapt `model.rs` to use rten's API (similar pattern: load bytes → run inference)
- Postprocessing remains the same (tensor output format is model-dependent, not runtime-dependent)

- [ ] **Step 9: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add Cargo.toml src/lib.rs .gitignore scripts/download_model.sh FORMATTING.md`
Run: `git commit -s -m "feat: scaffold project with Cargo.toml, wasm-bindgen entry point, model download script"`

---

### Task 2: Preprocessing - RGBA to RGB conversion

**Files:**
- Create: `src/preprocess.rs`
- Modify: `src/lib.rs` (add `mod preprocess;`)

- [ ] **Step 1: Write failing test for RGBA to RGB conversion**

Create `src/preprocess.rs`:

```rust
/// Image preprocessing pipeline: RGBA pixels → model input tensor.

/// Convert RGBA pixel buffer to RGB by dropping every 4th byte (alpha).
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_to_rgb_basic() {
        // Two pixels: red (255,0,0,255) and blue (0,0,255,128)
        let rgba = vec![255, 0, 0, 255, 0, 0, 255, 128];
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![255, 0, 0, 0, 0, 255]);
    }

    #[test]
    fn test_rgba_to_rgb_empty() {
        let rgba: Vec<u8> = vec![];
        let rgb = rgba_to_rgb(&rgba);
        assert!(rgb.is_empty());
    }

    #[test]
    fn test_rgba_to_rgb_single_pixel() {
        let rgba = vec![128, 64, 32, 200];
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![128, 64, 32]);
    }
}
```

Add `mod preprocess;` to `src/lib.rs`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_rgba_to_rgb -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement rgba_to_rgb**

```rust
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let pixel_count = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    rgb
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test test_rgba_to_rgb -- --nocapture`
Expected: All 3 tests PASS

- [ ] **Step 5: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/preprocess.rs src/lib.rs`
Run: `git commit -s -m "feat: add RGBA to RGB conversion in preprocess module"`

---

### Task 3: Preprocessing - Letterbox Resize

**Files:**
- Modify: `src/preprocess.rs`

- [ ] **Step 1: Write LetterboxInfo struct and failing test for letterbox resize**

Add to `src/preprocess.rs`:

```rust
/// Metadata from letterbox resize, needed to reverse-transform coordinates in postprocessing.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    /// Scale factor applied to the image (ratio of model_size / max(original_w, original_h))
    pub scale: f32,
    /// Horizontal padding in pixels (in model space)
    pub pad_x: f32,
    /// Vertical padding in pixels (in model space)
    pub pad_y: f32,
}

const MODEL_INPUT_SIZE: u32 = 640;
const LETTERBOX_PAD_VALUE: f32 = 114.0 / 255.0;

/// Resize RGB image to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE with letterboxing.
/// Returns (resized_rgb_f32, letterbox_info).
/// The output is normalized to [0.0, 1.0] and remains in HWC layout.
pub fn letterbox_resize(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> (Vec<f32>, LetterboxInfo) {
    todo!()
}
```

Add tests:

```rust
    #[test]
    fn test_letterbox_square_image() {
        // 2x2 white image → 640x640, no padding needed (pad_x=0, pad_y=0)
        let rgb = vec![255u8; 2 * 2 * 3];
        let (result, info) = letterbox_resize(&rgb, 2, 2);
        assert_eq!(result.len(), (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3) as usize);
        assert!((info.pad_x - 0.0).abs() < 1e-3);
        assert!((info.pad_y - 0.0).abs() < 1e-3);
        assert!((info.scale - 320.0).abs() < 1e-3); // 640 / 2 = 320
    }

    #[test]
    fn test_letterbox_wide_image() {
        // 640x320 image → scale to fit 640 wide, pad top/bottom
        let rgb = vec![128u8; 640 * 320 * 3];
        let (result, info) = letterbox_resize(&rgb, 640, 320);
        assert_eq!(result.len(), (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3) as usize);
        assert!((info.scale - 1.0).abs() < 1e-3); // 640/640 = 1.0
        assert!((info.pad_x - 0.0).abs() < 1e-3);
        assert!((info.pad_y - 160.0).abs() < 1e-3); // (640-320)/2 = 160
    }

    #[test]
    fn test_letterbox_tall_image() {
        // 320x640 image → scale to fit 640 tall, pad left/right
        let rgb = vec![64u8; 320 * 640 * 3];
        let (result, info) = letterbox_resize(&rgb, 320, 640);
        assert_eq!(result.len(), (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3) as usize);
        assert!((info.scale - 1.0).abs() < 1e-3); // 640/640 = 1.0
        assert!((info.pad_x - 160.0).abs() < 1e-3); // (640-320)/2 = 160
        assert!((info.pad_y - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_letterbox_padding_value() {
        // 1x1 black pixel → entire image should be black pixel scaled + gray padding
        let rgb = vec![0u8; 1 * 1 * 3];
        let (result, info) = letterbox_resize(&rgb, 1, 1);
        // The padding area should be LETTERBOX_PAD_VALUE
        // Check a pixel that is definitely in the padding area (last row, last pixel)
        let last_pixel_idx = ((MODEL_INPUT_SIZE * MODEL_INPUT_SIZE - 1) * 3) as usize;
        let pad_val = LETTERBOX_PAD_VALUE;
        // Corner pixel should be padding value (gray)
        assert!((result[last_pixel_idx] - pad_val).abs() < 1e-2);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test test_letterbox -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement letterbox_resize**

```rust
pub fn letterbox_resize(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> (Vec<f32>, LetterboxInfo) {
    let model_size = MODEL_INPUT_SIZE as f32;
    let w = width as f32;
    let h = height as f32;

    // Compute scale to fit the larger dimension into model_size
    let scale = model_size / w.max(h);
    let new_w = (w * scale).round() as u32;
    let new_h = (h * scale).round() as u32;

    let pad_x = (MODEL_INPUT_SIZE - new_w) as f32 / 2.0;
    let pad_y = (MODEL_INPUT_SIZE - new_h) as f32 / 2.0;

    let pad_x_left = pad_x.floor() as u32;
    let pad_y_top = pad_y.floor() as u32;

    // Initialize output with padding value (normalized gray)
    let total_pixels = (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3) as usize;
    let mut output = vec![LETTERBOX_PAD_VALUE; total_pixels];

    // Bilinear interpolation resize + normalize + place in padded output
    for dst_y in 0..new_h {
        for dst_x in 0..new_w {
            // Map destination pixel to source pixel
            let src_x = dst_x as f32 * (w / new_w as f32);
            let src_y = dst_y as f32 * (h / new_h as f32);

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let x_frac = src_x - x0 as f32;
            let y_frac = src_y - y0 as f32;

            let out_y = dst_y + pad_y_top;
            let out_x = dst_x + pad_x_left;
            let out_idx = ((out_y * MODEL_INPUT_SIZE + out_x) * 3) as usize;

            for c in 0..3 {
                let v00 = rgb[((y0 * width + x0) * 3 + c) as usize] as f32;
                let v01 = rgb[((y0 * width + x1) * 3 + c) as usize] as f32;
                let v10 = rgb[((y1 * width + x0) * 3 + c) as usize] as f32;
                let v11 = rgb[((y1 * width + x1) * 3 + c) as usize] as f32;

                let value = v00 * (1.0 - x_frac) * (1.0 - y_frac)
                    + v01 * x_frac * (1.0 - y_frac)
                    + v10 * (1.0 - x_frac) * y_frac
                    + v11 * x_frac * y_frac;

                output[out_idx + c as usize] = value / 255.0;
            }
        }
    }

    let info = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };
    (output, info)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test test_letterbox -- --nocapture`
Expected: All 4 tests PASS

- [ ] **Step 5: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/preprocess.rs`
Run: `git commit -s -m "feat: add letterbox resize with bilinear interpolation"`

---

### Task 4: Preprocessing - HWC to CHW + Batch Dimension

**Files:**
- Modify: `src/preprocess.rs`

- [ ] **Step 1: Write failing test for HWC→CHW conversion**

Add to `src/preprocess.rs`:

```rust
/// Convert HWC float buffer to CHW layout and add batch dimension.
/// Input: [H*W*3] f32 in HWC order
/// Output: [1*3*H*W] f32 in NCHW order
pub fn hwc_to_nchw(hwc: &[f32], height: u32, width: u32) -> Vec<f32> {
    todo!()
}
```

Add tests:

```rust
    #[test]
    fn test_hwc_to_nchw_2x2() {
        // 2x2 image with distinct channel values per pixel
        // Pixel layout (HWC): [R0,G0,B0, R1,G1,B1, R2,G2,B2, R3,G3,B3]
        let hwc = vec![
            0.1, 0.2, 0.3,  // pixel (0,0)
            0.4, 0.5, 0.6,  // pixel (0,1)
            0.7, 0.8, 0.9,  // pixel (1,0)
            1.0, 0.0, 0.5,  // pixel (1,1)
        ];
        let nchw = hwc_to_nchw(&hwc, 2, 2);

        // Expected NCHW: [batch=1, C=3, H=2, W=2]
        // Channel R: [0.1, 0.4, 0.7, 1.0]
        // Channel G: [0.2, 0.5, 0.8, 0.0]
        // Channel B: [0.3, 0.6, 0.9, 0.5]
        let expected = vec![
            0.1, 0.4, 0.7, 1.0,  // R
            0.2, 0.5, 0.8, 0.0,  // G
            0.3, 0.6, 0.9, 0.5,  // B
        ];
        assert_eq!(nchw.len(), expected.len());
        for (a, b) in nchw.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_hwc_to_nchw_output_length() {
        let hwc = vec![0.0f32; 640 * 640 * 3];
        let nchw = hwc_to_nchw(&hwc, 640, 640);
        assert_eq!(nchw.len(), 3 * 640 * 640);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test test_hwc_to_nchw -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement hwc_to_nchw**

```rust
pub fn hwc_to_nchw(hwc: &[f32], height: u32, width: u32) -> Vec<f32> {
    let h = height as usize;
    let w = width as usize;
    let mut nchw = vec![0.0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let hwc_idx = (y * w + x) * 3;
            for c in 0..3 {
                let nchw_idx = c * h * w + y * w + x;
                nchw[nchw_idx] = hwc[hwc_idx + c];
            }
        }
    }

    nchw
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test test_hwc_to_nchw -- --nocapture`
Expected: All 2 tests PASS

- [ ] **Step 5: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/preprocess.rs`
Run: `git commit -s -m "feat: add HWC to NCHW layout conversion"`

---

### Task 5: Preprocessing - Full Pipeline Function

**Files:**
- Modify: `src/preprocess.rs`

- [ ] **Step 1: Write failing test for full preprocess pipeline**

Add to `src/preprocess.rs`:

```rust
use tract_onnx::prelude::*;

/// Full preprocessing pipeline: RGBA pixels → tract Tensor [1, 3, 640, 640].
/// Returns the tensor and letterbox info for postprocess coordinate reversal.
pub fn preprocess(
    rgba_pixels: &[u8],
    width: u32,
    height: u32,
) -> (Tensor, LetterboxInfo) {
    todo!()
}
```

Add tests:

```rust
    #[test]
    fn test_preprocess_output_shape() {
        // 4x3 RGBA image (arbitrary)
        let rgba = vec![128u8; 4 * 3 * 4]; // 4 wide, 3 tall, RGBA
        let (tensor, info) = preprocess(&rgba, 4, 3);

        assert_eq!(tensor.shape(), &[1, 3, 640, 640]);
        assert_eq!(tensor.datum_type(), f32::datum_type());
        assert!(info.scale > 0.0);
    }

    #[test]
    fn test_preprocess_values_in_range() {
        let rgba = vec![200u8; 100 * 80 * 4];
        let (tensor, _info) = preprocess(&rgba, 100, 80);

        let data = tensor.as_slice::<f32>().unwrap();
        for &v in data {
            assert!(v >= 0.0 && v <= 1.0, "value out of range: {}", v);
        }
    }

    #[test]
    fn test_preprocess_preserves_content() {
        // Single red pixel (255, 0, 0, 255)
        let rgba = vec![255, 0, 0, 255];
        let (tensor, info) = preprocess(&rgba, 1, 1);

        let data = tensor.as_slice::<f32>().unwrap();
        // The red channel plane should have a non-zero value where the pixel was placed
        // Pixel is placed at approximately (pad_x, pad_y)
        let px = info.pad_x.floor() as usize;
        let py = info.pad_y.floor() as usize;
        let r_idx = 0 * 640 * 640 + py * 640 + px; // R channel
        assert!(data[r_idx] > 0.9, "red pixel should be bright, got {}", data[r_idx]);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test test_preprocess -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement preprocess function**

```rust
pub fn preprocess(
    rgba_pixels: &[u8],
    width: u32,
    height: u32,
) -> (Tensor, LetterboxInfo) {
    let rgb = rgba_to_rgb(rgba_pixels);
    let (hwc_f32, info) = letterbox_resize(&rgb, width, height);
    let nchw = hwc_to_nchw(&hwc_f32, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    let tensor = tract_ndarray::Array4::from_shape_vec(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        nchw,
    )
    .expect("preprocess: NCHW data does not match expected shape")
    .into();

    (tensor, info)
}
```

Note: Add `use tract_onnx::prelude::*;` at top of file if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test test_preprocess -- --nocapture`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all preprocess tests as regression check**

Run: `cargo test preprocess -- --nocapture`
Expected: All tests in module PASS (RGBA→RGB, letterbox, HWC→NCHW, full pipeline)

- [ ] **Step 6: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/preprocess.rs`
Run: `git commit -s -m "feat: complete preprocessing pipeline (RGBA→tensor [1,3,640,640])"`

---

## Chunk 2: Postprocessing & Model

### Task 6: Postprocessing - Detection Struct & COCO Classes

**Files:**
- Create: `src/postprocess.rs`
- Modify: `src/lib.rs` (add `mod postprocess;`)

- [ ] **Step 1: Write Detection struct, COCO names, and failing tests**

Create `src/postprocess.rs`:

```rust
use serde::Serialize;

use crate::preprocess::LetterboxInfo;

/// A single object detection result in original image coordinates.
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: &'static str,
}

/// Full detection response serialized to JSON.
#[derive(Debug, Serialize)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
    pub inference_time_ms: u64,
    pub image_width: u32,
    pub image_height: u32,
}

/// COCO class names (80 classes, indices 0-79).
pub const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

/// Look up COCO class name by index.
pub fn class_name(class_id: u32) -> &'static str {
    COCO_CLASSES.get(class_id as usize).unwrap_or(&"unknown")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_class_count() {
        assert_eq!(COCO_CLASSES.len(), 80);
    }

    #[test]
    fn test_class_name_lookup() {
        assert_eq!(class_name(0), "person");
        assert_eq!(class_name(2), "car");
        assert_eq!(class_name(79), "toothbrush");
    }

    #[test]
    fn test_class_name_out_of_range() {
        assert_eq!(class_name(80), "unknown");
        assert_eq!(class_name(255), "unknown");
    }

    #[test]
    fn test_detection_json_serialization() {
        let det = Detection {
            x: 120.0,
            y: 45.0,
            width: 200.0,
            height: 400.0,
            confidence: 0.92,
            class_id: 0,
            class_name: "person",
        };
        let json = serde_json::to_string(&det).unwrap();
        assert!(json.contains("\"class_name\":\"person\""));
        assert!(json.contains("\"confidence\":0.92"));
    }

    #[test]
    fn test_detection_result_json() {
        let result = DetectionResult {
            detections: vec![],
            inference_time_ms: 350,
            image_width: 1920,
            image_height: 1080,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"detections\":[]"));
        assert!(json.contains("\"inference_time_ms\":350"));
    }
}
```

Add `mod postprocess;` to `src/lib.rs`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test postprocess -- --nocapture`
Expected: All 5 tests PASS (these tests don't require todo — struct and const are immediately available)

- [ ] **Step 3: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/postprocess.rs src/lib.rs`
Run: `git commit -s -m "feat: add Detection struct, COCO class names, JSON serialization"`

---

### Task 7: Postprocessing - Confidence Filtering & Coordinate Transform

**Files:**
- Modify: `src/postprocess.rs`

- [ ] **Step 1: Write failing tests for confidence filtering and coordinate transform**

Add to `src/postprocess.rs`:

```rust
/// Filter raw model detections by confidence threshold and transform coordinates
/// from model space (640x640) to original image space.
///
/// `raw_detections` is a slice of [cx, cy, w, h, class_scores...] per detection,
/// already transposed from the model output tensor to [N, 84] layout.
pub fn filter_and_transform(
    raw_detections: &[Vec<f32>],
    confidence_threshold: f32,
    letterbox_info: LetterboxInfo,
    image_width: u32,
    image_height: u32,
) -> Vec<Detection> {
    todo!()
}
```

Add tests:

```rust
    #[test]
    fn test_filter_by_confidence() {
        let info = LetterboxInfo { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        // Detection with class 0 (person) at confidence 0.9
        let mut det_high = vec![320.0, 320.0, 100.0, 200.0]; // cx, cy, w, h
        let mut scores_high = vec![0.0f32; 80];
        scores_high[0] = 0.9; // person
        det_high.extend(scores_high);

        // Detection with class 2 (car) at confidence 0.1
        let mut det_low = vec![100.0, 100.0, 50.0, 50.0];
        let mut scores_low = vec![0.0f32; 80];
        scores_low[2] = 0.1; // car
        det_low.extend(scores_low);

        let results = filter_and_transform(
            &[det_high, det_low], 0.25, info, 640, 640,
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "person");
        assert!((results[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_coordinate_transform_no_padding() {
        // No padding, scale=1.0 → coords should convert from center to top-left
        let info = LetterboxInfo { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        let mut det = vec![320.0, 240.0, 100.0, 200.0]; // cx=320, cy=240, w=100, h=200
        let mut scores = vec![0.0f32; 80];
        scores[0] = 0.95;
        det.extend(scores);

        let results = filter_and_transform(&[det], 0.25, info, 640, 640);
        assert_eq!(results.len(), 1);
        // x = cx - w/2 = 320 - 50 = 270, y = cy - h/2 = 240 - 100 = 140
        assert!((results[0].x - 270.0).abs() < 1e-3);
        assert!((results[0].y - 140.0).abs() < 1e-3);
        assert!((results[0].width - 100.0).abs() < 1e-3);
        assert!((results[0].height - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_coordinate_transform_with_padding() {
        // Image was 320x640 → scale=1.0, pad_x=160, pad_y=0
        let info = LetterboxInfo { scale: 1.0, pad_x: 160.0, pad_y: 0.0 };
        let mut det = vec![320.0, 320.0, 100.0, 100.0]; // in model space
        let mut scores = vec![0.0f32; 80];
        scores[0] = 0.8;
        det.extend(scores);

        let results = filter_and_transform(&[det], 0.25, info, 320, 640);
        // x = (cx - w/2 - pad_x) / scale = (320 - 50 - 160) / 1.0 = 110
        // y = (cy - h/2 - pad_y) / scale = (320 - 50 - 0) / 1.0 = 270
        assert!((results[0].x - 110.0).abs() < 1e-3);
        assert!((results[0].y - 270.0).abs() < 1e-3);
    }

    #[test]
    fn test_coordinate_transform_with_scale() {
        // Image was 1280x960 → scale=0.5, pad_x=0, pad_y=80
        // scale = 640/1280 = 0.5, new_h = 960*0.5 = 480, pad_y = (640-480)/2 = 80
        let info = LetterboxInfo { scale: 0.5, pad_x: 0.0, pad_y: 80.0 };
        let mut det = vec![320.0, 320.0, 200.0, 200.0];
        let mut scores = vec![0.0f32; 80];
        scores[1] = 0.7; // bicycle
        det.extend(scores);

        let results = filter_and_transform(&[det], 0.25, info, 1280, 960);
        // x = (320 - 100 - 0) / 0.5 = 440
        // y = (320 - 100 - 80) / 0.5 = 280
        // w = 200 / 0.5 = 400, h = 200 / 0.5 = 400
        assert!((results[0].x - 440.0).abs() < 1e-3);
        assert!((results[0].y - 280.0).abs() < 1e-3);
        assert!((results[0].width - 400.0).abs() < 1e-3);
        assert!((results[0].height - 400.0).abs() < 1e-3);
        assert_eq!(results[0].class_name, "bicycle");
    }

    #[test]
    fn test_coordinate_clamping() {
        // Detection near edge should be clamped to image bounds
        let info = LetterboxInfo { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        let mut det = vec![10.0, 10.0, 100.0, 100.0]; // will produce negative x,y
        let mut scores = vec![0.0f32; 80];
        scores[0] = 0.8;
        det.extend(scores);

        let results = filter_and_transform(&[det], 0.25, info, 640, 640);
        assert!(results[0].x >= 0.0, "x should be clamped to >= 0");
        assert!(results[0].y >= 0.0, "y should be clamped to >= 0");
    }

    #[test]
    fn test_empty_detections() {
        let info = LetterboxInfo { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        let results = filter_and_transform(&[], 0.25, info, 640, 640);
        assert!(results.is_empty());
    }

    #[test]
    fn test_all_filtered_out() {
        let info = LetterboxInfo { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        let mut det = vec![320.0, 320.0, 100.0, 100.0];
        let mut scores = vec![0.0f32; 80];
        scores[0] = 0.1; // below threshold
        det.extend(scores);

        let results = filter_and_transform(&[det], 0.25, info, 640, 640);
        assert!(results.is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test test_filter test_coordinate test_empty test_all_filtered -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement filter_and_transform**

```rust
pub fn filter_and_transform(
    raw_detections: &[Vec<f32>],
    confidence_threshold: f32,
    letterbox_info: LetterboxInfo,
    image_width: u32,
    image_height: u32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    let img_w = image_width as f32;
    let img_h = image_height as f32;

    for raw in raw_detections {
        if raw.len() < 84 {
            continue;
        }

        let cx = raw[0];
        let cy = raw[1];
        let w = raw[2];
        let h = raw[3];

        // Find best class
        let class_scores = &raw[4..84];
        let (class_id, &max_conf) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        if max_conf < confidence_threshold {
            continue;
        }

        // Convert from center to top-left, undo letterbox
        let x = (cx - w / 2.0 - letterbox_info.pad_x) / letterbox_info.scale;
        let y = (cy - h / 2.0 - letterbox_info.pad_y) / letterbox_info.scale;
        let det_w = w / letterbox_info.scale;
        let det_h = h / letterbox_info.scale;

        // Clamp to image bounds
        let x = x.max(0.0).min(img_w);
        let y = y.max(0.0).min(img_h);
        let det_w = det_w.min(img_w - x);
        let det_h = det_h.min(img_h - y);

        detections.push(Detection {
            x,
            y,
            width: det_w,
            height: det_h,
            confidence: max_conf,
            class_id: class_id as u32,
            class_name: class_name(class_id as u32),
        });
    }

    detections
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test postprocess -- --nocapture`
Expected: All postprocess tests PASS

- [ ] **Step 5: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/postprocess.rs`
Run: `git commit -s -m "feat: add confidence filtering and coordinate transform in postprocessing"`

---

### Task 8: Postprocessing - Parse Raw Tract Output Tensor

**Files:**
- Modify: `src/postprocess.rs`

- [ ] **Step 1: Write failing test for tensor parsing**

Add to `src/postprocess.rs`:

```rust
use tract_onnx::prelude::*;

/// Parse tract output tensor into per-detection vectors.
/// Handles the expected YOLO output shape [1, 84, N] (transposed to [N, 84]).
/// Also handles [1, N, 84] if that's what the model outputs.
pub fn parse_output_tensor(tensor: &Tensor) -> Result<Vec<Vec<f32>>, String> {
    todo!()
}
```

Add tests:

```rust
    #[test]
    fn test_parse_output_tensor_1_84_n() {
        // Shape [1, 84, 3] → 3 detections
        let data: Vec<f32> = (0..84 * 3).map(|i| i as f32).collect();
        let tensor: Tensor = tract_ndarray::Array3::from_shape_vec((1, 84, 3), data)
            .unwrap()
            .into();

        let detections = parse_output_tensor(&tensor).unwrap();
        assert_eq!(detections.len(), 3);
        assert_eq!(detections[0].len(), 84);
        // First detection should have values [0, 3, 6, ...] (transposed from [1,84,N])
        assert!((detections[0][0] - 0.0).abs() < 1e-6);  // row 0, col 0
        assert!((detections[0][1] - 3.0).abs() < 1e-6);  // row 1, col 0
    }

    #[test]
    fn test_parse_output_tensor_1_n_84() {
        // Shape [1, 2, 84] → 2 detections, already in [N, 84] layout
        let data: Vec<f32> = (0..2 * 84).map(|i| i as f32).collect();
        let tensor: Tensor = tract_ndarray::Array3::from_shape_vec((1, 2, 84), data)
            .unwrap()
            .into();

        let detections = parse_output_tensor(&tensor).unwrap();
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].len(), 84);
    }

    #[test]
    fn test_parse_output_tensor_empty() {
        let tensor: Tensor = tract_ndarray::Array3::<f32>::zeros((1, 84, 0)).into();
        let detections = parse_output_tensor(&tensor).unwrap();
        assert!(detections.is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test test_parse_output -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement parse_output_tensor**

```rust
pub fn parse_output_tensor(tensor: &Tensor) -> Result<Vec<Vec<f32>>, String> {
    let shape = tensor.shape();
    if shape.len() != 3 || shape[0] != 1 {
        return Err(format!("unexpected output shape: {:?}, expected [1, ?, ?]", shape));
    }

    let dim1 = shape[1];
    let dim2 = shape[2];
    let data = tensor
        .as_slice::<f32>()
        .map_err(|e| format!("failed to read tensor as f32: {}", e))?;

    // Determine if shape is [1, 84, N] or [1, N, 84]
    let (num_detections, num_classes_plus_4, is_transposed) = if dim1 == 84 || (dim1 > dim2 && dim1 >= 84) {
        // [1, 84, N] — standard YOLO format, needs transpose
        (dim2, dim1, true)
    } else {
        // [1, N, 84] — already per-detection layout
        (dim1, dim2, false)
    };

    if num_classes_plus_4 < 84 {
        return Err(format!(
            "expected at least 84 values per detection (4 bbox + 80 classes), got {}",
            num_classes_plus_4
        ));
    }

    let mut detections = Vec::with_capacity(num_detections);
    for i in 0..num_detections {
        let mut det = Vec::with_capacity(84);
        for j in 0..84 {
            let value = if is_transposed {
                // [1, 84, N] → index is [0, j, i] = j * dim2 + i
                data[j * dim2 + i]
            } else {
                // [1, N, 84] → index is [0, i, j] = i * dim2 + j
                data[i * dim2 + j]
            };
            det.push(value);
        }
        detections.push(det);
    }

    Ok(detections)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test test_parse_output -- --nocapture`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all postprocess tests as regression check**

Run: `cargo test postprocess -- --nocapture`
Expected: All tests PASS

- [ ] **Step 6: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/postprocess.rs`
Run: `git commit -s -m "feat: add YOLO output tensor parser (handles [1,84,N] and [1,N,84] shapes)"`

---

### Task 9: Model Wrapper (tract ONNX)

**Files:**
- Create: `src/model.rs`
- Modify: `src/lib.rs` (add `mod model;`)

- [ ] **Step 1: Write model wrapper with error handling test**

Create `src/model.rs`:

```rust
use tract_onnx::prelude::*;

type ModelPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Wrapper around a tract optimized ONNX model for YOLO26 inference.
pub struct YoloModel {
    plan: ModelPlan,
}

impl YoloModel {
    /// Load YOLO26 ONNX model from raw bytes.
    pub fn load(onnx_bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = std::io::Cursor::new(onnx_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| format!("failed to parse ONNX model: {}", e))?
            .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())
            .map_err(|e| format!("failed to set input shape: {}", e))?
            .into_optimized()
            .map_err(|e| format!("failed to optimize model: {}", e))?
            .into_runnable()
            .map_err(|e| format!("failed to create runnable plan: {}", e))?;

        Ok(YoloModel { plan: model })
    }

    /// Run inference on a preprocessed input tensor [1, 3, 640, 640].
    /// Returns the raw output tensor.
    pub fn run(&self, input: Tensor) -> Result<Tensor, String> {
        let result = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| format!("inference failed: {}", e))?;

        let output = result
            .get(0)
            .ok_or_else(|| "model produced no output".to_string())?
            .clone()
            .into_tensor();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_invalid_bytes() {
        let result = YoloModel::load(b"not a valid onnx model");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to parse ONNX"));
    }
}
```

Add `mod model;` to `src/lib.rs`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test model -- --nocapture`
Expected: `test_load_invalid_bytes` PASS

- [ ] **Step 3: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/model.rs src/lib.rs`
Run: `git commit -s -m "feat: add tract ONNX model wrapper with load and run methods"`

---

## Chunk 3: WASM Glue & HTML UI

### Task 10: WASM Entry Points (lib.rs)

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Add js-sys dependency to Cargo.toml**

Add under `[dependencies]`:
```toml
js-sys = "0.3"
```

- [ ] **Step 2: Implement init_model and detect WASM exports**

Replace `src/lib.rs` with:

```rust
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

mod model;
mod postprocess;
mod preprocess;

use model::YoloModel;
use postprocess::{filter_and_transform, parse_output_tensor, DetectionResult};
use preprocess::preprocess;

/// Global model instance. OnceLock is Sync (required for static).
/// In single-threaded WASM there is no contention.
static MODEL: OnceLock<YoloModel> = OnceLock::new();

/// Initialize panic hook for readable WASM error messages.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Load ONNX model bytes into memory. Called once on page load.
#[wasm_bindgen]
pub fn init_model(weights: &[u8]) -> Result<(), JsValue> {
    let model = YoloModel::load(weights).map_err(|e| JsValue::from_str(&e))?;

    MODEL
        .set(model)
        .map_err(|_| JsValue::from_str("model already initialized"))?;

    Ok(())
}

/// Run inference on RGBA pixels. Returns JSON string with detections.
#[wasm_bindgen]
pub fn detect(
    pixels: &[u8],
    width: u32,
    height: u32,
    confidence_threshold: f32,
) -> Result<String, JsValue> {
    let model = MODEL
        .get()
        .ok_or_else(|| JsValue::from_str("model not loaded — call init_model first"))?;

    let start = js_sys::Date::now();

    // Preprocess
    let (input_tensor, letterbox_info) = preprocess(pixels, width, height);

    // Inference
    let output_tensor = model.run(input_tensor).map_err(|e| JsValue::from_str(&e))?;

    let elapsed = (js_sys::Date::now() - start) as u64;

    // Postprocess
    let raw_detections =
        parse_output_tensor(&output_tensor).map_err(|e| JsValue::from_str(&e))?;

    let detections = filter_and_transform(
        &raw_detections,
        confidence_threshold,
        letterbox_info,
        width,
        height,
    );

    let result = DetectionResult {
        detections,
        inference_time_ms: elapsed,
        image_width: width,
        image_height: height,
    };

    serde_json::to_string(&result)
        .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))
}
```

- [ ] **Step 3: Verify it compiles for WASM target**

Run: `wasm-pack build --target web --dev`
Expected: Success, `pkg/` directory updated

- [ ] **Step 5: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add src/lib.rs Cargo.toml`
Run: `git commit -s -m "feat: add WASM entry points (init_model, detect) with global model state"`

---

### Task 11: HTML Demo UI

**Files:**
- Create: `index.html`

- [ ] **Step 1: Create index.html with full demo UI**

Create `index.html` with these sections:
1. Header with title and subtitle
2. File upload area (button + drag-and-drop zone)
3. Confidence slider (min=0.05, max=1.0, step=0.05, default=0.25)
4. Canvas for image + bounding box rendering
5. Status display
6. Detection results table
7. JS logic for:
   - WASM module loading
   - Model fetching and initialization
   - Image decoding via Canvas API
   - Calling `detect()` with threshold=0.0
   - Caching full results for slider filtering
   - Canvas bounding box rendering (HSL color per class)
   - Responsive layout (< 768px stacks vertically)
   - Error handling in status area

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO26 Rust WASM Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5; color: #333; line-height: 1.5;
        }
        .container {
            max-width: 960px; margin: 0 auto; padding: 20px;
        }
        header {
            text-align: center; padding: 20px 0;
        }
        header h1 { font-size: 1.8rem; color: #1a1a1a; }
        header p { color: #666; font-size: 0.95rem; }

        .upload-zone {
            border: 2px dashed #ccc; border-radius: 12px;
            padding: 40px; text-align: center; margin: 20px 0;
            cursor: pointer; transition: border-color 0.2s, background 0.2s;
        }
        .upload-zone:hover, .upload-zone.drag-over {
            border-color: #4a90d9; background: #f0f7ff;
        }
        .upload-zone input[type="file"] { display: none; }
        .upload-btn {
            display: inline-block; padding: 10px 24px;
            background: #4a90d9; color: white; border: none;
            border-radius: 8px; font-size: 1rem; cursor: pointer;
        }
        .upload-btn:hover { background: #357abd; }
        .upload-hint { color: #888; margin-top: 8px; font-size: 0.9rem; }

        .controls {
            display: flex; align-items: center; gap: 12px;
            margin: 16px 0; flex-wrap: wrap;
        }
        .controls label { font-weight: 500; }
        .controls input[type="range"] { flex: 1; min-width: 200px; }
        .controls .threshold-value {
            font-family: monospace; font-size: 1rem;
            min-width: 3em; text-align: right;
        }

        .canvas-container {
            position: relative; background: #000; border-radius: 8px;
            overflow: hidden; margin: 16px 0;
            display: none; /* hidden until image loaded */
        }
        .canvas-container canvas {
            display: block; width: 100%; height: auto;
        }

        .status {
            padding: 12px; border-radius: 8px; margin: 12px 0;
            font-size: 0.95rem;
        }
        .status.loading { background: #fff3cd; color: #856404; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }

        .detections-table {
            width: 100%; border-collapse: collapse;
            margin: 16px 0; display: none;
        }
        .detections-table th, .detections-table td {
            padding: 8px 12px; text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .detections-table th { background: #f8f8f8; font-weight: 600; }
        .detections-table .class-badge {
            display: inline-block; padding: 2px 8px;
            border-radius: 4px; color: white; font-size: 0.85rem;
        }

        @media (max-width: 768px) {
            .container { padding: 12px; }
            header h1 { font-size: 1.4rem; }
            .upload-zone { padding: 24px; }
            .controls { flex-direction: column; align-items: stretch; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>YOLO26 Rust WASM Demo</h1>
            <p>Browser-based object detection powered by Rust + WASM. No server needed.</p>
        </header>

        <div class="upload-zone" id="uploadZone">
            <button class="upload-btn" id="uploadBtn">Select Image</button>
            <input type="file" id="fileInput" accept="image/*">
            <p class="upload-hint">or drag and drop an image here</p>
        </div>

        <div class="controls">
            <label for="threshold">Confidence:</label>
            <span class="threshold-value" id="thresholdValue">0.25</span>
            <input type="range" id="threshold" min="0.05" max="1.0" step="0.05" value="0.25">
        </div>

        <div id="status" class="status loading">Loading WASM module...</div>

        <div class="canvas-container" id="canvasContainer">
            <canvas id="canvas"></canvas>
        </div>

        <table class="detections-table" id="detectionsTable">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>Position (x, y, w, h)</th>
                </tr>
            </thead>
            <tbody id="detectionsBody"></tbody>
        </table>
    </div>

    <script type="module">
        import init, { init_model, detect } from './pkg/yolo26_rust_wasm.js';

        const MAX_IMAGE_DIM = 4096;

        let wasmReady = false;
        let modelReady = false;
        let cachedDetections = null;
        let currentImage = null;

        const statusEl = document.getElementById('status');
        const canvasContainer = document.getElementById('canvasContainer');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        const uploadBtn = document.getElementById('uploadBtn');
        const thresholdSlider = document.getElementById('threshold');
        const thresholdValue = document.getElementById('thresholdValue');
        const detectionsTable = document.getElementById('detectionsTable');
        const detectionsBody = document.getElementById('detectionsBody');

        function setStatus(message, type = 'loading') {
            statusEl.textContent = message;
            statusEl.className = `status ${type}`;
        }

        function classColor(classId) {
            const hue = (classId * 137.5) % 360;
            return `hsl(${hue}, 70%, 50%)`;
        }

        function renderDetections(detections) {
            // Clear canvas and redraw image (currentImage is the offscreen canvas
            // at the dimensions actually sent to WASM, so coordinates match)
            canvas.width = currentImage.width;
            canvas.height = currentImage.height;
            ctx.drawImage(currentImage, 0, 0);

            // Draw bounding boxes
            for (const det of detections) {
                const color = classColor(det.class_id);

                // Semi-transparent fill
                ctx.fillStyle = color.replace(')', ', 0.15)').replace('hsl', 'hsla');
                ctx.fillRect(det.x, det.y, det.width, det.height);

                // Border
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(det.x, det.y, det.width, det.height);

                // Label
                const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
                ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
                const textWidth = ctx.measureText(label).width;
                const labelHeight = 20;
                const labelY = det.y > labelHeight ? det.y - labelHeight : det.y;

                ctx.fillStyle = color;
                ctx.fillRect(det.x, labelY, textWidth + 8, labelHeight);
                ctx.fillStyle = 'white';
                ctx.fillText(label, det.x + 4, labelY + 15);
            }

            // Update table
            detectionsBody.innerHTML = '';
            detections.forEach((det, i) => {
                const row = document.createElement('tr');
                const color = classColor(det.class_id);
                row.innerHTML = `
                    <td>${i + 1}</td>
                    <td><span class="class-badge" style="background:${color}">${det.class_name}</span></td>
                    <td>${Math.round(det.confidence * 100)}%</td>
                    <td>(${Math.round(det.x)}, ${Math.round(det.y)}, ${Math.round(det.width)}, ${Math.round(det.height)})</td>
                `;
                detectionsBody.appendChild(row);
            });
            detectionsTable.style.display = detections.length > 0 ? 'table' : 'none';
        }

        function filterAndRender() {
            if (!cachedDetections) return;
            const threshold = parseFloat(thresholdSlider.value);
            const filtered = cachedDetections.filter(d => d.confidence >= threshold);
            renderDetections(filtered);
            setStatus(`Detected ${filtered.length} objects (${cachedDetections.length} total, threshold ${threshold.toFixed(2)})`, 'success');
        }

        async function processImage(file) {
            if (!modelReady) {
                setStatus('Model not loaded yet', 'error');
                return;
            }

            setStatus('Detecting...', 'loading');

            const img = new Image();
            const url = URL.createObjectURL(file);

            img.onload = () => {
                URL.revokeObjectURL(url);

                // Resize if too large
                let w = img.width;
                let h = img.height;
                if (w > MAX_IMAGE_DIM || h > MAX_IMAGE_DIM) {
                    const scale = MAX_IMAGE_DIM / Math.max(w, h);
                    w = Math.round(w * scale);
                    h = Math.round(h * scale);
                }

                // Decode to RGBA pixels via offscreen canvas
                const offscreen = document.createElement('canvas');
                offscreen.width = w;
                offscreen.height = h;
                const offCtx = offscreen.getContext('2d');
                offCtx.drawImage(img, 0, 0, w, h);
                const imageData = offCtx.getImageData(0, 0, w, h);

                // Store the working dimensions (may differ from img.naturalWidth if resized)
                currentImage = offscreen;
                canvasContainer.style.display = 'block';

                try {
                    // Call WASM with threshold=0.0 to get all detections
                    // w, h match the pixels we actually sent to WASM
                    const jsonStr = detect(imageData.data, w, h, 0.0);
                    const result = JSON.parse(jsonStr);

                    cachedDetections = result.detections;
                    filterAndRender();

                    const threshold = parseFloat(thresholdSlider.value);
                    const filtered = cachedDetections.filter(d => d.confidence >= threshold);
                    setStatus(
                        `Detected ${filtered.length} objects in ${result.inference_time_ms}ms`,
                        'success'
                    );
                } catch (e) {
                    setStatus(`Detection failed: ${e}`, 'error');
                }
            };

            img.onerror = () => {
                URL.revokeObjectURL(url);
                setStatus('Failed to load image', 'error');
            };

            img.src = url;
        }

        // Event listeners
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) processImage(e.target.files[0]);
        });

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files[0]) processImage(e.dataTransfer.files[0]);
        });

        thresholdSlider.addEventListener('input', () => {
            thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
            filterAndRender();
        });

        // Initialize
        async function startup() {
            try {
                await init();
                wasmReady = true;
                setStatus('Loading model...', 'loading');

                const modelUrl = 'weights/yolo26n.onnx';
                const response = await fetch(modelUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
                }
                const bytes = new Uint8Array(await response.arrayBuffer());

                init_model(bytes);
                modelReady = true;
                const sizeMB = (bytes.length / 1024 / 1024).toFixed(1);
                setStatus(`Model loaded (${sizeMB} MB). Select an image to begin.`, 'success');
            } catch (e) {
                setStatus(`Failed to initialize: ${e}`, 'error');
            }
        }

        startup();
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify WASM build still works**

Run: `wasm-pack build --target web --dev`
Expected: Success

- [ ] **Step 3: Run formatter and commit**

Run: `cargo fmt --all`
Run: `git add index.html`
Run: `git commit -s -m "feat: add index.html demo UI with file upload, canvas rendering, confidence slider"`

---

## Chunk 4: Integration, README & Polish

### Task 12: WASM Build Verification & Model Download

**Files:**
- Modify: `scripts/download_model.sh` (if needed)

- [ ] **Step 1: Ensure WASM release build succeeds**

Run: `wasm-pack build --target web --release`
Expected: Success. Check `pkg/` for `.wasm` file. Note the file size.

- [ ] **Step 2: Download/export the YOLO26n ONNX model**

Run: `./scripts/download_model.sh`
Expected: `weights/yolo26n.onnx` exists. If the script fails (ultralytics not installed or YOLO26 not supported yet), manually export:

```bash
pip install ultralytics
python3 -c "from ultralytics import YOLO; m = YOLO('yolo26n.pt'); m.export(format='onnx', simplify=True, opset=17, imgsz=640)"
mv yolo26n.onnx weights/
```

- [ ] **Step 3: Inspect ONNX output shape**

Run:
```bash
python3 -c "
import onnx
model = onnx.load('weights/yolo26n.onnx')
for output in model.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in output.type.tensor_type.shape.dim]
    print(f'{output.name}: {dims}')
"
```
Expected: Output shape printed. If it differs from `[1, 84, N]`, update `src/postprocess.rs:parse_output_tensor` to handle the actual shape.

- [ ] **Step 4: Test locally in browser**

Run: `npx serve . -l 3000`
Open browser to `http://localhost:3000/index.html`
Expected: Page loads, model initializes, status shows "Model loaded (X.X MB)"

- [ ] **Step 5: Upload a test image and verify detection**

Upload any image with common objects (people, cars, etc.)
Expected:
- "Detecting..." status briefly shown
- Bounding boxes appear on canvas
- Detection table populates
- Confidence slider filters detections in real-time

- [ ] **Step 6: Test on mobile viewport**

In browser dev tools, toggle device toolbar (mobile view, 375px width).
Expected: Layout stacks vertically, canvas fills width, table scrolls.

- [ ] **Step 7: Commit any adjustments**

Run: `cargo fmt --all`
Run: `git status` (review what changed)
Run: `git add src/ index.html scripts/ Cargo.toml` (only stage relevant files)
Run: `git commit -s -m "fix: adjustments from integration testing"`

---

### Task 13: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# YOLO26 Rust WASM Demo

Browser-based object detection using YOLO26n, running entirely in-browser via Rust + WebAssembly. No server-side inference needed.

## Features

- YOLO26n object detection (80 COCO classes) compiled to WASM
- Single `index.html` — no framework dependencies
- Image upload via button or drag-and-drop
- Real-time confidence threshold slider
- Bounding box visualization on Canvas
- Mobile-responsive layout
- All processing happens locally — no data leaves the browser

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) with `wasm32-unknown-unknown` target
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- Python 3 + [ultralytics](https://pypi.org/project/ultralytics/) (for model export)

### Setup

```bash
# Install WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Download/export YOLO26n model
pip install ultralytics
./scripts/download_model.sh

# Build WASM
wasm-pack build --target web --release

# Serve locally
npx serve .
```

Open `http://localhost:3000` in your browser.

## Architecture

```
Browser (Vanilla JS)
  → Image decoded to RGBA pixels via Canvas API
  → Passed to WASM module
    → Preprocessing: RGBA→RGB, letterbox 640x640, normalize, HWC→CHW
    → Inference: tract ONNX runtime
    → Postprocessing: confidence filter, coordinate transform
  → JSON results rendered as bounding boxes on Canvas
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Inference runtime | [tract](https://github.com/sonos/tract) (ONNX) |
| WASM toolchain | wasm-pack + wasm-bindgen |
| Model | YOLO26n (NMS-Free, 80 COCO classes) |
| Frontend | Vanilla JS, Canvas API |

## License

MIT
```

- [ ] **Step 2: Commit README**

Run: `git add README.md`
Run: `git commit -s -m "docs: add README with setup instructions and architecture overview"`

---

### Task 14: Final Cleanup

**Files:**
- Verify all files are committed
- Run final tests

- [ ] **Step 1: Run full test suite**

Run: `cargo test -- --nocapture`
Expected: All unit tests pass

- [ ] **Step 2: Run WASM release build**

Run: `RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release`
Expected: Success. SIMD flag enables WASM SIMD128 for ~2.6x inference speedup.

- [ ] **Step 3: Verify .gitignore is complete**

Run: `git status`
Expected: Clean working tree (no untracked files that should be ignored)

- [ ] **Step 4: Final format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

- [ ] **Step 5: Review all changes since initial commit**

Run: `git log --oneline`
Verify commits are clean and well-described.

---

### Task 15: Create Pull Request

- [ ] **Step 1: Push feature branch**

Run: `git push -u origin feat/initial-implementation`

- [ ] **Step 2: Create PR**

Run:
```bash
gh pr create --title "feat: YOLO26 Rust WASM browser demo" --body "$(cat <<'EOF'
## Summary
- Rust WASM module (tract ONNX) for YOLO26n inference in the browser
- Single index.html demo with image upload, canvas rendering, confidence slider
- Preprocessing (RGBA→RGB, letterbox, normalize) and postprocessing (coord transform, JSON)
- Mobile responsive, no server-side inference

## Test plan
- [ ] `cargo test` passes all unit tests
- [ ] `wasm-pack build --target web --release` succeeds
- [ ] Upload image in Chrome → bounding boxes rendered correctly
- [ ] Confidence slider filters detections without WASM re-invocation
- [ ] Mobile viewport layout stacks correctly
EOF
)"
```

- [ ] **Step 3: Return to main repo**

Run: `cd /Users/yhkwon/Documents/Projects/yolo26-rust-wasm`
