# Design Spec: YOLO26 Rust WASM HTML Demo

**Date**: 2026-03-13
**Status**: Draft
**Scope**: Phase 1 (MVP)

---

## 1. Overview

Browser-based YOLO26 object detection demo. The inference pipeline (preprocess, inference, postprocess) runs entirely inside a Rust WASM module using tract as the ONNX runtime. A single `index.html` with vanilla JS handles image upload, WASM invocation, and Canvas-based result rendering.

**Key properties:**
- No server-side inference — everything runs in the browser
- No framework dependencies — vanilla JS + WASM
- YOLO26n ONNX model fetched separately (Phase 1); embedded in WASM binary in Phase 2
- NMS-Free model — no NMS implementation required

## 2. Architecture

```
Browser (index.html + vanilla JS)
  │ RGBA pixels (from Canvas API)
  ▼
Rust WASM Module (tract ONNX runtime)
  ├── lib.rs          → wasm-bindgen entry points
  ├── model.rs        → tract ONNX model load + inference
  ├── preprocess.rs   → RGBA→RGB, letterbox 640x640, normalize, HWC→CHW
  └── postprocess.rs  → parse output tensor, confidence filter, coord transform
  │ JSON string (detection results)
  ▼
Canvas 2D rendering (bounding boxes + class labels + confidence)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `index.html` | Image upload (button + drag-and-drop), Canvas rendering, confidence slider, detection list, mobile responsive layout |
| `lib.rs` | wasm-bindgen exports (`init_model`, `detect`), glue between JS and Rust internals |
| `model.rs` | Load ONNX bytes into tract `SimplePlan`, run inference, return raw output tensor |
| `preprocess.rs` | Convert RGBA pixels to model input tensor [1, 3, 640, 640] f32 |
| `postprocess.rs` | Parse model output tensor into `Vec<Detection>`, coordinate transform, JSON serialization |

## 3. Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inference runtime | tract | Mature pure-Rust ONNX runtime with proven WASM support, wide operator coverage |
| Model format | ONNX | Direct export from ultralytics, avoids custom architecture implementation |
| Model loading (Phase 1) | JS `fetch` → pass bytes to WASM | Simple, allows CDN caching, decoupled from WASM binary size |
| Image decoding | Browser Canvas API | Supports all image formats natively, avoids `image` crate bloating WASM binary (~1MB saved) |
| Confidence slider | JS-side filtering of cached results | First inference uses threshold=0.0 to get all detections; slider filters without re-invoking WASM |
| Build tool | wasm-pack | Standard Rust-to-WASM toolchain, generates JS glue code |

## 4. WASM API

```rust
use wasm_bindgen::prelude::*;

/// Load ONNX model bytes into memory. Called once on page load.
#[wasm_bindgen]
pub fn init_model(weights: &[u8]) -> Result<(), JsValue>;

/// Run inference on RGBA pixels. Returns JSON string with detections.
/// The HTML demo passes confidence_threshold=0.0 and filters in JS.
#[wasm_bindgen]
pub fn detect(
    pixels: &[u8],              // RGBA raw pixels from canvas.getImageData()
    width: u32,                 // Original image width
    height: u32,                // Original image height
    confidence_threshold: f32,  // Minimum confidence (0.0-1.0)
) -> Result<String, JsValue>;  // JSON string
```

### Detection JSON Format

```json
{
  "detections": [
    {
      "x": 120,
      "y": 45,
      "width": 200,
      "height": 400,
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "inference_time_ms": 350,
  "image_width": 1920,
  "image_height": 1080
}
```

- `x, y`: bounding box top-left corner in original image pixels
- `width, height`: bounding box dimensions in original image pixels
- `class_id`: COCO class ID (0-79)
- `class_name`: COCO class name string

## 5. Preprocessing Pipeline (`preprocess.rs`)

Input: `&[u8]` RGBA pixels + width + height

```
1. RGBA → RGB
   Drop every 4th byte (alpha channel)

2. Letterbox Resize → 640x640
   Maintain aspect ratio, pad with gray (114/255)
   Record: scale_factor, pad_x, pad_y (needed for postprocess coord transform)

3. Normalize
   [0, 255] u8 → [0.0, 1.0] f32

4. HWC → CHW
   (640, 640, 3) → (3, 640, 640)

5. Add batch dimension
   (3, 640, 640) → (1, 3, 640, 640)
```

Output: tract `Tensor` [1, 3, 640, 640] f32 + `LetterboxInfo { scale, pad_x, pad_y }`

## 6. Postprocessing Pipeline (`postprocess.rs`)

YOLO26 is NMS-Free — no NMS step required.

Input: tract output tensor + `confidence_threshold` + `LetterboxInfo`

```
1. Parse output tensor
   Expected shape: [1, 84, N] where 84 = 4 (cx, cy, w, h) + 80 (class scores)
   Transpose to [N, 84] for easier iteration

2. For each detection:
   a. Extract bbox: cx, cy, w, h (in 640x640 model space)
   b. Extract 80 class scores, find max → confidence + class_id
   c. Skip if confidence < threshold

3. Coordinate transform (model space → original image space):
   x = (cx - w/2 - pad_x) / scale
   y = (cy - h/2 - pad_y) / scale
   width = w / scale
   height = h / scale
   Clamp to image bounds

4. Build Vec<Detection> with COCO class name lookup

5. Serialize to JSON via serde_json
```

Output: JSON string

### COCO Class Names

Static array of 80 COCO class names (`["person", "bicycle", "car", ...]`) defined as a constant in `postprocess.rs`.

## 7. Model (`model.rs`)

```rust
use tract_onnx::prelude::*;

/// Wrapper around tract's optimized ONNX model
pub struct YoloModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl YoloModel {
    /// Load from ONNX bytes
    pub fn load(onnx_bytes: &[u8]) -> Result<Self, Error>;

    /// Run inference, return raw output tensor
    pub fn run(&self, input: Tensor) -> Result<Tensor, Error>;
}
```

Global model state stored in a `thread_local!` or `OnceCell` for WASM (single-threaded).

## 8. HTML UI (`index.html`)

### Layout (Responsive)

```
+----------------------------------------------------------+
|  YOLO26 Rust WASM Demo                                    |
|  Browser-based object detection powered by Rust + WASM    |
+----------------------------------------------------------+
|                                                           |
|  [Image Select]  or drag-and-drop here                    |
|                                                           |
|  Confidence: 0.25  ========o========== 1.0                |
|                                                           |
|  +----------------------------------------------------+  |
|  |                                                    |  |
|  |              Canvas                                |  |
|  |         (image + bounding boxes)                   |  |
|  |                                                    |  |
|  +----------------------------------------------------+  |
|                                                           |
|  Detected 5 objects in 342ms                              |
|                                                           |
|  +-- Detections ------------------------------------+    |
|  |  #  Class     Conf    Position                   |    |
|  |  1  person    92%     (120, 45, 200, 400)        |    |
|  |  2  car       87%     (500, 200, 300, 180)       |    |
|  +--------------------------------------------------+    |
+----------------------------------------------------------+
```

### Mobile (< 768px)
- All elements stacked vertically
- Canvas width = 100% viewport
- Detection list scrollable below canvas
- File input shows camera option (`accept="image/*"`)

### Bounding Box Rendering
- Canvas 2D Context
- Color per class: HSL with `hue = class_id * 137.5 % 360`, saturation 70%, lightness 50%
- Box: `strokeRect` 2px + semi-transparent fill (alpha 0.15)
- Label: colored background rect + white text `"class_name confidence%"` above box
- Canvas resolution = original image size; CSS scales to container

### User Flow

1. **Page load**: Load WASM → fetch ONNX model → `init_model(bytes)` → status "Model loaded (X.X MB)"
2. **Image selected**: Decode via Canvas → get RGBA pixels → status "Detecting..." → `detect(pixels, w, h, 0.0)` → parse JSON → render boxes → update list → status "Detected N objects in Xms"
3. **Slider change**: Filter cached detections by threshold → re-render Canvas + list (no WASM call)

## 9. Project Structure

```
yolo26-rust-wasm/
├── Cargo.toml                  # tract-onnx, wasm-bindgen, serde, serde_json
├── src/
│   ├── lib.rs                  # wasm-bindgen entry points, global model state
│   ├── model.rs                # tract ONNX model wrapper
│   ├── preprocess.rs           # image preprocessing pipeline
│   └── postprocess.rs          # output parsing, coord transform, JSON serialization
├── scripts/
│   └── download_model.sh       # Download YOLO26n ONNX from known URL
├── weights/                    # .gitignore'd
│   └── yolo26n.onnx
├── index.html                  # Demo UI (single file)
├── pkg/                        # wasm-pack build output (.gitignore'd)
├── tests/                      # Integration tests
├── PRD.md
├── CLAUDE.md
├── METHODOLOGY.md
├── FORMATTING.md
└── README.md
```

## 10. Dependencies (Cargo.toml)

| Crate | Purpose |
|-------|---------|
| `tract-onnx` | ONNX model loading and inference |
| `wasm-bindgen` | Rust ↔ JS FFI for WASM |
| `serde` + `serde_json` | Detection struct serialization to JSON |
| `web-sys` | Access to `console::log` for WASM logging (optional) |
| `getrandom` (with `js` feature) | Required by some crates in WASM env |

## 11. Build & Setup

```bash
# Prerequisites
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Download model
./scripts/download_model.sh

# Build WASM
wasm-pack build --target web --release

# Run locally
npx serve .
# → http://localhost:3000/index.html
```

## 12. Testing Strategy

Following TDD (Red-Green-Refactor) per project rules.

### Unit Tests (native target, `cargo test`)

| Module | Test Cases |
|--------|-----------|
| `preprocess` | RGBA→RGB conversion correctness; letterbox resize with known dimensions; normalization range [0,1]; output tensor shape [1,3,640,640]; padding values for various aspect ratios |
| `postprocess` | Confidence filtering (above/below threshold); coordinate transform accuracy (model→original space); JSON serialization format; edge cases (no detections, all filtered out); COCO class name lookup |
| `model` | ONNX load from valid bytes; error on invalid bytes; output tensor shape verification |

### Integration Tests (`wasm-pack test --headless`)

- Full pipeline: load model → preprocess test image → inference → postprocess → verify JSON structure
- Round-trip coordinate accuracy: known input image with known objects → verify bbox positions

### Manual Verification

- COCO validation images in Chrome + Mobile Chrome
- Drag-and-drop functionality
- Confidence slider responsiveness
- Mobile layout

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| YOLO26 ONNX uses ops not supported by tract | Check tract op coverage against exported ONNX graph before implementation. Fall back to rten if needed. |
| ONNX output tensor shape differs from expected [1,84,N] | Inspect actual ONNX output shape after export, adapt postprocessing accordingly |
| tract WASM binary too large | Enable LTO, use `wasm-opt -Oz`, strip debug symbols |
| Model download URL becomes unavailable | Document manual export steps as fallback in README |
| Mobile WASM memory limits | JS-side image resize to max 4096px before passing to WASM |

## 14. Out of Scope (Phase 1)

Per PRD non-goals:
- Blur/anonymization
- Box editing UI
- Framework dependencies (React/Next.js)
- Multi-threading (SharedArrayBuffer)
- WebGPU acceleration
- Export functionality
- Video inference
- Custom model training
- Model embedding in WASM binary (Phase 2)
