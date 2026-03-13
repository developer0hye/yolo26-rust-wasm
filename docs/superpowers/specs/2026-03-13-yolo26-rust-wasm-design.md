# Design Spec: YOLO26 Rust WASM HTML Demo

**Date**: 2026-03-13
**Status**: Draft
**Scope**: Phase 1 (MVP)

---

## 1. Overview

Browser-based YOLO26 object detection demo. The inference pipeline (preprocess, inference, postprocess) runs entirely inside a Rust WASM module using candle (HuggingFace) as the tensor computation framework. A single `index.html` with vanilla JS handles image upload, WASM invocation, and Canvas-based result rendering.

**Key properties:**
- No server-side inference — everything runs in the browser
- No framework dependencies — vanilla JS + WASM
- YOLO26n SafeTensors model fetched separately (Phase 1); embedded in WASM binary in Phase 2
- NMS-Free end2end model — no NMS implementation required

## 2. Architecture

```
Browser (index.html + vanilla JS)
  │ RGBA pixels (from Canvas API)
  ▼
Rust WASM Module (candle tensor framework)
  ├── lib.rs              → wasm-bindgen entry points
  ├── model.rs            → top-level YOLO26 model (backbone + neck + head)
  ├── model/backbone.rs   → Conv, C3k2, SPPF, C2PSA backbone layers
  ├── model/neck.rs       → FPN + PAN neck with Upsample + Concat + C3k2
  ├── model/head.rs       → Detect head (end2end one2one, box regression + classification)
  ├── model/blocks.rs     → C3k2, C3k, C2PSA, PSABlock, Attention, SPPF, Bottleneck, ConvBlock
  ├── preprocess.rs       → RGBA→RGB, letterbox 640x640, normalize, HWC→CHW
  └── postprocess.rs      → parse output tensor, confidence filter, coord transform
  │ JSON string (detection results)
  ▼
Canvas 2D rendering (bounding boxes + class labels + confidence)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `index.html` | Image upload (button + drag-and-drop), Canvas rendering, confidence slider, detection list, mobile responsive layout |
| `lib.rs` | wasm-bindgen exports (`init_model`, `detect`), glue between JS and Rust internals |
| `model.rs` | Top-level YOLO26 struct, load SafeTensors weights via `VarBuilder`, run forward pass through backbone/neck/head |
| `model/blocks.rs` | Reusable building blocks: ConvBlock (Conv2d+BN+SiLU), Bottleneck, C3k2, C3k, C2PSA, PSABlock, Attention, SPPF, DWConv |
| `model/backbone.rs` | Backbone: Conv stem + C3k2 stages + SPPF + C2PSA |
| `model/neck.rs` | FPN-PAN neck: Upsample + Concat + C3k2 feature fusion |
| `model/head.rs` | Detect head: one2one box/cls branches, DFL (reg_max=1), anchor generation, end2end postprocess (topk) |
| `preprocess.rs` | Convert RGBA pixels to model input tensor [1, 3, 640, 640] f32 |
| `postprocess.rs` | Parse model output tensor into `Vec<Detection>`, coordinate transform, JSON serialization |

## 3. Technology Choices

**Why candle instead of tract?** Candle has proven WASM+wasm-bindgen support with official YOLO WASM examples (`huggingface/candle/candle-wasm-examples/yolo/`). tract has a known wasm-bindgen crash risk (sonos/tract#2001). The YOLO26 architecture can be built on top of candle's existing YOLOv8 implementation, reusing ConvBlock, Upsample, and Bottleneck, while adding YOLO26-specific modules (C3k2, C2PSA, modified SPPF, modified Detect).

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inference runtime | candle (not tract) | Proven WASM+wasm-bindgen support (official YOLO WASM examples); tract has wasm-bindgen crash risk (sonos/tract#2001) |
| Model format | SafeTensors (not ONNX) | Candle native SafeTensors support via `VarBuilder`; smaller file (~5MB FP16 vs ~10MB ONNX) |
| Model loading (Phase 1) | JS `fetch` → pass bytes to WASM | Simple, allows CDN caching, decoupled from WASM binary size |
| Image decoding | Browser Canvas API | Supports all image formats natively, avoids `image` crate bloating WASM binary (~1MB saved) |
| Confidence slider | JS-side filtering of cached results | First inference uses threshold=0.0 to get all detections; slider filters without re-invoking WASM |
| Build tool | wasm-pack | Standard Rust-to-WASM toolchain, generates JS glue code |

## 4. WASM API

```rust
use wasm_bindgen::prelude::*;

/// Load SafeTensors model bytes into memory. Called once on page load.
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

Output: candle `Tensor` [1, 3, 640, 640] f32 + `LetterboxInfo { scale, pad_x, pad_y }`

## 6. Postprocessing Pipeline (`postprocess.rs`)

YOLO26 is NMS-Free (end2end mode) — the model performs internal topk selection, so no NMS or per-class deduplication is needed.

Input: candle output tensor + `confidence_threshold` + `LetterboxInfo`

```
1. Parse output tensor
   YOLO26 end2end output shape: [1, 300, 6]
   Format per row: [x1, y1, x2, y2, confidence, class_id]
   The 300 comes from max_det=300 topk selection inside the Detect head.
   No transpose needed — already in [N, 6] layout (after removing batch dim).

2. For each of the 300 detections:
   a. Extract bbox: x1, y1, x2, y2 (xyxy format in 640x640 model space)
   b. Extract confidence score (already max across classes, post-sigmoid)
   c. Extract class_id (float, cast to u32)
   d. Skip if confidence < threshold

3. Coordinate transform (model space → original image space):
   Coordinates are already in xyxy format, apply letterbox reverse:
   x1_orig = (x1 - pad_x) / scale
   y1_orig = (y1 - pad_y) / scale
   x2_orig = (x2 - pad_x) / scale
   y2_orig = (y2 - pad_y) / scale
   Convert to (x, y, width, height) for JSON output:
   x = x1_orig, y = y1_orig
   width = x2_orig - x1_orig
   height = y2_orig - y1_orig
   Clamp to image bounds

4. Build Vec<Detection> with COCO class name lookup

5. Serialize to JSON via serde_json
```

Output: JSON string

### COCO Class Names

Static array of 80 COCO class names (`["person", "bicycle", "car", ...]`) defined as a constant in `postprocess.rs`.

## 7. Model (`model.rs` + `model/`)

YOLO26 architecture implemented natively in candle, based on the ultralytics YOLO26 config (`yolo26.yaml`). The model loads SafeTensors weights via `VarBuilder::from_buffered_safetensors`.

### Architecture (YOLO26n: depth=0.50, width=0.25, max_channels=1024)

**Backbone** (layers 0-10):
| # | Module | Args (after scaling) | Output |
|---|--------|---------------------|--------|
| 0 | Conv | ch=16, k=3, s=2 | P1/2 |
| 1 | Conv | ch=32, k=3, s=2 | P2/4 |
| 2 | C3k2 | ch=64, c3k=False, e=0.25, n=1 | |
| 3 | Conv | ch=64, k=3, s=2 | P3/8 |
| 4 | C3k2 | ch=128, c3k=False, e=0.25, n=1 | |
| 5 | Conv | ch=128, k=3, s=2 | P4/16 |
| 6 | C3k2 | ch=128, c3k=True, n=1 | |
| 7 | Conv | ch=256, k=3, s=2 | P5/32 |
| 8 | C3k2 | ch=256, c3k=True, n=1 | |
| 9 | SPPF | ch=256, k=5, n=3, shortcut=True | |
| 10 | C2PSA | ch=256, n=1 | |

**Neck** (layers 11-22, FPN+PAN):
| # | Module | From | Args | Note |
|---|--------|------|------|------|
| 11 | Upsample | 10 | scale=2, nearest | |
| 12 | Concat | [11,6] | dim=1 | cat P4 |
| 13 | C3k2 | 12 | ch=128, c3k=True, n=1 | |
| 14 | Upsample | 13 | scale=2, nearest | |
| 15 | Concat | [14,4] | dim=1 | cat P3 |
| 16 | C3k2 | 15 | ch=64, c3k=True, n=1 | P3/8-small |
| 17 | Conv | 16 | ch=64, k=3, s=2 | downsample |
| 18 | Concat | [17,13] | dim=1 | cat P4 |
| 19 | C3k2 | 18 | ch=128, c3k=True, n=1 | P4/16-medium |
| 20 | Conv | 19 | ch=128, k=3, s=2 | downsample |
| 21 | Concat | [20,10] | dim=1 | cat P5 |
| 22 | C3k2 | 21 | ch=256, c3k=True, e=0.5, attn=True, n=1 | P5/32-large |

**Head** (layer 23):
| # | Module | From | Args |
|---|--------|------|------|
| 23 | Detect | [16,19,22] | nc=80, reg_max=1, end2end=True |

### Building Blocks

| Block | Description | Reusable from candle YOLOv8? |
|-------|-------------|------------------------------|
| ConvBlock | Conv2d + BatchNorm + SiLU activation | Yes |
| Upsample | Nearest-neighbor 2x upscale | Yes |
| Bottleneck | Two ConvBlocks with optional residual shortcut | Yes |
| C3k2 | C2f variant: cv1 splits → n Bottleneck/C3k/Bottleneck+PSABlock branches → cv2 merges. When `c3k=False`, uses Bottleneck; when `c3k=True`, uses C3k (CSP bottleneck with k=3); when `attn=True`, uses Bottleneck+PSABlock | New (extends C2f) |
| C3k | C3 variant with configurable kernel size, containing n Bottleneck(k=3,3) blocks | New |
| C2PSA | Conv split → n PSABlock → Conv merge (position-sensitive attention wrapper) | New |
| PSABlock | Attention + FFN (Conv→Conv) with residual connections | New |
| Attention | Multi-head self-attention via 1x1 Conv QKV + DWConv positional encoding | New |
| DWConv | Depthwise convolution (groups=channels), used in Detect cls branch | New |
| SPPF | Spatial Pyramid Pooling Fast: Conv → n sequential MaxPool2d(k=5) → Concat → Conv, with optional residual shortcut (YOLO26 uses shortcut=True, n=3) | Modified (added n, shortcut params) |
| Detect | End2end detection head: one2one box regression (Conv→Conv→Conv2d, 4*reg_max outputs) + classification (DWConv→Conv→DWConv→Conv→Conv2d, nc outputs) per scale; DFL disabled (reg_max=1 → Identity); dist2bbox in xyxy mode; topk postprocess (max_det=300) | Modified (end2end, DWConv cls, reg_max=1) |

### Weight Loading

```rust
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

pub struct Yolo26Model {
    backbone: Backbone,
    neck: Neck,
    head: Detect,
}

impl Yolo26Model {
    /// Load from SafeTensors bytes
    pub fn load(safetensors_bytes: &[u8], device: &Device) -> Result<Self>;

    /// Run forward pass, return output tensor [1, 300, 6]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor>;
}
```

Weights loaded via `VarBuilder::from_buffered_safetensors(safetensors_bytes, DType::F32, device)`. SafeTensors key names follow the ultralytics naming convention (e.g., `model.0.conv.weight`, `model.10.m.0.attn.qkv.conv.weight`).

### File Organization

Split into multiple files for clarity:
- `model.rs` — top-level `Yolo26Model` struct, `load` and `forward`
- `model/backbone.rs` — `Backbone` struct with layers 0-10
- `model/neck.rs` — `Neck` struct with layers 11-22
- `model/head.rs` — `Detect` struct with end2end one2one branches + postprocess
- `model/blocks.rs` — all reusable blocks (ConvBlock, Bottleneck, C3k2, C3k, C2PSA, PSABlock, Attention, DWConv, SPPF)

Global model state stored in `std::cell::OnceCell` (single-threaded WASM — no need for `thread_local!`).

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

### Confidence Slider

- Range: `min=0.05, max=1.0, step=0.05, default=0.25` (per PRD US-3)
- Current value displayed next to slider
- Changes filter cached detections in JS — no WASM re-invocation

### Error Handling (JS side)

All errors display in the status area:
- Model fetch failure (network/404): "Failed to load model: [error]"
- Invalid SafeTensors / candle load error: "Failed to initialize model: [error]"
- `detect` called before `init_model`: "Model not loaded yet"
- WASM detect error: "Detection failed: [error]"
- Image too large: JS resizes to max 4096px before passing to WASM

### User Flow

1. **Page load**: Load WASM → fetch SafeTensors model → `init_model(bytes)` → status "Model loaded (X.X MB)"
2. **Image selected**: Decode via Canvas → get RGBA pixels → status "Detecting..." → `detect(pixels, w, h, 0.0)` → parse JSON → render boxes → update list → status "Detected N objects in Xms"
3. **Slider change**: Filter cached detections by threshold → re-render Canvas + list (no WASM call)

## 9. Project Structure

```
yolo26-rust-wasm/
├── Cargo.toml                      # candle-core, candle-nn, safetensors, wasm-bindgen, serde, serde_json
├── src/
│   ├── lib.rs                      # wasm-bindgen entry points, global model state
│   ├── model.rs                    # top-level Yolo26Model struct, load + forward
│   ├── model/
│   │   ├── backbone.rs             # Backbone (Conv stem + C3k2 stages + SPPF + C2PSA)
│   │   ├── neck.rs                 # FPN-PAN neck (Upsample + Concat + C3k2)
│   │   ├── head.rs                 # Detect head (end2end one2one, DFL, topk postprocess)
│   │   └── blocks.rs               # ConvBlock, Bottleneck, C3k2, C3k, C2PSA, PSABlock, Attention, DWConv, SPPF
│   ├── preprocess.rs               # image preprocessing pipeline
│   └── postprocess.rs              # output parsing, coord transform, JSON serialization
├── scripts/
│   └── download_model.sh           # Exports YOLO26n SafeTensors via ultralytics Python CLI
│                                    # Fallback: manual export instructions in README
├── weights/                        # .gitignore'd
│   └── yolo26n.safetensors
├── index.html                      # Demo UI (single file)
├── pkg/                            # wasm-pack build output (.gitignore'd)
├── tests/                          # Integration tests
├── PRD.md
├── CLAUDE.md
├── METHODOLOGY.md
├── FORMATTING.md
└── README.md
```

## 10. Dependencies (Cargo.toml)

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations, Device abstraction (CPU for WASM) |
| `candle-nn` | Neural network layers (Conv2d, BatchNorm, Linear), `VarBuilder` for weight loading |
| `safetensors` | SafeTensors file parsing (used by `VarBuilder::from_buffered_safetensors`) |
| `wasm-bindgen` | Rust ↔ JS FFI for WASM |
| `serde` + `serde_json` | Detection struct serialization to JSON |
| `console_error_panic_hook` | Readable panic messages in browser console (essential for WASM debugging) |
| `web-sys` | Access to `console::log` for WASM logging (optional) |

## 11. Build & Setup

```bash
# Prerequisites
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Export SafeTensors model
# download_model.sh runs: python -c "from ultralytics import YOLO; m = YOLO('yolo26n.pt'); m.export(format='safetensors')"
# This produces weights/yolo26n.safetensors (~5MB FP16)
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
| `model` | SafeTensors load from valid bytes; error on invalid bytes; output tensor shape [1, 300, 6]; individual block forward pass shape verification (ConvBlock, C3k2, SPPF, C2PSA, Detect) |

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
| SafeTensors key name mismatch between ultralytics export and candle VarBuilder | Dump key names from exported SafeTensors and verify against model layer naming; add key remapping layer if needed |
| YOLO26 architecture implementation complexity (C3k2, C2PSA, Attention, modified Detect) | Build incrementally: start from candle YOLOv8 base, add one block at a time, verify each with shape tests; reference ultralytics Python source as ground truth |
| candle WASM binary too large | Enable LTO, use `wasm-opt -Oz`, strip debug symbols; candle's WASM examples demonstrate reasonable sizes |
| Numerical precision differences between Python (PyTorch) and Rust (candle) | Compare intermediate tensor outputs layer-by-layer against PyTorch reference; accept small float differences, flag large deviations |
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
