# YOLO26 Rust WASM Demo

Browser-based YOLO26 object detection demo. All inference runs entirely in the browser via Rust compiled to WebAssembly using [candle](https://github.com/huggingface/candle).

## Features

- YOLO26n model implemented natively in Rust (candle framework)
- SafeTensors weight loading (~5MB FP16)
- Single `index.html` — no framework dependencies
- Image upload with drag-and-drop
- Canvas bounding box rendering with class-specific colors
- Confidence threshold slider (filters without re-running inference)
- Mobile responsive layout
- WASM SIMD128 acceleration

## Prerequisites

- Rust (stable)
- `wasm-pack`: `cargo install wasm-pack`
- `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- Python 3 + `ultralytics` + `safetensors` (for model export only)

## Setup

### 1. Export Model Weights

```bash
pip install ultralytics safetensors
./scripts/download_model.sh
```

This downloads YOLO26n and exports weights to `weights/yolo26n.safetensors`.

### 2. Build WASM

```bash
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release
```

### 3. Run

```bash
npx serve .
# Open http://localhost:3000
```

## Architecture

```
Browser (index.html + vanilla JS)
  │ RGBA pixels (Canvas API)
  ▼
Rust WASM Module (candle)
  ├── preprocess.rs    → RGBA→RGB, letterbox 640×640, normalize, HWC→CHW
  ├── model/
  │   ├── blocks.rs    → ConvBlock, Bottleneck, C3k2, C3k, SPPF, C2PSA, Attention
  │   ├── backbone.rs  → Layers 0-10 (Conv + C3k2 + SPPF + C2PSA)
  │   ├── neck.rs      → Layers 11-22 (FPN-PAN feature fusion)
  │   └── head.rs      → Layer 23 (Detect: end2end, topk-300, NMS-free)
  └── postprocess.rs   → Parse [1,300,6], coord transform, JSON
  │ JSON (detection results)
  ▼
Canvas 2D (bounding boxes + labels)
```

## WASM API

```rust
// Load SafeTensors model (called once)
init_model(weights: &[u8]) -> Result<(), JsValue>

// Run detection on RGBA pixels
detect(pixels: &[u8], width: u32, height: u32, confidence_threshold: f32) -> Result<String, JsValue>
```

## Testing

```bash
cargo test
```

34 tests covering preprocessing, postprocessing, all building blocks, backbone, neck, detect head, and full pipeline.
