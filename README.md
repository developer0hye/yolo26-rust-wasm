# YOLO26 Rust WASM

Real-time object detection running entirely in the browser — no server, no upload, no API calls. All YOLO26 model scales (n/s/m/l/x) are implemented natively in Rust using [candle](https://github.com/huggingface/candle) and compiled to WebAssembly.

![Demo](assets/demo.png)

## Why This Exists

Most browser-based ML demos rely on ONNX Runtime or TensorFlow.js. This project takes a different approach: the entire YOLO26 architecture is implemented from scratch in Rust and compiled to WASM. Every layer — convolutions, batch norm, attention, detect head — runs as native Rust code in the browser.

This means:

- **Zero server dependency** — inference happens on the client. No data leaves the browser.
- **No runtime framework** — no ONNX, no TF.js. Just Rust → WASM → Canvas.
- **Portable** — one `.wasm` binary + SafeTensors weights. Works anywhere WebAssembly runs.

## Features

- All YOLO26 scales (n/s/m/l/x) natively implemented in Rust
- In-browser model selector for switching between scales
- FP16 SafeTensors weights (auto-converted to FP32 at load time — zero accuracy loss)
- WASM SIMD128 acceleration for vectorized matrix operations
- Web Worker inference (non-blocking UI)
- EXIF-aware image handling for correct orientation
- Confidence threshold slider (filters without re-running inference)
- Click-to-zoom full resolution view
- Responsive viewport-fit layout
- 36 unit tests covering all building blocks and full pipeline

## Architecture

```
Browser (Next.js + Web Worker)
  │ File → createImageBitmap (EXIF-normalized)
  │ → Canvas → RGBA pixels
  ▼
Rust WASM Module (candle)
  ├── preprocess.rs    → Bilinear resize, letterbox 640×640, normalize, HWC→CHW
  ├── model/
  │   ├── config.rs    → ModelScale (n/s/m/l/x) channel/repeat scaling
  │   ├── backbone.rs  → Conv, C3k2, SPPF, C2PSA (layers 0-10)
  │   ├── neck.rs      → FPN-PAN feature fusion (layers 11-22)
  │   └── head.rs      → Detect: end2end, topk-300, NMS-free (layer 23)
  └── postprocess.rs   → [1,300,6] → coord transform → JSON
  ▼
Canvas 2D (bounding boxes + labels)
```

## Prerequisites

- Rust (stable) with `wasm32-unknown-unknown` target
- `wasm-pack`: `cargo install wasm-pack`
- Node.js 18+ (for the web app)
- Python 3 + `ultralytics` + `safetensors` (for model export only)

## Quick Start

### 1. Export Model Weights

```bash
pip install ultralytics safetensors
python scripts/export_all_sizes.py
```

Exports all 5 scales as FP16 SafeTensors (~240 MB total):

| Model | Params | Weights |
|-------|--------|---------|
| yolo26n | 2.6M | 5 MB |
| yolo26s | 10.0M | 20 MB |
| yolo26m | 21.9M | 44 MB |
| yolo26l | 26.3M | 53 MB |
| yolo26x | 59.0M | 118 MB |

### 2. Build WASM

```bash
wasm-pack build --target web --out-dir web/public/wasm
```

SIMD is enabled automatically via `.cargo/config.toml`.

### 3. Run Web App

```bash
mkdir -p web/public/weights
cp yolo26*.safetensors web/public/weights/
cd web
npm install && npm run dev
```

Open http://localhost:3000.

## WASM API

```rust
// Load SafeTensors model with scale identifier
init_model(weights: &[u8], model_name: &str) -> Result<(), JsValue>
// model_name: "yolo26n", "yolo26s", "yolo26m", "yolo26l", or "yolo26x"

// Run detection on RGBA pixels
detect(pixels: &[u8], width: u32, height: u32, confidence_threshold: f32) -> Result<String, JsValue>
```

## Testing

```bash
cargo test
```

49 tests covering preprocessing, postprocessing, all building blocks (ConvBlock, Bottleneck, C3k2, C3k, SPPF, C2PSA, Attention), model scaling config, backbone, neck, detect head, and full pipeline across multiple scales.

## Acknowledgments

The YOLO26 architecture is designed by [Ultralytics](https://github.com/ultralytics/ultralytics). Their consistent work on pushing real-time object detection forward — from YOLOv5 through YOLO26 — makes projects like this possible. Model weights are exported from the official Ultralytics package.
