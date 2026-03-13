# Reference Projects Index

| Project | Location | Relevance |
|---------|----------|-----------|
| ultralytics/ultralytics | `references/ultralytics/` (cloned) | YOLO26 architecture source of truth: model config, layer implementations, detection head |
| candle WASM YOLO example | Remote: `huggingface/candle/candle-wasm-examples/yolo/` | WASM YOLO inference pattern, wasm-bindgen usage, Cargo.toml |
| candle YOLOv8 model | Remote: `huggingface/candle/candle-examples/examples/yolo-v8/model.rs` | YOLOv8 candle architecture (base to modify for YOLO26) |

## Key Files in ultralytics (local)

- `ultralytics/cfg/models/26/yolo26.yaml` — YOLO26n model config (layers, channels, backbone/neck/head)
- `ultralytics/nn/modules/block.py` — C3k2, C2PSA, SPPF, PSABlock, Attention, C3k, Bottleneck
- `ultralytics/nn/modules/head.py` — Detect (end2end, one2one heads, dist2bbox, make_anchors)
- `ultralytics/nn/modules/conv.py` — Conv, DWConv, autopad
- `ultralytics/utils/tal.py` — make_anchors, dist2bbox
