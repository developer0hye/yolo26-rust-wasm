# YOLO26 Rust WASM Demo Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based YOLO26 object detection demo where all inference runs inside a Rust WASM module using candle (HuggingFace) as the native tensor framework, served from a single `index.html`.

**Architecture:** Rust WASM module implements YOLO26n architecture natively in candle, loads SafeTensors weights via `VarBuilder`, exports `init_model` and `detect` functions via wasm-bindgen. JS handles image decoding (Canvas API), WASM handles preprocess/inference/postprocess, JS renders bounding boxes on Canvas.

**Tech Stack:** Rust, wasm-pack, candle-core, candle-nn, safetensors, wasm-bindgen, serde_json, vanilla JS, Canvas API

**Spec:** `docs/superpowers/specs/2026-03-13-yolo26-rust-wasm-design.md`

**Reference:** `references/ultralytics/` (YOLO26 architecture source), candle `yolo-v8` example (candle patterns)

---

## YOLO26n Architecture Quick Reference

Channel sizes (after width=0.25 scaling): [16, 32, 64, 128, 256]. All depths n=1 (after depth=0.50 scaling).

**Backbone outputs for neck skip connections:** Layer 4 → [128, 80, 80], Layer 6 → [128, 40, 40], Layer 10 → [256, 20, 20]

**Detect head inputs:** Layer 16 → [64, 80, 80], Layer 19 → [128, 40, 40], Layer 22 → [256, 20, 20]

**SafeTensors weight key pattern:** `model.{layer_idx}.{subpath}` (e.g., `model.0.conv.weight`, `model.2.cv1.bn.weight`, `model.23.one2one_cv2.0.0.conv.weight`)

---

## Chunk 0: Branch Setup

### Task 0: Create Feature Branch via Git Worktree

Per `CLAUDE.md`: all changes go through PRs, use git worktrees for branch work.

- [ ] **Step 1: Create worktree and feature branch**

Run: `git worktree add ../yolo26-rust-wasm-feat-initial-implementation -b feat/initial-implementation`
Run: `cd ../yolo26-rust-wasm-feat-initial-implementation`

All subsequent tasks execute from inside this worktree directory.

---

## Chunk 1: Project Scaffolding

### Task 1: Project Setup

**Files:** Create `Cargo.toml`, `src/lib.rs`, `src/preprocess.rs`, `src/postprocess.rs`, `src/model/mod.rs`, `src/model/blocks.rs`, `src/model/backbone.rs`, `src/model/neck.rs`, `src/model/head.rs`. Modify `.gitignore`. Create `scripts/download_model.sh`, `FORMATTING.md`.

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "yolo26-rust-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
safetensors = "0.4"
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
console_error_panic_hook = "0.1"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
lto = true
opt-level = "z"
```

`rlib` in crate-type enables `cargo test` on native target. `cdylib` for wasm-pack.

- [ ] **Step 2: Create module stubs**

`src/lib.rs`: Declare modules (`pub mod preprocess; pub mod postprocess; pub mod model;`).

Create empty files: `src/preprocess.rs`, `src/postprocess.rs`, `src/model/mod.rs` (with `pub mod blocks; pub mod backbone; pub mod neck; pub mod head;`), `src/model/blocks.rs`, `src/model/backbone.rs`, `src/model/neck.rs`, `src/model/head.rs`.

- [ ] **Step 3: Update .gitignore**

Add: `weights/`, `pkg/`, `target/`

- [ ] **Step 4: Create scripts/download_model.sh**

```bash
#!/bin/bash
set -euo pipefail
mkdir -p weights
python3 -c "
from ultralytics import YOLO
from safetensors.torch import save_file
model = YOLO('yolo26n.pt')
sd = model.model.state_dict()
sd_f32 = {k: v.float() if v.is_floating_point() else v.float() for k, v in sd.items()}
save_file(sd_f32, 'weights/yolo26n.safetensors')
print(f'Saved {len(sd_f32)} tensors')
for k, v in sorted(sd_f32.items())[:20]:
    print(f'  {k}: {list(v.shape)}')
print('  ...')
"
echo "Done: weights/yolo26n.safetensors"
```

- [ ] **Step 5: Create FORMATTING.md**

Content: `` # Formatting\n\n`cargo fmt` ``

- [ ] **Step 6: Verify and commit**

Run: `cargo check && cargo fmt`
Commit: `feat: project scaffolding with candle dependencies`

---

## Chunk 2: Preprocessing (TDD)

### Task 2: Preprocessing Module

**Files:** `src/preprocess.rs`

Converts RGBA pixels → candle Tensor [1, 3, 640, 640] with letterbox transform.

- [ ] **Step 1: Write failing tests (Red)**

Tests in `#[cfg(test)] mod tests`:

1. `test_rgba_to_rgb` — 2×2 RGBA (16 bytes) → 12 RGB bytes, alpha stripped
2. `test_letterbox_square` — 640×640 input → scale=1.0, pad_x=0, pad_y=0
3. `test_letterbox_landscape` — 1280×640 → scale=0.5, pad_y=80
4. `test_letterbox_portrait` — 640×1280 → scale=0.5, pad_x=80
5. `test_normalize_range` — pixel 255 → 1.0, pixel 0 → 0.0
6. `test_output_tensor_shape` — output shape is [1, 3, 640, 640]
7. `test_letterbox_info` — LetterboxInfo fields correct

Run: `cargo test preprocess` → all fail

- [ ] **Step 2: Implement (Green)**

```rust
use candle_core::{Device, Result, Tensor};

pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

pub fn preprocess(
    rgba: &[u8], width: u32, height: u32, device: &Device,
) -> Result<(Tensor, LetterboxInfo)> {
    // 1. RGBA→RGB (drop every 4th byte)
    // 2. Letterbox: scale = min(640/w, 640/h), new_w/h = round(w/h * scale)
    //    pad_x = (640-new_w)/2, pad_y = (640-new_h)/2, fill gray=114
    // 3. Nearest-neighbor resize (no image crate — manual impl)
    // 4. Normalize [0,255]→[0.0,1.0]
    // 5. HWC→CHW, add batch dim → [1,3,640,640]
}
```

Nearest-neighbor resize: for each dst pixel (dx,dy), src pixel = (dx*src_w/dst_w, dy*src_h/dst_h).

Run: `cargo test preprocess` → all pass

- [ ] **Step 3: Refactor, format, commit**

Run: `cargo fmt && cargo test preprocess`
Commit: `feat: preprocessing pipeline (RGBA→RGB, letterbox, normalize)`

---

## Chunk 3: Postprocessing (TDD)

### Task 3: Postprocessing Module

**Files:** `src/postprocess.rs`

Parses model output tensor [1, 300, 6] → detection JSON.

- [ ] **Step 1: Write failing tests (Red)**

1. `test_parse_detections` — hand-crafted [1, 300, 6] tensor → correct Detection fields
2. `test_confidence_filter` — detections below threshold filtered out
3. `test_coordinate_transform` — xyxy model coords → original image coords via LetterboxInfo
4. `test_no_detections` — all below threshold → empty vec
5. `test_clamp_bounds` — coords clamped to [0, img_width/height]
6. `test_coco_class_name` — id 0 → "person", id 79 → "toothbrush"
7. `test_json_format` — serialized JSON matches expected structure

Run: `cargo test postprocess` → all fail

- [ ] **Step 2: Implement (Green)**

```rust
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub x: f32, pub y: f32, pub width: f32, pub height: f32,
    pub confidence: f32, pub class_id: u32, pub class_name: String,
}

#[derive(Serialize)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
    pub inference_time_ms: f64,
    pub image_width: u32, pub image_height: u32,
}

pub const COCO_CLASSES: [&str; 80] = ["person", "bicycle", /* ... */ "toothbrush"];

pub fn postprocess(
    output: &Tensor,              // [1, 300, 6] from Detect head
    letterbox: &LetterboxInfo,
    img_w: u32, img_h: u32,
    conf_thresh: f32,
) -> Result<Vec<Detection>> {
    // 1. Squeeze batch → [300, 6]
    // 2. Each row: [x1, y1, x2, y2, confidence, class_id]
    // 3. Filter by confidence
    // 4. Reverse letterbox: x = (x - pad_x) / scale
    // 5. xyxy → xywh: w = x2-x1, h = y2-y1
    // 6. Clamp to image bounds
    // 7. COCO class name lookup
}
```

Run: `cargo test postprocess` → all pass

- [ ] **Step 3: Commit**

Run: `cargo fmt && cargo test postprocess`
Commit: `feat: postprocessing (output parsing, coord transform, JSON)`

---

## Chunk 4: Model Building Blocks (TDD)

### Task 4: Building Blocks

**Files:** `src/model/blocks.rs`

All blocks use candle Conv2d + BatchNorm, loaded via VarBuilder. ConvBlock fuses BN into Conv via `absorb_bn()` for inference.

- [ ] **Step 1: Write shape tests (Red)**

Use `VarBuilder::zeros(DType::F32, &Device::Cpu)` for tests (no real weights needed for shape verification).

Tests:
1. `test_conv_block_shape` — ConvBlock(16→32, k=3, s=2): [1,16,640,640] → [1,32,320,320]
2. `test_conv_block_no_act` — ConvBlock with act=false: same shape, no SiLU
3. `test_bottleneck_shape` — Bottleneck(64→64, shortcut=true): shape preserved
4. `test_bottleneck_no_shortcut` — Bottleneck(64→32): [1,64,H,W] → [1,32,H,W]
5. `test_dwconv_shape` — DWConv(64, k=3): shape preserved
6. `test_c3k_shape` — C3k(128→128, n=2): shape preserved
7. `test_c3k2_no_c3k` — C3k2(32→64, c3k=F, e=0.25, n=1): [1,32,H,W] → [1,64,H,W]
8. `test_c3k2_with_c3k` — C3k2(128→128, c3k=T, e=0.5, n=1): shape preserved
9. `test_sppf_shape` — SPPF(256→256, k=5, n=3, shortcut=T): shape preserved
10. `test_attention_shape` — Attention(128, num_heads=2): shape preserved
11. `test_psa_block_shape` — PSABlock(128, num_heads=2): shape preserved
12. `test_c2psa_shape` — C2PSA(256→256, n=1): shape preserved

Run: `cargo test blocks` → all fail

- [ ] **Step 2: Implement ConvBlock**

```rust
pub struct ConvBlock {
    conv: candle_nn::Conv2d, // BN absorbed
    act: bool,
}

impl ConvBlock {
    pub fn load(
        vb: VarBuilder, c1: usize, c2: usize,
        k: usize, s: usize, groups: usize, act: bool,
    ) -> Result<Self> {
        let p = k / 2;
        let cfg = candle_nn::Conv2dConfig {
            stride: s, padding: p, groups, ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?;
        let bn = candle_nn::batch_norm(c2, 1e-5, vb.pp("bn"))?;
        let conv = conv.absorb_bn(&bn)?;
        Ok(Self { conv, act })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        if self.act { candle_nn::ops::silu(&x) } else { Ok(x) }
    }
}
```

Weight keys: `{prefix}.conv.weight`, `{prefix}.bn.weight`, `{prefix}.bn.bias`, `{prefix}.bn.running_mean`, `{prefix}.bn.running_var`.

- [ ] **Step 3: Implement Bottleneck**

```rust
pub struct Bottleneck {
    cv1: ConvBlock, // (c1→c2, k[0], s=1)
    cv2: ConvBlock, // (c2→c2, k[1], s=1)
    shortcut: bool, // add residual if c1==c2
}
```

Default k=(3,3), e=1.0. `cv1(c1→c2*e, k[0])`, `cv2(c2*e→c2, k[1])`.

- [ ] **Step 4: Implement DWConv**

Just ConvBlock with `groups=c1` (depthwise). Wrapper function:

```rust
pub fn dwconv(vb: VarBuilder, c: usize, k: usize, s: usize, act: bool) -> Result<ConvBlock> {
    ConvBlock::load(vb, c, c, k, s, c, act)
}
```

- [ ] **Step 5: Implement C3k (C3 variant)**

```rust
pub struct C3k {
    cv1: ConvBlock, // c1→c_ (1×1), where c_ = c2*e (e=1.0)
    cv2: ConvBlock, // c1→c_ (1×1)
    cv3: ConvBlock, // 2*c_→c2 (1×1)
    m: Vec<Bottleneck>, // n Bottleneck(c_, c_, k=(3,3))
}
```

Forward: `cv3(cat(m(cv1(x)), cv2(x), dim=1))`

Weight keys: `{p}.cv1.*`, `{p}.cv2.*`, `{p}.cv3.*`, `{p}.m.{i}.cv1.*`, `{p}.m.{i}.cv2.*`

- [ ] **Step 6: Implement C3k2 (C2f variant)**

C2f with configurable branch type. Hidden channels: `c_ = floor(c2 * e)`.

```rust
pub struct C3k2 {
    cv1: ConvBlock, // c1→2*c_ (1×1)
    cv2: ConvBlock, // (2+n)*c_→c2 (1×1)
    m: Vec<C3k2Branch>,
}

pub enum C3k2Branch {
    BottleneckBranch(Bottleneck),
    C3kBranch(C3k),
    C3kPsaBranch(C3k), // C3k with PSABlock replacing internal m (attn=True)
}
```

Forward (C2f pattern):
```
y = cv1(x).chunk(2, dim=1)  // split into 2 halves of c_ each
for m_i in m: y.push(m_i(y.last()))
cv2(cat(y, dim=1))
```

When `attn=True`: last C3k in m has its internal `self.m` replaced by PSABlock. Handle by storing the PSABlock inside C3k when loading.

**Channel math per usage:**
| Layer | c1 | c2 | c3k | e | n | c_ | cv1_out | cv2_in |
|-------|----|----|-----|---|---|----|---------|--------|
| 2 | 32 | 64 | F | 0.25 | 1 | 16 | 32 | 48 |
| 4 | 64 | 128 | F | 0.25 | 1 | 32 | 64 | 96 |
| 6 | 128 | 128 | T | 0.5 | 1 | 64 | 128 | 192 |
| 8 | 256 | 256 | T | 0.5 | 1 | 128 | 256 | 384 |
| 13 | 384 | 128 | T | 0.5 | 1 | 64 | 128 | 192 |
| 16 | 256 | 64 | T | 0.5 | 1 | 32 | 64 | 96 |
| 19 | 192 | 128 | T | 0.5 | 1 | 64 | 128 | 192 |
| 22 | 384 | 256 | T | 0.5 | 1 | 128 | 256 | 384 |

- [ ] **Step 7: Implement SPPF**

```rust
pub struct Sppf {
    cv1: ConvBlock, // c1→c1 (1×1, act=false!)
    cv2: ConvBlock, // c1*(n+1)→c2 (1×1, act=true)
    k: usize,       // MaxPool kernel (5)
    n: usize,       // Sequential maxpool count (3)
    shortcut: bool, // residual add (True for YOLO26)
}
```

Forward: `x=cv1(input)` → sequential `maxpool2d(k=5,s=1,p=2)` n times → `cat([x, pool1, pool2, pool3])` → `cv2(cat)` + input (if shortcut).

YOLO26n layer 9: c1=c2=256, k=5, n=3, shortcut=True.

Note: candle doesn't have built-in maxpool2d. Implement via `Tensor::max_pool2d(k, s=1)` with manual padding, or use candle's `candle_nn::pool::avg_pool2d` pattern — check candle API. If unavailable, implement using unfold+max operations.

- [ ] **Step 8: Implement Attention**

```rust
pub struct Attention {
    qkv: ConvBlock,  // Conv(dim→dim+2*nh_kd, k=1, act=false)
    proj: ConvBlock,  // Conv(dim→dim, k=1, act=false)
    pe: ConvBlock,    // DWConv(dim→dim, k=3, groups=dim, act=false)
    num_heads: usize,
    key_dim: usize,
    head_dim: usize,
    scale: f64,
}
```

For dim=128, num_heads=2: head_dim=64, key_dim=32 (attn_ratio=0.5), nh_kd=64, qkv_out=256.

Forward:
```
qkv = self.qkv(x)  // [B, 256, H, W]
reshape to [B, num_heads, key_dim*2+head_dim, N] where N=H*W
split → q[B,nh,kd,N], k[B,nh,kd,N], v[B,nh,hd,N]
attn = softmax(q^T @ k * scale)  // [B,nh,N,N]
out = (v @ attn^T).reshape(B,C,H,W) + pe(v.reshape(B,C,H,W))
out = proj(out)
```

- [ ] **Step 9: Implement PSABlock and C2PSA**

PSABlock:
```rust
pub struct PsaBlock {
    attn: Attention,
    ffn_0: ConvBlock, // Conv(c→2c, k=1, act=true)
    ffn_1: ConvBlock, // Conv(2c→c, k=1, act=false)
}
```
Forward: `x = x + attn(x); x = x + ffn_1(ffn_0(x))`

C2PSA:
```rust
pub struct C2psa {
    cv1: ConvBlock, // c1→2*c (1×1)
    cv2: ConvBlock, // 2*c→c1 (1×1)
    m: Vec<PsaBlock>,
    c: usize, // split channel = c1*e
}
```
Forward: `a, b = cv1(x).split(c, dim=1); b = m(b); cv2(cat(a, b))`

YOLO26n layer 10: c1=c2=256, c=128, n=1. PSABlock dim=128, num_heads=2.

- [ ] **Step 10: Run tests, format, commit**

Run: `cargo fmt && cargo test blocks`
Commit: `feat: model building blocks (ConvBlock, Bottleneck, C3k2, C3k, SPPF, C2PSA, Attention)`

---

## Chunk 5: Backbone & Neck

### Task 5: Backbone

**Files:** `src/model/backbone.rs`

Layers 0–10. Returns outputs at layers 4, 6, 10 for neck skip connections.

- [ ] **Step 1: Write shape test (Red)**

`test_backbone_output_shapes` — input [1,3,640,640] → p3 [1,128,80,80], p4 [1,128,40,40], p5 [1,256,20,20].

- [ ] **Step 2: Implement Backbone**

```rust
pub struct Backbone {
    l0: ConvBlock,  // 3→16, k=3, s=2
    l1: ConvBlock,  // 16→32, k=3, s=2
    l2: C3k2,       // 32→64, c3k=F, e=0.25, n=1
    l3: ConvBlock,  // 64→64, k=3, s=2
    l4: C3k2,       // 64→128, c3k=F, e=0.25, n=1
    l5: ConvBlock,  // 128→128, k=3, s=2
    l6: C3k2,       // 128→128, c3k=T, e=0.5, n=1
    l7: ConvBlock,  // 128→256, k=3, s=2
    l8: C3k2,       // 256→256, c3k=T, e=0.5, n=1
    l9: Sppf,       // 256→256, k=5, n=3, shortcut=T
    l10: C2psa,     // 256→256, n=1, e=0.5
}

pub struct BackboneOutput {
    pub p3: Tensor,  // layer 4 output [128, 80, 80]
    pub p4: Tensor,  // layer 6 output [128, 40, 40]
    pub p5: Tensor,  // layer 10 output [256, 20, 20]
}

impl Backbone {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        // vb prefix should be "model"
        // Each layer: vb.pp("{layer_idx}")
        let l0 = ConvBlock::load(vb.pp("0"), 3, 16, 3, 2, 1, true)?;
        // ... etc
    }

    pub fn forward(&self, x: &Tensor) -> Result<BackboneOutput> {
        let x = self.l0.forward(x)?;
        // ... run through all layers, save p3/p4/p5
    }
}
```

- [ ] **Step 3: Commit**

Run: `cargo fmt && cargo test backbone`
Commit: `feat: YOLO26n backbone (layers 0-10)`

### Task 6: Neck

**Files:** `src/model/neck.rs`

FPN-PAN neck (layers 11–22). Takes backbone outputs, returns 3-scale features for Detect head.

- [ ] **Step 1: Write shape test (Red)**

`test_neck_output_shapes` — p3/p4/p5 backbone outputs → small [1,64,80,80], medium [1,128,40,40], large [1,256,20,20].

- [ ] **Step 2: Implement Neck**

```rust
pub struct Neck {
    // FPN (top-down)
    l13: C3k2,     // 384→128, c3k=T, n=1
    l16: C3k2,     // 256→64, c3k=T, n=1
    // PAN (bottom-up)
    l17: ConvBlock, // 64→64, k=3, s=2
    l19: C3k2,     // 192→128, c3k=T, n=1
    l20: ConvBlock, // 128→128, k=3, s=2
    l22: C3k2,     // 384→256, c3k=T, e=0.5, n=1, attn=T
}

pub struct NeckOutput {
    pub small: Tensor,  // layer 16 [64, 80, 80]
    pub medium: Tensor, // layer 19 [128, 40, 40]
    pub large: Tensor,  // layer 22 [256, 20, 20]
}
```

Upsample: `Tensor::upsample_nearest2d(h*2, w*2)` (candle built-in).
Concat: `Tensor::cat(&[a, b], 1)` along channel dim.

Forward flow:
```
l11 = upsample(p5)                       // [256,40,40]
l12 = cat(l11, p4)                        // [384,40,40]
l13 = c3k2(l12)                           // [128,40,40]
l14 = upsample(l13)                       // [128,80,80]
l15 = cat(l14, p3)                        // [256,80,80]
l16 = c3k2(l15)                           // [64,80,80]  → small
l17 = conv(l16)                           // [64,40,40]
l18 = cat(l17, l13)                       // [192,40,40]
l19 = c3k2(l18)                           // [128,40,40] → medium
l20 = conv(l19)                           // [128,20,20]
l21 = cat(l20, p5)                        // [384,20,20]
l22 = c3k2(l21, attn=T)                  // [256,20,20] → large
```

- [ ] **Step 3: Commit**

Run: `cargo fmt && cargo test neck`
Commit: `feat: YOLO26n FPN-PAN neck (layers 11-22)`

---

## Chunk 6: Detect Head & Full Model

### Task 7: Detect Head

**Files:** `src/model/head.rs`

End2end Detect head (layer 23) with topk postprocessing. Uses one2one branches only.

- [ ] **Step 1: Write tests (Red)**

1. `test_make_anchors` — verify anchor count: 80×80+40×40+20×20 = 8400
2. `test_dist2bbox_xyxy` — known anchor + distance → correct xyxy coords
3. `test_detect_output_shape` — input 3 feature maps → output [1, 300, 6]

- [ ] **Step 2: Implement make_anchors and dist2bbox**

```rust
/// Generate anchor points and stride tensor for all feature map scales.
/// Returns (anchors [2, N], strides [1, N]) where N = sum of h_i*w_i.
pub fn make_anchors(
    feat_sizes: &[(usize, usize)], // [(80,80), (40,40), (20,20)]
    strides: &[f32],               // [8.0, 16.0, 32.0]
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // For each scale: meshgrid of (0..w)+0.5, (0..h)+0.5
    // Stack (sx, sy) → [h*w, 2], stride_tensor [h*w, 1]
    // Cat all scales, transpose → anchors [2, N], strides [1, N]
}

/// dist2bbox in xyxy mode (for end2end): x1y1 = anchor - lt, x2y2 = anchor + rb
pub fn dist2bbox_xyxy(distance: &Tensor, anchors: &Tensor) -> Result<Tensor> {
    let (lt, rb) = distance.chunk(2, 1)?; // [B, 2, N] each
    let x1y1 = anchors.broadcast_sub(&lt)?;
    let x2y2 = anchors.broadcast_add(&rb)?;
    Tensor::cat(&[x1y1, x2y2], 1) // [B, 4, N]
}
```

- [ ] **Step 3: Implement Detect struct**

```rust
pub struct Detect {
    // one2one box branch per scale: Conv→Conv→Conv2d
    one2one_cv2: Vec<(ConvBlock, ConvBlock, candle_nn::Conv2d)>,
    // one2one cls branch per scale: (DWConv+Conv)→(DWConv+Conv)→Conv2d
    one2one_cv3: Vec<(ConvBlock, ConvBlock, ConvBlock, ConvBlock, candle_nn::Conv2d)>,
    nc: usize,    // 80
    reg_max: usize, // 1
    strides: Vec<f32>, // [8, 16, 32]
}
```

YOLO26n head channels: c2=16 (box), c3=80 (cls), input ch=[64, 128, 256].

Weight key pattern for one2one branches:
- Box: `model.23.one2one_cv2.{s}.0.conv.weight` (Conv), `.1.conv.weight` (Conv), `.2.weight` (Conv2d+bias)
- Cls: `model.23.one2one_cv3.{s}.0.0.conv.weight` (DWConv), `.0.1.conv.weight` (Conv), `.1.0.conv.weight` (DWConv), `.1.1.conv.weight` (Conv), `.2.weight` (Conv2d+bias)

- [ ] **Step 4: Implement forward + postprocess**

```rust
impl Detect {
    pub fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        // 1. Per scale: box = cv2(feat), cls = cv3(feat)
        //    box: [B, 4, H_i*W_i], cls: [B, 80, H_i*W_i]
        // 2. Cat across scales: boxes [B, 4, 8400], scores [B, 80, 8400]
        // 3. Compute anchors from feature map sizes
        // 4. dbox = dist2bbox_xyxy(boxes, anchors) * strides  // [B, 4, 8400]
        // 5. cls_scores = sigmoid(scores)                      // [B, 80, 8400]
        // 6. y = cat(dbox, cls_scores, dim=1)                  // [B, 84, 8400]
        // 7. y = y.permute(0, 2, 1)                            // [B, 8400, 84]
        // 8. topk_postprocess(y)                                // [B, 300, 6]
    }

    fn topk_postprocess(&self, preds: &Tensor) -> Result<Tensor> {
        // preds: [B, N, 84] where N=8400
        // 1. Split → boxes [B, N, 4], scores [B, N, 80]
        // 2. max_scores = scores.max(dim=-1) → [B, N]
        // 3. topk_indices = max_scores.topk(300) → [B, 300]
        // 4. Gather boxes and scores by topk indices
        // 5. For each selected: find best class → conf + class_id
        // 6. Return [B, 300, 6] = [x1, y1, x2, y2, conf, class_id]
    }
}
```

Note: Simplified topk vs ultralytics' two-stage topk. The simplified version picks top-300 anchors by max class score, then extracts the best class per anchor. This is equivalent for the end user.

- [ ] **Step 5: Commit**

Run: `cargo fmt && cargo test head`
Commit: `feat: Detect head with end2end topk postprocessing`

### Task 8: Top-Level Model

**Files:** `src/model/mod.rs`

Ties together Backbone + Neck + Detect into Yolo26Model.

- [ ] **Step 1: Write shape test (Red)**

`test_model_output_shape` — input [1,3,640,640] → output [1,300,6]

- [ ] **Step 2: Implement Yolo26Model**

```rust
pub struct Yolo26Model {
    backbone: Backbone,
    neck: Neck,
    head: Detect,
}

impl Yolo26Model {
    pub fn load(weights_bytes: &[u8], device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, device)?;
        let vb = vb.pp("model");  // weight keys start with "model."
        let backbone = Backbone::load(vb.clone())?;
        let neck = Neck::load(vb.clone())?;
        let head = Detect::load(vb.pp("23"), &[64, 128, 256])?;
        Ok(Self { backbone, neck, head })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let bb = self.backbone.forward(input)?;
        let neck = self.neck.forward(&bb)?;
        self.head.forward(&[&neck.small, &neck.medium, &neck.large])
    }
}
```

- [ ] **Step 3: Commit**

Run: `cargo fmt && cargo test model`
Commit: `feat: top-level Yolo26Model (backbone + neck + head)`

---

## Chunk 7: WASM Integration

### Task 9: WASM Bindings

**Files:** `src/lib.rs`

Wire up `init_model` and `detect` wasm-bindgen exports using `OnceLock` for global model state.

- [ ] **Step 1: Write test for detect pipeline (Red)**

`test_detect_pipeline` — Native test: load real SafeTensors → preprocess synthetic image → forward → postprocess → verify JSON structure and detection count > 0 (if real weights available). Skip if weights file not present (`#[ignore]`).

- [ ] **Step 2: Implement lib.rs**

```rust
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

mod model;
mod preprocess;
mod postprocess;

use model::Yolo26Model;
use preprocess::{preprocess, LetterboxInfo};
use postprocess::{postprocess, DetectionResult};

static MODEL: OnceLock<Yolo26Model> = OnceLock::new();

#[wasm_bindgen]
pub fn init_model(weights: &[u8]) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let device = candle_core::Device::Cpu;
    let model = Yolo26Model::load(weights, &device)
        .map_err(|e| JsValue::from_str(&format!("Failed to load model: {e}")))?;
    MODEL.set(model)
        .map_err(|_| JsValue::from_str("Model already initialized"))?;
    Ok(())
}

#[wasm_bindgen]
pub fn detect(
    pixels: &[u8], width: u32, height: u32, confidence_threshold: f32,
) -> Result<String, JsValue> {
    let model = MODEL.get()
        .ok_or_else(|| JsValue::from_str("Model not loaded yet"))?;
    let device = candle_core::Device::Cpu;
    let start = js_sys::Date::now(); // WASM timestamp

    let (input, letterbox) = preprocess(pixels, width, height, &device)
        .map_err(|e| JsValue::from_str(&format!("Preprocess error: {e}")))?;
    let output = model.forward(&input)
        .map_err(|e| JsValue::from_str(&format!("Inference error: {e}")))?;
    let detections = postprocess(&output, &letterbox, width, height, confidence_threshold)
        .map_err(|e| JsValue::from_str(&format!("Postprocess error: {e}")))?;

    let elapsed = js_sys::Date::now() - start;
    let result = DetectionResult {
        detections,
        inference_time_ms: elapsed,
        image_width: width,
        image_height: height,
    };
    serde_json::to_string(&result)
        .map_err(|e| JsValue::from_str(&format!("JSON error: {e}")))
}
```

Add `js-sys` to dependencies for `Date::now()`. For native tests, use `std::time::Instant` behind `#[cfg(not(target_arch = "wasm32"))]`.

- [ ] **Step 3: Commit**

Run: `cargo fmt && cargo test`
Commit: `feat: WASM bindings (init_model, detect) with OnceLock state`

---

## Chunk 8: WASM Build & HTML UI

### Task 10: HTML Demo

**Files:** Create `index.html`

- [ ] **Step 1: Build WASM and verify**

Run: `RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release`

Verify `pkg/` contains `yolo26_rust_wasm.js` and `yolo26_rust_wasm_bg.wasm`.

If build errors occur, fix Cargo.toml dependencies or code issues before proceeding.

- [ ] **Step 2: Export model weights**

Run: `chmod +x scripts/download_model.sh && ./scripts/download_model.sh`

Verify `weights/yolo26n.safetensors` exists and print file size.

- [ ] **Step 3: Create index.html**

Single-file demo per spec. Key sections:

**HTML structure:**
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YOLO26 Rust WASM Demo</title>
  <style>/* Responsive CSS: max-width container, mobile stacking at 768px */</style>
</head>
<body>
  <h1>YOLO26 Rust WASM Demo</h1>
  <p>Browser-based object detection powered by Rust + WASM</p>
  <div id="status">Loading model...</div>
  <input type="file" id="file-input" accept="image/*">
  <div id="drop-zone">or drag-and-drop here</div>
  <label>Confidence: <span id="conf-val">0.25</span>
    <input type="range" id="conf-slider" min="0.05" max="1.0" step="0.05" value="0.25">
  </label>
  <canvas id="canvas"></canvas>
  <div id="detections"></div>
  <script type="module">/* JS logic */</script>
</body>
</html>
```

**JS logic (inside `<script type="module">`):**
```js
import init, { init_model, detect } from './pkg/yolo26_rust_wasm.js';

let cachedDetections = null;
let currentImage = null;

async function setup() {
  await init();
  const resp = await fetch('weights/yolo26n.safetensors');
  if (!resp.ok) throw new Error(`Failed to load model: ${resp.status}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  init_model(bytes);
  status.textContent = `Model loaded (${(bytes.length/1e6).toFixed(1)} MB)`;
}

function runDetection(img) {
  // Draw to offscreen canvas → getImageData → detect(rgba, w, h, 0.0)
  // Parse JSON → cache detections → render
}

function renderDetections(threshold) {
  // Filter cached by threshold → draw boxes on canvas
  // Color: hsl(class_id * 137.5 % 360, 70%, 50%)
  // strokeRect 2px + fillRect alpha 0.15 + label above box
}

// Event listeners: file input, drag-and-drop, slider
```

**Responsive CSS:**
- Container max-width: 900px, centered
- Canvas: width 100%, height auto
- Below 768px: stack everything vertically, detection list scrollable
- `accept="image/*"` for mobile camera

**Bounding box rendering:**
- Canvas resolution matches original image size (set canvas.width/height)
- CSS `max-width: 100%` scales display
- Color per class: HSL `hue = class_id * 137.5 % 360`, saturation 70%, lightness 50%
- Label: colored rect background + white text above box

**Error handling in JS:**
- Model fetch failure: display in #status
- init_model error: display in #status
- detect error: display in #status
- Image too large: resize to max 4096px before passing to WASM

- [ ] **Step 4: Manual smoke test**

Run: `npx serve .` → open `http://localhost:3000/index.html`
Verify: model loads, image upload works, bounding boxes render, slider filters.

- [ ] **Step 5: Commit**

Run: `cargo fmt`
Commit: `feat: HTML demo UI with Canvas rendering and confidence slider`

---

## Chunk 9: Integration Test, README & PR

### Task 11: Integration Test & Polish

**Files:** Modify `README.md`, verify all tests pass.

- [ ] **Step 1: Run full test suite**

Run: `cargo test`

All unit tests for preprocess, postprocess, blocks, backbone, neck, head, model should pass.

- [ ] **Step 2: WASM integration test (if possible)**

If `wasm-pack test --headless --chrome` is available:

Create `tests/web.rs`:
```rust
use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_init_and_detect() {
    // Load model bytes, init, run detect on synthetic image
    // Verify JSON output structure
}
```

If headless browser not available, mark as `#[ignore]` with manual test instructions.

- [ ] **Step 3: Update README.md**

Add: project description, prerequisites (Rust, wasm-pack, Python+ultralytics for model export), build instructions, usage instructions, architecture overview.

- [ ] **Step 4: Commit**

Run: `cargo fmt && cargo test`
Commit: `docs: README with build instructions and architecture overview`

### Task 12: Create Pull Request

- [ ] **Step 1: Push branch**

Run: `git push -u origin feat/initial-implementation`

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "feat: YOLO26 Rust WASM browser demo" --body "$(cat <<'EOF'
## Summary
- Implements YOLO26n object detection model natively in Rust using candle framework
- Compiles to WASM for in-browser inference with no server dependency
- Single index.html demo with drag-and-drop, Canvas rendering, confidence slider
- SafeTensors weight loading, ~5MB FP16 model

## Architecture
Rust WASM (candle): preprocess → backbone → FPN-PAN neck → Detect head → postprocess
JS: image decode → WASM inference → Canvas bounding box rendering

## Test plan
- [ ] `cargo test` — all unit tests pass (preprocess, postprocess, model blocks, backbone, neck, head)
- [ ] `wasm-pack build --target web --release` — WASM builds successfully
- [ ] Manual: load index.html, upload COCO image, verify bounding boxes render correctly
- [ ] Manual: confidence slider filters detections without re-running inference
- [ ] Manual: mobile responsive layout (< 768px)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Follow PR merge procedure from CLAUDE.md**

Wait for CI, review diff, merge when ready.
