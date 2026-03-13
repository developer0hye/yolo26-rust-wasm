use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, Conv2dConfig, VarBuilder};

use super::blocks::ConvBlock;

/// YOLO26n Detect head (layer 23): end2end with one2one branches.
/// Input: 3 feature maps at different scales.
/// Output: [B, 300, 6] = [x1, y1, x2, y2, confidence, class_id]
pub struct Detect {
    /// Box branch per scale: Conv(ch→c2,3) → Conv(c2→c2,3) → Conv2d(c2→4,1)
    box_branches: Vec<BoxBranch>,
    /// Cls branch per scale: (DWConv+Conv) → (DWConv+Conv) → Conv2d(c3→nc,1)
    cls_branches: Vec<ClsBranch>,
    strides: Vec<f32>,
    nc: usize,
    max_det: usize,
}

struct BoxBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
    cv2: candle_nn::Conv2d, // final Conv2d with bias
}

struct ClsBranch {
    dw0: ConvBlock,         // DWConv(ch, ch, 3)
    cv0: ConvBlock,         // Conv(ch, c3, 1)
    dw1: ConvBlock,         // DWConv(c3, c3, 3)
    cv1: ConvBlock,         // Conv(c3, c3, 1)
    cv2: candle_nn::Conv2d, // Conv2d(c3, nc, 1) with bias
}

impl Detect {
    pub fn load(vb: VarBuilder, input_channels: &[usize], nc: usize) -> Result<Self> {
        let reg_max: usize = 1;
        let c2: usize = 16_usize.max(input_channels[0] / 4).max(reg_max * 4); // box channels
        let c3: usize = input_channels[0].max(nc.min(100)); // cls channels

        let strides: Vec<f32> = vec![8.0, 16.0, 32.0];
        let conv2d_cfg = Conv2dConfig::default();

        let mut box_branches: Vec<BoxBranch> = Vec::new();
        let mut cls_branches: Vec<ClsBranch> = Vec::new();

        for (i, &ch) in input_channels.iter().enumerate() {
            let bvb = vb.pp("one2one_cv2").pp(i.to_string());
            let box_branch = BoxBranch {
                cv0: ConvBlock::load(bvb.pp("0"), ch, c2, 3, 1, 1, true)?,
                cv1: ConvBlock::load(bvb.pp("1"), c2, c2, 3, 1, 1, true)?,
                cv2: conv2d(c2, 4 * reg_max, 1, conv2d_cfg, bvb.pp("2"))?,
            };
            box_branches.push(box_branch);

            let cvb = vb.pp("one2one_cv3").pp(i.to_string());
            let cls_branch = ClsBranch {
                dw0: ConvBlock::load(cvb.pp("0").pp("0"), ch, ch, 3, 1, ch, true)?,
                cv0: ConvBlock::load(cvb.pp("0").pp("1"), ch, c3, 1, 1, 1, true)?,
                dw1: ConvBlock::load(cvb.pp("1").pp("0"), c3, c3, 3, 1, c3, true)?,
                cv1: ConvBlock::load(cvb.pp("1").pp("1"), c3, c3, 1, 1, 1, true)?,
                cv2: conv2d(c3, nc, 1, conv2d_cfg, cvb.pp("2"))?,
            };
            cls_branches.push(cls_branch);
        }

        Ok(Self {
            box_branches,
            cls_branches,
            strides,
            nc,
            max_det: 300,
        })
    }

    pub fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        let device: &Device = features[0].device();
        let batch_size: usize = features[0].dim(0)?;

        let mut all_boxes: Vec<Tensor> = Vec::new();
        let mut all_scores: Vec<Tensor> = Vec::new();
        let mut feat_sizes: Vec<(usize, usize)> = Vec::new();

        for (i, feat) in features.iter().enumerate() {
            let (_, _, h, w) = feat.dims4()?;
            feat_sizes.push((h, w));

            // Box branch: Conv→Conv→Conv2d → [B, 4, H*W]
            let bx = self.box_branches[i].cv0.forward(feat)?;
            let bx = self.box_branches[i].cv1.forward(&bx)?;
            let bx = self.box_branches[i].cv2.forward(&bx)?; // [B, 4, H, W]
            let bx = bx.reshape((batch_size, 4, h * w))?;
            all_boxes.push(bx);

            // Cls branch: DWConv+Conv → DWConv+Conv → Conv2d → [B, nc, H*W]
            let cx = self.cls_branches[i].dw0.forward(feat)?;
            let cx = self.cls_branches[i].cv0.forward(&cx)?;
            let cx = self.cls_branches[i].dw1.forward(&cx)?;
            let cx = self.cls_branches[i].cv1.forward(&cx)?;
            let cx = self.cls_branches[i].cv2.forward(&cx)?; // [B, nc, H, W]
            let cx = cx.reshape((batch_size, self.nc, h * w))?;
            all_scores.push(cx);
        }

        // Concat across scales: boxes [B, 4, N], scores [B, nc, N] where N=sum(H_i*W_i)
        let boxes = Tensor::cat(&all_boxes.iter().collect::<Vec<_>>(), 2)?;
        let scores = Tensor::cat(&all_scores.iter().collect::<Vec<_>>(), 2)?;

        // Generate anchors and strides
        let (anchors, stride_tensor) = make_anchors(&feat_sizes, &self.strides, device)?;

        // dist2bbox xyxy: dbox = (anchor ± distance) * stride
        let dbox = dist2bbox_xyxy(&boxes, &anchors)?; // [B, 4, N]
        let dbox = dbox.broadcast_mul(&stride_tensor)?; // [B, 4, N]

        // Sigmoid on cls scores
        let cls_scores = candle_nn::ops::sigmoid(&scores)?; // [B, nc, N]

        // Concat and permute: [B, 4+nc, N] → [B, N, 4+nc]
        let y = Tensor::cat(&[&dbox, &cls_scores], 1)?; // [B, 84, N]
        let y = y.transpose(1, 2)?; // [B, N, 84]

        // Top-k postprocess → [B, 300, 6]
        self.topk_postprocess(&y, batch_size)
    }

    fn topk_postprocess(&self, preds: &Tensor, batch_size: usize) -> Result<Tensor> {
        // preds: [B, N, 84] where N=8400
        let n_anchors: usize = preds.dim(1)?;
        let k: usize = self.max_det.min(n_anchors);

        // Split into boxes [B, N, 4] and scores [B, N, nc]
        let boxes = preds.narrow(2, 0, 4)?;
        let scores = preds.narrow(2, 4, self.nc)?;

        // For batch_size=1 (WASM inference), simplified topk:
        // 1. Max score per anchor → [B, N]
        // 2. Sort descending, take top-k
        // 3. Gather boxes and best class info
        assert_eq!(
            batch_size, 1,
            "Only batch_size=1 supported for WASM inference"
        );

        let scores_2d = scores.squeeze(0)?; // [N, nc]
        let boxes_2d = boxes.squeeze(0)?; // [N, 4]

        // Max class score and class id per anchor
        let max_scores = scores_2d.max(1)?; // [N]
        let max_indices = scores_2d.argmax(1)?; // [N]

        let max_scores_vec: Vec<f32> = max_scores.to_vec1()?;
        let max_indices_vec: Vec<u32> = max_indices.to_vec1()?;
        let boxes_data: Vec<f32> = boxes_2d.flatten_all()?.to_vec1()?;

        // Sort by score descending, take top-k
        let mut scored: Vec<(usize, f32)> = max_scores_vec
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build output: [1, k, 6] = [x1, y1, x2, y2, score, class_id]
        let mut output: Vec<f32> = Vec::with_capacity(k * 6);
        for &(idx, score) in scored.iter().take(k) {
            let base: usize = idx * 4;
            output.push(boxes_data[base]); // x1
            output.push(boxes_data[base + 1]); // y1
            output.push(boxes_data[base + 2]); // x2
            output.push(boxes_data[base + 3]); // y2
            output.push(score);
            output.push(max_indices_vec[idx] as f32);
        }

        Tensor::from_vec(output, (1, k, 6), preds.device())
    }
}

/// Generate anchor points and stride tensor for all feature map scales.
/// Returns (anchors [1, 2, N], strides [1, 1, N]) for broadcasting with [B, 4, N].
fn make_anchors(
    feat_sizes: &[(usize, usize)],
    strides: &[f32],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut anchor_data: Vec<f32> = Vec::new();
    let mut stride_data: Vec<f32> = Vec::new();

    for (i, &(h, w)) in feat_sizes.iter().enumerate() {
        let stride: f32 = strides[i];
        for y in 0..h {
            for x in 0..w {
                anchor_data.push(x as f32 + 0.5); // sx
                anchor_data.push(y as f32 + 0.5); // sy
                stride_data.push(stride);
            }
        }
    }

    let n: usize = stride_data.len();
    // anchors: [N, 2] → transpose to [2, N] → unsqueeze to [1, 2, N]
    let anchors = Tensor::from_vec(anchor_data, (n, 2), device)?;
    let anchors = anchors.transpose(0, 1)?.unsqueeze(0)?; // [1, 2, N]

    // strides: [N] → reshape to [1, 1, N]
    let stride_tensor = Tensor::from_vec(stride_data, (1, 1, n), device)?;

    Ok((anchors, stride_tensor))
}

/// dist2bbox in xyxy mode (for end2end): x1y1 = anchor - lt, x2y2 = anchor + rb
fn dist2bbox_xyxy(distance: &Tensor, anchors: &Tensor) -> Result<Tensor> {
    // distance: [B, 4, N], anchors: [1, 2, N]
    let lt = distance.narrow(1, 0, 2)?; // [B, 2, N]
    let rb = distance.narrow(1, 2, 2)?; // [B, 2, N]
    let x1y1 = anchors.broadcast_sub(&lt)?; // [B, 2, N]
    let x2y2 = anchors.broadcast_add(&rb)?; // [B, 2, N]
    Tensor::cat(&[&x1y1, &x2y2], 1) // [B, 4, N]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_make_anchors() {
        let device = Device::Cpu;
        let feat_sizes = vec![(80, 80), (40, 40), (20, 20)];
        let strides = vec![8.0, 16.0, 32.0];
        let (anchors, stride_t) = make_anchors(&feat_sizes, &strides, &device).unwrap();

        let total: usize = 80 * 80 + 40 * 40 + 20 * 20; // 8400
        assert_eq!(anchors.dims(), &[1, 2, total]);
        assert_eq!(stride_t.dims(), &[1, 1, total]);
    }

    #[test]
    fn test_dist2bbox_xyxy() {
        let device = Device::Cpu;
        // Single anchor at (5.5, 5.5), distance [1, 2, 3, 4]
        let anchors = Tensor::new(&[5.5f32, 5.5], &device)
            .unwrap()
            .reshape((1, 2, 1))
            .unwrap();
        let distance = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
            .unwrap()
            .reshape((1, 4, 1))
            .unwrap();
        let result = dist2bbox_xyxy(&distance, &anchors).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // x1 = 5.5-1 = 4.5, y1 = 5.5-2 = 3.5, x2 = 5.5+3 = 8.5, y2 = 5.5+4 = 9.5
        assert!((data[0] - 4.5).abs() < 1e-5);
        assert!((data[1] - 3.5).abs() < 1e-5);
        assert!((data[2] - 8.5).abs() < 1e-5);
        assert!((data[3] - 9.5).abs() < 1e-5);
    }

    #[test]
    fn test_detect_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = Detect::load(vb.pp("model").pp("23"), &[64, 128, 256], 80).unwrap();

        let f0 = Tensor::zeros((1, 64, 80, 80), DType::F32, &device).unwrap();
        let f1 = Tensor::zeros((1, 128, 40, 40), DType::F32, &device).unwrap();
        let f2 = Tensor::zeros((1, 256, 20, 20), DType::F32, &device).unwrap();
        let out = head.forward(&[&f0, &f1, &f2]).unwrap();
        assert_eq!(out.dims(), &[1, 300, 6]);
    }
}
