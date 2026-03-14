pub mod backbone;
pub mod blocks;
pub mod config;
pub mod head;
pub mod neck;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use backbone::Backbone;
use config::ModelScale;
use head::Detect;
use neck::Neck;

const NC: usize = 80;

/// Top-level YOLO26 model: Backbone + Neck + Detect head.
/// Supports all scales (n/s/m/l/x) via ModelScale parameterization.
pub struct Yolo26Model {
    backbone: Backbone,
    neck: Neck,
    head: Detect,
}

impl Yolo26Model {
    /// Load model from SafeTensors bytes with the given scale.
    /// Weight keys start with "model." prefix (ultralytics convention).
    pub fn load(weights_bytes: Vec<u8>, device: &Device, scale: ModelScale) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, device)?;
        let vb = vb.pp("model");
        let backbone: Backbone = Backbone::load(vb.clone(), scale)?;
        let neck: Neck = Neck::load(vb.clone(), scale)?;
        let head_channels: [usize; 3] = scale.head_input_channels();
        let head: Detect = Detect::load(vb.pp("23"), &head_channels, NC)?;
        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    /// Construct from pre-built parts (for testing with VarMap).
    #[cfg(test)]
    pub fn new_from_parts(backbone: Backbone, neck: Neck, head: Detect) -> Self {
        Self {
            backbone,
            neck,
            head,
        }
    }

    /// Run forward pass: input [1, 3, 640, 640] → output [1, 300, 6]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let bb = self.backbone.forward(input)?;
        let neck_out = self.neck.forward(&bb)?;
        self.head
            .forward(&[&neck_out.small, &neck_out.medium, &neck_out.large])
    }

    /// Return pre-topk decoded output [B, N, 84] for comparison testing.
    pub fn forward_pre_topk(&self, input: &Tensor) -> Result<Tensor> {
        let bb = self.backbone.forward(input)?;
        let neck_out = self.neck.forward(&bb)?;
        self.head
            .forward_pre_topk(&[&neck_out.small, &neck_out.medium, &neck_out.large])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_model_output_shape_n() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let scale = ModelScale::N;
        let backbone = Backbone::load(vb.pp("model"), scale).unwrap();
        let neck = Neck::load(vb.pp("model"), scale).unwrap();
        let head = Detect::load(vb.pp("model").pp("23"), &scale.head_input_channels(), NC).unwrap();
        let model = Yolo26Model::new_from_parts(backbone, neck, head);
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = model.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 300, 6]);
    }

    #[test]
    fn test_model_output_shape_m() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let scale = ModelScale::M;
        let backbone = Backbone::load(vb.pp("model"), scale).unwrap();
        let neck = Neck::load(vb.pp("model"), scale).unwrap();
        let head = Detect::load(vb.pp("model").pp("23"), &scale.head_input_channels(), NC).unwrap();
        let model = Yolo26Model::new_from_parts(backbone, neck, head);
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = model.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 300, 6]);
    }
}
