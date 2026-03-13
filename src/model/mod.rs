pub mod backbone;
pub mod blocks;
pub mod head;
pub mod neck;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use backbone::Backbone;
use head::Detect;
use neck::Neck;

const NC: usize = 80;
const HEAD_INPUT_CHANNELS: [usize; 3] = [64, 128, 256];

/// Top-level YOLO26n model: Backbone + Neck + Detect head.
pub struct Yolo26Model {
    backbone: Backbone,
    neck: Neck,
    head: Detect,
}

impl Yolo26Model {
    /// Load model from SafeTensors bytes.
    /// Weight keys start with "model." prefix (ultralytics convention).
    pub fn load(weights_bytes: Vec<u8>, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, device)?;
        let vb = vb.pp("model");
        let backbone: Backbone = Backbone::load(vb.clone())?;
        let neck: Neck = Neck::load(vb.clone())?;
        let head: Detect = Detect::load(vb.pp("23"), &HEAD_INPUT_CHANNELS, NC)?;
        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    /// Run forward pass: input [1, 3, 640, 640] → output [1, 300, 6]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let bb = self.backbone.forward(input)?;
        let neck_out = self.neck.forward(&bb)?;
        self.head
            .forward(&[&neck_out.small, &neck_out.medium, &neck_out.large])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_model_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model")).unwrap();
        let neck = Neck::load(vb.pp("model")).unwrap();
        let head = Detect::load(vb.pp("model").pp("23"), &HEAD_INPUT_CHANNELS, NC).unwrap();
        let model = Yolo26Model {
            backbone,
            neck,
            head,
        };
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = model.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 300, 6]);
    }
}
