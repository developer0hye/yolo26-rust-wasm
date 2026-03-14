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

/// Model size scaling factors derived from ultralytics yolo26.yaml.
/// Controls channel widths (width), layer repetitions (depth), and max channel cap (max_ch).
#[derive(Debug, Clone, Copy)]
pub struct Multiples {
    pub depth: f64,
    pub width: f64,
    pub max_ch: usize,
}

impl Multiples {
    pub fn n() -> Self {
        Self {
            depth: 0.50,
            width: 0.25,
            max_ch: 1024,
        }
    }
    pub fn s() -> Self {
        Self {
            depth: 0.50,
            width: 0.50,
            max_ch: 1024,
        }
    }
    pub fn m() -> Self {
        Self {
            depth: 0.50,
            width: 1.00,
            max_ch: 512,
        }
    }
    pub fn l() -> Self {
        Self {
            depth: 1.00,
            width: 1.00,
            max_ch: 512,
        }
    }
    pub fn x() -> Self {
        Self {
            depth: 1.00,
            width: 1.50,
            max_ch: 512,
        }
    }

    /// Look up Multiples by single-letter size name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "n" => Some(Self::n()),
            "s" => Some(Self::s()),
            "m" => Some(Self::m()),
            "l" => Some(Self::l()),
            "x" => Some(Self::x()),
            _ => None,
        }
    }
}

/// Compute scaled channel count: floor(base * width), capped at max_ch.
pub fn make_ch(base: usize, m: &Multiples) -> usize {
    ((base as f64 * m.width) as usize).min(m.max_ch)
}

/// Compute scaled depth (layer repeat count): round(base * depth), minimum 1.
pub fn make_depth(base: usize, m: &Multiples) -> usize {
    (base as f64 * m.depth).round().max(1.0) as usize
}

/// Top-level YOLO26 model: Backbone + Neck + Detect head.
pub struct Yolo26Model {
    backbone: Backbone,
    neck: Neck,
    head: Detect,
}

impl Yolo26Model {
    /// Load model from SafeTensors bytes with given size multiples.
    /// Weight keys start with "model." prefix (ultralytics convention).
    pub fn load(weights_bytes: Vec<u8>, device: &Device, multiples: &Multiples) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, device)?;
        let vb = vb.pp("model");
        let backbone: Backbone = Backbone::load(vb.clone(), multiples)?;
        let neck: Neck = Neck::load(vb.clone(), multiples)?;
        let head_input_channels = [
            make_ch(256, multiples),
            make_ch(512, multiples),
            make_ch(1024, multiples),
        ];
        let head: Detect = Detect::load(vb.pp("23"), &head_input_channels, NC)?;
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

    fn build_model_with_multiples(m: &Multiples) -> Yolo26Model {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), m).unwrap();
        let neck = Neck::load(vb.pp("model"), m).unwrap();
        let head_ch = [make_ch(256, m), make_ch(512, m), make_ch(1024, m)];
        let head = Detect::load(vb.pp("model").pp("23"), &head_ch, NC).unwrap();
        Yolo26Model::new_from_parts(backbone, neck, head)
    }

    #[test]
    fn test_model_output_shape_n() {
        let model = build_model_with_multiples(&Multiples::n());
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = model.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 300, 6]);
    }

    #[test]
    fn test_model_output_shape_s() {
        let model = build_model_with_multiples(&Multiples::s());
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = model.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 300, 6]);
    }

    #[test]
    fn test_make_ch() {
        let n = Multiples::n();
        assert_eq!(make_ch(64, &n), 16);
        assert_eq!(make_ch(128, &n), 32);
        assert_eq!(make_ch(256, &n), 64);
        assert_eq!(make_ch(512, &n), 128);
        assert_eq!(make_ch(1024, &n), 256);

        let m = Multiples::m();
        assert_eq!(make_ch(64, &m), 64);
        assert_eq!(make_ch(128, &m), 128);
        assert_eq!(make_ch(256, &m), 256);
        assert_eq!(make_ch(512, &m), 512);
        assert_eq!(make_ch(1024, &m), 512); // capped at max_ch=512

        let x = Multiples::x();
        assert_eq!(make_ch(64, &x), 96);
        assert_eq!(make_ch(128, &x), 192);
        assert_eq!(make_ch(256, &x), 384);
        assert_eq!(make_ch(512, &x), 512); // capped
        assert_eq!(make_ch(1024, &x), 512); // capped
    }

    #[test]
    fn test_make_depth() {
        let n = Multiples::n();
        assert_eq!(make_depth(2, &n), 1); // round(2*0.5)=1

        let l = Multiples::l();
        assert_eq!(make_depth(2, &l), 2); // round(2*1.0)=2
    }

    #[test]
    fn test_multiples_from_name() {
        assert!(Multiples::from_name("n").is_some());
        assert!(Multiples::from_name("s").is_some());
        assert!(Multiples::from_name("m").is_some());
        assert!(Multiples::from_name("l").is_some());
        assert!(Multiples::from_name("x").is_some());
        assert!(Multiples::from_name("z").is_none());
    }
}
