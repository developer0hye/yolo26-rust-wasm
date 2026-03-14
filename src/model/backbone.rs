use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::blocks::{C2psa, C3k2, ConvBlock, Sppf};
use super::{make_ch, make_depth, Multiples};

pub struct BackboneOutput {
    pub p3: Tensor, // layer 4 output [c4, 80, 80]
    pub p4: Tensor, // layer 6 output [c4, 40, 40]
    pub p5: Tensor, // layer 10 output [c5, 20, 20]
}

/// YOLO26 backbone: layers 0-10
/// Input: [1, 3, 640, 640] → Outputs: p3/p4/p5 for neck skip connections
pub struct Backbone {
    l0: ConvBlock,
    l1: ConvBlock,
    l2: C3k2,
    l3: ConvBlock,
    l4: C3k2,
    l5: ConvBlock,
    l6: C3k2,
    l7: ConvBlock,
    l8: C3k2,
    l9: Sppf,
    l10: C2psa,
}

impl Backbone {
    pub fn load(vb: VarBuilder, m: &Multiples) -> Result<Self> {
        let c1 = make_ch(64, m);
        let c2 = make_ch(128, m);
        let c3 = make_ch(256, m);
        let c4 = make_ch(512, m);
        let c5 = make_ch(1024, m);
        let d = make_depth(2, m); // base depth=2 from yolo26.yaml

        let l0 = ConvBlock::load(vb.pp("0"), 3, c1, 3, 2, 1, true)?;
        let l1 = ConvBlock::load(vb.pp("1"), c1, c2, 3, 2, 1, true)?;
        let l2 = C3k2::load(vb.pp("2"), c2, c3, d, false, 0.25, true, false)?;
        let l3 = ConvBlock::load(vb.pp("3"), c3, c3, 3, 2, 1, true)?;
        let l4 = C3k2::load(vb.pp("4"), c3, c4, d, false, 0.25, true, false)?;
        let l5 = ConvBlock::load(vb.pp("5"), c4, c4, 3, 2, 1, true)?;
        let l6 = C3k2::load(vb.pp("6"), c4, c4, d, true, 0.5, true, false)?;
        let l7 = ConvBlock::load(vb.pp("7"), c4, c5, 3, 2, 1, true)?;
        let l8 = C3k2::load(vb.pp("8"), c5, c5, d, true, 0.5, true, false)?;
        let l9 = Sppf::load(vb.pp("9"), c5, c5, 5, 3, true)?;
        let l10 = C2psa::load(vb.pp("10"), c5, c5, d)?;
        Ok(Self {
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
            l10,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<BackboneOutput> {
        let x = self.l0.forward(x)?;
        let x = self.l1.forward(&x)?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;
        let p3 = self.l4.forward(&x)?;
        let x = self.l5.forward(&p3)?;
        let p4 = self.l6.forward(&x)?;
        let x = self.l7.forward(&p4)?;
        let x = self.l8.forward(&x)?;
        let x = self.l9.forward(&x)?;
        let p5 = self.l10.forward(&x)?;
        Ok(BackboneOutput { p3, p4, p5 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_backbone_output_shapes_n() {
        let m = Multiples::n();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), &m).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = backbone.forward(&x).unwrap();
        // n: c4=128, c4=128, c5=256
        assert_eq!(out.p3.dims(), &[1, 128, 80, 80]);
        assert_eq!(out.p4.dims(), &[1, 128, 40, 40]);
        assert_eq!(out.p5.dims(), &[1, 256, 20, 20]);
    }

    #[test]
    fn test_backbone_output_shapes_s() {
        let m = Multiples::s();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), &m).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = backbone.forward(&x).unwrap();
        // s: c4=256, c4=256, c5=512
        assert_eq!(out.p3.dims(), &[1, 256, 80, 80]);
        assert_eq!(out.p4.dims(), &[1, 256, 40, 40]);
        assert_eq!(out.p5.dims(), &[1, 512, 20, 20]);
    }
}
