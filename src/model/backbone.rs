use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::blocks::{C2psa, C3k2, ConvBlock, Sppf};

pub struct BackboneOutput {
    pub p3: Tensor, // layer 4 output [128, 80, 80]
    pub p4: Tensor, // layer 6 output [128, 40, 40]
    pub p5: Tensor, // layer 10 output [256, 20, 20]
}

/// YOLO26n backbone: layers 0-10
/// Input: [1, 3, 640, 640] → Outputs: p3/p4/p5 for neck skip connections
pub struct Backbone {
    l0: ConvBlock, // 3→16, k=3, s=2
    l1: ConvBlock, // 16→32, k=3, s=2
    l2: C3k2,      // 32→64, c3k=F, e=0.25, n=1
    l3: ConvBlock, // 64→64, k=3, s=2
    l4: C3k2,      // 64→128, c3k=F, e=0.25, n=1
    l5: ConvBlock, // 128→128, k=3, s=2
    l6: C3k2,      // 128→128, c3k=T, e=0.5, n=1
    l7: ConvBlock, // 128→256, k=3, s=2
    l8: C3k2,      // 256→256, c3k=T, e=0.5, n=1
    l9: Sppf,      // 256→256, k=5, n=3, shortcut=T
    l10: C2psa,    // 256→256, n=1
}

impl Backbone {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        // vb prefix is "model" — each layer uses vb.pp("{layer_idx}")
        let l0 = ConvBlock::load(vb.pp("0"), 3, 16, 3, 2, 1, true)?;
        let l1 = ConvBlock::load(vb.pp("1"), 16, 32, 3, 2, 1, true)?;
        let l2 = C3k2::load(vb.pp("2"), 32, 64, 1, false, 0.25, true, false)?;
        let l3 = ConvBlock::load(vb.pp("3"), 64, 64, 3, 2, 1, true)?;
        let l4 = C3k2::load(vb.pp("4"), 64, 128, 1, false, 0.25, true, false)?;
        let l5 = ConvBlock::load(vb.pp("5"), 128, 128, 3, 2, 1, true)?;
        let l6 = C3k2::load(vb.pp("6"), 128, 128, 1, true, 0.5, true, false)?;
        let l7 = ConvBlock::load(vb.pp("7"), 128, 256, 3, 2, 1, true)?;
        let l8 = C3k2::load(vb.pp("8"), 256, 256, 1, true, 0.5, true, false)?;
        let l9 = Sppf::load(vb.pp("9"), 256, 256, 5, 3, true)?;
        let l10 = C2psa::load(vb.pp("10"), 256, 256, 1)?;
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
        let x = self.l0.forward(x)?; // [16, 320, 320]
        let x = self.l1.forward(&x)?; // [32, 160, 160]
        let x = self.l2.forward(&x)?; // [64, 160, 160]
        let x = self.l3.forward(&x)?; // [64, 80, 80]
        let p3 = self.l4.forward(&x)?; // [128, 80, 80] ← skip
        let x = self.l5.forward(&p3)?; // [128, 40, 40]
        let p4 = self.l6.forward(&x)?; // [128, 40, 40] ← skip
        let x = self.l7.forward(&p4)?; // [256, 20, 20]
        let x = self.l8.forward(&x)?; // [256, 20, 20]
        let x = self.l9.forward(&x)?; // [256, 20, 20]
        let p5 = self.l10.forward(&x)?; // [256, 20, 20] ← skip
        Ok(BackboneOutput { p3, p4, p5 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_backbone_output_shapes() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model")).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let out = backbone.forward(&x).unwrap();
        assert_eq!(out.p3.dims(), &[1, 128, 80, 80]);
        assert_eq!(out.p4.dims(), &[1, 128, 40, 40]);
        assert_eq!(out.p5.dims(), &[1, 256, 20, 20]);
    }
}
