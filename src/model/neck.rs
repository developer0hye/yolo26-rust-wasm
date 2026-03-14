use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::backbone::BackboneOutput;
use super::blocks::{C3k2, ConvBlock};

pub struct NeckOutput {
    pub small: Tensor,  // layer 16 [64, 80, 80] — P3/8
    pub medium: Tensor, // layer 19 [128, 40, 40] — P4/16
    pub large: Tensor,  // layer 22 [256, 20, 20] — P5/32
}

/// YOLO26n FPN-PAN neck: layers 11-22
/// Takes backbone p3/p4/p5 skip connections, outputs 3-scale features for Detect head.
pub struct Neck {
    // FPN (top-down)
    l13: C3k2, // 384→128, c3k=T, n=1
    l16: C3k2, // 256→64, c3k=T, n=1
    // PAN (bottom-up)
    l17: ConvBlock, // 64→64, k=3, s=2
    l19: C3k2,      // 192→128, c3k=T, n=1
    l20: ConvBlock, // 128→128, k=3, s=2
    l22: C3k2,      // 384→256, c3k=T, e=0.5, n=1, attn=T
}

impl Neck {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        // Layers 11, 12, 14, 15 are Upsample/Concat — no weights
        let l13 = C3k2::load(vb.pp("13"), 384, 128, 1, true, 0.5, true, false)?;
        let l16 = C3k2::load(vb.pp("16"), 256, 64, 1, true, 0.5, true, false)?;
        let l17 = ConvBlock::load(vb.pp("17"), 64, 64, 3, 2, 1, true)?;
        let l19 = C3k2::load(vb.pp("19"), 192, 128, 1, true, 0.5, true, false)?;
        let l20 = ConvBlock::load(vb.pp("20"), 128, 128, 3, 2, 1, true)?;
        let l22 = C3k2::load(vb.pp("22"), 384, 256, 1, true, 0.5, true, true)?;
        Ok(Self {
            l13,
            l16,
            l17,
            l19,
            l20,
            l22,
        })
    }

    pub fn forward(&self, bb: &BackboneOutput) -> Result<NeckOutput> {
        // FPN: top-down path
        let (_, _, h4, w4) = bb.p4.dims4()?;
        let l11 = bb.p5.upsample_nearest2d(h4, w4)?; // [256, 40, 40]
        let l12 = Tensor::cat(&[&l11, &bb.p4], 1)?; // [384, 40, 40]
        let l13 = self.l13.forward(&l12)?; // [128, 40, 40]

        let (_, _, h3, w3) = bb.p3.dims4()?;
        let l14 = l13.upsample_nearest2d(h3, w3)?; // [128, 80, 80]
        let l15 = Tensor::cat(&[&l14, &bb.p3], 1)?; // [256, 80, 80]
        let l16 = self.l16.forward(&l15)?; // [64, 80, 80] → small

        // PAN: bottom-up path
        let l17 = self.l17.forward(&l16)?; // [64, 40, 40]
        let l18 = Tensor::cat(&[&l17, &l13], 1)?; // [192, 40, 40]
        let l19 = self.l19.forward(&l18)?; // [128, 40, 40] → medium

        let l20 = self.l20.forward(&l19)?; // [128, 20, 20]
        let l21 = Tensor::cat(&[&l20, &bb.p5], 1)?; // [384, 20, 20]
        let l22 = self.l22.forward(&l21)?; // [256, 20, 20] → large

        Ok(NeckOutput {
            small: l16,
            medium: l19,
            large: l22,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::backbone::Backbone;
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_neck_output_shapes() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model")).unwrap();
        let neck = Neck::load(vb.pp("model")).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let bb = backbone.forward(&x).unwrap();
        let out = neck.forward(&bb).unwrap();
        assert_eq!(out.small.dims(), &[1, 64, 80, 80]);
        assert_eq!(out.medium.dims(), &[1, 128, 40, 40]);
        assert_eq!(out.large.dims(), &[1, 256, 20, 20]);
    }
}
