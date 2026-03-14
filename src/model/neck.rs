use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::backbone::BackboneOutput;
use super::blocks::{C3k2, ConvBlock};
use super::{make_ch, make_depth, Multiples};

pub struct NeckOutput {
    pub small: Tensor,  // layer 16 [c3, 80, 80] — P3/8
    pub medium: Tensor, // layer 19 [c4, 40, 40] — P4/16
    pub large: Tensor,  // layer 22 [c5, 20, 20] — P5/32
}

/// YOLO26 FPN-PAN neck: layers 11-22
/// Takes backbone p3/p4/p5 skip connections, outputs 3-scale features for Detect head.
pub struct Neck {
    // FPN (top-down)
    l13: C3k2,
    l16: C3k2,
    // PAN (bottom-up)
    l17: ConvBlock,
    l19: C3k2,
    l20: ConvBlock,
    l22: C3k2,
}

impl Neck {
    pub fn load(vb: VarBuilder, m: &Multiples) -> Result<Self> {
        let c3 = make_ch(256, m);
        let c4 = make_ch(512, m);
        let c5 = make_ch(1024, m);
        let d = make_depth(2, m);

        // Concat input channels: p5+p4, l13_out+p3, l16_out+l13, l19_out+p5
        let l13_in = c5 + c4; // upsample(p5) ++ p4
        let l16_in = c4 + c4; // upsample(l13_out=c4) ++ p3(=c4)
        let l19_in = c3 + c4; // downsample(l16_out=c3) ++ l13(=c4)
        let l22_in = c4 + c5; // downsample(l19_out=c4) ++ p5(=c5)

        let l13 = C3k2::load(vb.pp("13"), l13_in, c4, d, true, 0.5, true, false)?;
        let l16 = C3k2::load(vb.pp("16"), l16_in, c3, d, true, 0.5, true, false)?;
        let l17 = ConvBlock::load(vb.pp("17"), c3, c3, 3, 2, 1, true)?;
        let l19 = C3k2::load(vb.pp("19"), l19_in, c4, d, true, 0.5, true, false)?;
        let l20 = ConvBlock::load(vb.pp("20"), c4, c4, 3, 2, 1, true)?;
        let l22 = C3k2::load(vb.pp("22"), l22_in, c5, d, true, 0.5, true, true)?;
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
        let l11 = bb.p5.upsample_nearest2d(h4, w4)?;
        let l12 = Tensor::cat(&[&l11, &bb.p4], 1)?;
        let l13 = self.l13.forward(&l12)?;

        let (_, _, h3, w3) = bb.p3.dims4()?;
        let l14 = l13.upsample_nearest2d(h3, w3)?;
        let l15 = Tensor::cat(&[&l14, &bb.p3], 1)?;
        let l16 = self.l16.forward(&l15)?;

        // PAN: bottom-up path
        let l17 = self.l17.forward(&l16)?;
        let l18 = Tensor::cat(&[&l17, &l13], 1)?;
        let l19 = self.l19.forward(&l18)?;

        let l20 = self.l20.forward(&l19)?;
        let l21 = Tensor::cat(&[&l20, &bb.p5], 1)?;
        let l22 = self.l22.forward(&l21)?;

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
    fn test_neck_output_shapes_n() {
        let m = Multiples::n();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), &m).unwrap();
        let neck = Neck::load(vb.pp("model"), &m).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let bb = backbone.forward(&x).unwrap();
        let out = neck.forward(&bb).unwrap();
        // n: c3=64, c4=128, c5=256
        assert_eq!(out.small.dims(), &[1, 64, 80, 80]);
        assert_eq!(out.medium.dims(), &[1, 128, 40, 40]);
        assert_eq!(out.large.dims(), &[1, 256, 20, 20]);
    }

    #[test]
    fn test_neck_output_shapes_s() {
        let m = Multiples::s();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), &m).unwrap();
        let neck = Neck::load(vb.pp("model"), &m).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let bb = backbone.forward(&x).unwrap();
        let out = neck.forward(&bb).unwrap();
        // s: c3=128, c4=256, c5=512
        assert_eq!(out.small.dims(), &[1, 128, 80, 80]);
        assert_eq!(out.medium.dims(), &[1, 256, 40, 40]);
        assert_eq!(out.large.dims(), &[1, 512, 20, 20]);
    }
}
