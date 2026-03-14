use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::backbone::BackboneOutput;
use super::blocks::{C3k2, ConvBlock};
use super::config::ModelScale;

pub struct NeckOutput {
    pub small: Tensor,  // layer 16 — P3/8
    pub medium: Tensor, // layer 19 — P4/16
    pub large: Tensor,  // layer 22 — P5/32
}

/// YOLO26 FPN-PAN neck: layers 11-22.
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
    pub fn load(vb: VarBuilder, scale: ModelScale) -> Result<Self> {
        // Scaled YAML output channels
        let c_256: usize = scale.channel(256);
        let c_512: usize = scale.channel(512);
        let c_1024: usize = scale.channel(1024);

        // Backbone output channels for concat computations
        // p3 = layer 4 output = ch(512), p4 = layer 6 output = ch(512), p5 = layer 10 output = ch(1024)
        let p3_ch: usize = c_512;
        let p4_ch: usize = c_512;
        let p5_ch: usize = c_1024;

        // Concat input channels
        let l12_in: usize = p5_ch + p4_ch; // upsample(p5) + p4
        let l15_in: usize = c_512 + p3_ch; // upsample(l13_out=ch(512)) + p3
        let l18_in: usize = c_256 + c_512; // l17_out=ch(256) + l13_out=ch(512)
        let l21_in: usize = c_512 + p5_ch; // l20_out=ch(512) + p5

        // YAML repeats=2 for layers 13, 16, 19
        let n: usize = scale.repeat(2);

        // Layers 11, 12, 14, 15 are Upsample/Concat — no weights
        let l13 = C3k2::load(vb.pp("13"), l12_in, c_512, n, true, 0.5, true, false)?;
        let l16 = C3k2::load(vb.pp("16"), l15_in, c_256, n, true, 0.5, true, false)?;
        let l17 = ConvBlock::load(vb.pp("17"), c_256, c_256, 3, 2, 1, true)?;
        let l19 = C3k2::load(vb.pp("19"), l18_in, c_512, n, true, 0.5, true, false)?;
        let l20 = ConvBlock::load(vb.pp("20"), c_512, c_512, 3, 2, 1, true)?;
        // Layer 22: YAML repeats=1, always c3k=True, attn=True
        let l22 = C3k2::load(vb.pp("22"), l21_in, c_1024, 1, true, 0.5, true, true)?;

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
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), ModelScale::N).unwrap();
        let neck = Neck::load(vb.pp("model"), ModelScale::N).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let bb = backbone.forward(&x).unwrap();
        let out = neck.forward(&bb).unwrap();
        assert_eq!(out.small.dims(), &[1, 64, 80, 80]);
        assert_eq!(out.medium.dims(), &[1, 128, 40, 40]);
        assert_eq!(out.large.dims(), &[1, 256, 20, 20]);
    }

    #[test]
    fn test_neck_output_shapes_m() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let backbone = Backbone::load(vb.pp("model"), ModelScale::M).unwrap();
        let neck = Neck::load(vb.pp("model"), ModelScale::M).unwrap();
        let x = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let bb = backbone.forward(&x).unwrap();
        let out = neck.forward(&bb).unwrap();
        assert_eq!(out.small.dims(), &[1, 256, 80, 80]);
        assert_eq!(out.medium.dims(), &[1, 512, 40, 40]);
        assert_eq!(out.large.dims(), &[1, 512, 20, 20]);
    }
}
