use candle_core::{Module, Result, Tensor};
use candle_nn::{batch_norm, conv2d_no_bias, BatchNormConfig, Conv2d, Conv2dConfig, VarBuilder};

// ---------------------------------------------------------------------------
// ConvBlock: Conv2d + BatchNorm (fused via absorb_bn) + optional SiLU
// ---------------------------------------------------------------------------

pub struct ConvBlock {
    conv: Conv2d,
    is_activated: bool,
}

impl ConvBlock {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        is_activated: bool,
    ) -> Result<Self> {
        let padding: usize = kernel_size / 2;
        let cfg = Conv2dConfig {
            padding,
            stride,
            groups,
            ..Default::default()
        };
        let conv: Conv2d = conv2d_no_bias(c_in, c_out, kernel_size, cfg, vb.pp("conv"))?;
        let bn = batch_norm(c_out, BatchNormConfig::default(), vb.pp("bn"))?;
        let conv: Conv2d = conv.absorb_bn(&bn)?;
        Ok(Self { conv, is_activated })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.conv.forward(x)?;
        if self.is_activated {
            candle_nn::ops::silu(&x)
        } else {
            Ok(x)
        }
    }
}

// ---------------------------------------------------------------------------
// Bottleneck: cv1(c1→c2, k[0]) + cv2(c2→c2, k[1]) with optional residual
// ---------------------------------------------------------------------------

pub struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    has_shortcut: bool,
}

impl Bottleneck {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        has_shortcut: bool,
        k: (usize, usize),
    ) -> Result<Self> {
        let cv1: ConvBlock = ConvBlock::load(vb.pp("cv1"), c_in, c_out, k.0, 1, 1, true)?;
        let cv2: ConvBlock = ConvBlock::load(vb.pp("cv2"), c_out, c_out, k.1, 1, 1, true)?;
        let has_shortcut: bool = has_shortcut && c_in == c_out;
        Ok(Self {
            cv1,
            cv2,
            has_shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y: Tensor = self.cv2.forward(&self.cv1.forward(x)?)?;
        if self.has_shortcut {
            x + y
        } else {
            Ok(y)
        }
    }
}

// ---------------------------------------------------------------------------
// C3k: C3 variant with k=3 bottleneck kernels
// cv1(c1→c_, 1×1), cv2(c1→c_, 1×1), m = Sequential(Bottleneck(c_,c_,k=3)), cv3(2*c_→c2, 1×1)
// Optionally, m can be replaced by a PsaBlock (when parent C3k2 has attn=True)
// ---------------------------------------------------------------------------

pub enum C3kInternal {
    Bottlenecks(Vec<Bottleneck>),
    Psa(PsaBlock),
}

pub struct C3k {
    cv1: ConvBlock,
    cv2: ConvBlock,
    cv3: ConvBlock,
    m: C3kInternal,
}

impl C3k {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        n: usize,
        has_shortcut: bool,
        use_psa: bool,
    ) -> Result<Self> {
        // C3's hidden channels: c_ = c_out * e, where e=1.0 for C3k
        let c_hidden: usize = c_out;
        let cv1: ConvBlock = ConvBlock::load(vb.pp("cv1"), c_in, c_hidden, 1, 1, 1, true)?;
        let cv2: ConvBlock = ConvBlock::load(vb.pp("cv2"), c_in, c_hidden, 1, 1, 1, true)?;
        let cv3: ConvBlock = ConvBlock::load(vb.pp("cv3"), 2 * c_hidden, c_out, 1, 1, 1, true)?;

        let m: C3kInternal = if use_psa {
            // When attn=True in parent C3k2, m is replaced by a single PSABlock
            let num_heads: usize = (c_hidden / 64).max(1);
            let psa: PsaBlock = PsaBlock::load(vb.pp("m"), c_hidden, num_heads)?;
            C3kInternal::Psa(psa)
        } else {
            let mut bottlenecks: Vec<Bottleneck> = Vec::with_capacity(n);
            for i in 0..n {
                let b: Bottleneck = Bottleneck::load(
                    vb.pp("m").pp(i.to_string()),
                    c_hidden,
                    c_hidden,
                    has_shortcut,
                    (3, 3),
                )?;
                bottlenecks.push(b);
            }
            C3kInternal::Bottlenecks(bottlenecks)
        };

        Ok(Self { cv1, cv2, cv3, m })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a: Tensor = self.cv1.forward(x)?;
        let a: Tensor = match &self.m {
            C3kInternal::Bottlenecks(bns) => {
                let mut y: Tensor = a;
                for bn in bns {
                    y = bn.forward(&y)?;
                }
                y
            }
            C3kInternal::Psa(psa) => psa.forward(&a)?,
        };
        let b: Tensor = self.cv2.forward(x)?;
        let cat: Tensor = Tensor::cat(&[&a, &b], 1)?;
        self.cv3.forward(&cat)
    }
}

// ---------------------------------------------------------------------------
// C3k2: C2f variant with configurable branch type
// cv1(c1→2*c_, 1×1) → split → n branches → cv2((2+n)*c_→c2, 1×1)
// Branches: Bottleneck (c3k=false) or C3k (c3k=true)
// ---------------------------------------------------------------------------

pub enum C3k2Branch {
    BottleneckBranch(Bottleneck),
    C3kBranch(C3k),
}

impl C3k2Branch {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            C3k2Branch::BottleneckBranch(b) => b.forward(x),
            C3k2Branch::C3kBranch(c) => c.forward(x),
        }
    }
}

pub struct C3k2 {
    cv1: ConvBlock,
    cv2: ConvBlock,
    branches: Vec<C3k2Branch>,
    c_hidden: usize,
}

impl C3k2 {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        n: usize,
        is_c3k: bool,
        expansion: f32,
        has_shortcut: bool,
        has_attn: bool,
    ) -> Result<Self> {
        let c_hidden: usize = (c_out as f32 * expansion) as usize;
        let cv1: ConvBlock = ConvBlock::load(vb.pp("cv1"), c_in, 2 * c_hidden, 1, 1, 1, true)?;
        let cv2: ConvBlock =
            ConvBlock::load(vb.pp("cv2"), (2 + n) * c_hidden, c_out, 1, 1, 1, true)?;

        let mut branches: Vec<C3k2Branch> = Vec::with_capacity(n);
        for i in 0..n {
            // Last branch gets PSABlock when attn=True and c3k=True
            let use_psa: bool = has_attn && is_c3k && i == n - 1;
            let branch: C3k2Branch = if is_c3k {
                let c3k: C3k = C3k::load(
                    vb.pp("m").pp(i.to_string()),
                    c_hidden,
                    c_hidden,
                    2,
                    has_shortcut,
                    use_psa,
                )?;
                C3k2Branch::C3kBranch(c3k)
            } else {
                let b: Bottleneck = Bottleneck::load(
                    vb.pp("m").pp(i.to_string()),
                    c_hidden,
                    c_hidden,
                    has_shortcut,
                    (3, 3),
                )?;
                C3k2Branch::BottleneckBranch(b)
            };
            branches.push(branch);
        }

        Ok(Self {
            cv1,
            cv2,
            branches,
            c_hidden,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y: Tensor = self.cv1.forward(x)?;
        // Split cv1 output into 2 chunks of c_hidden each
        let mut chunks: Vec<Tensor> = y.chunk(2, 1)?;
        // Apply each branch to the last chunk, appending results
        for branch in &self.branches {
            let last: &Tensor = chunks.last().unwrap();
            let result: Tensor = branch.forward(last)?;
            chunks.push(result);
        }
        let cat: Tensor = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 1)?;
        self.cv2.forward(&cat)
    }
}

// ---------------------------------------------------------------------------
// SPPF: Spatial Pyramid Pooling Fast
// cv1(c1→c1, 1×1, act=false) → n sequential MaxPool2d(k, s=1, p=k/2) → cat → cv2 → + shortcut
// ---------------------------------------------------------------------------

pub struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    kernel_size: usize,
    pool_count: usize,
    has_shortcut: bool,
}

impl Sppf {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        pool_count: usize,
        has_shortcut: bool,
    ) -> Result<Self> {
        // cv1: c1→c1 with act=false
        let cv1: ConvBlock = ConvBlock::load(vb.pp("cv1"), c_in, c_in, 1, 1, 1, false)?;
        // cv2: c1*(pool_count+1) → c2
        let cv2: ConvBlock =
            ConvBlock::load(vb.pp("cv2"), c_in * (pool_count + 1), c_out, 1, 1, 1, true)?;
        let has_shortcut: bool = has_shortcut && c_in == c_out;
        Ok(Self {
            cv1,
            cv2,
            kernel_size,
            pool_count,
            has_shortcut,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.cv1.forward(input)?;
        let mut pools: Vec<Tensor> = vec![x.clone()];
        let mut current: Tensor = x;

        for _ in 0..self.pool_count {
            // Pad with zeros then maxpool (approximates -inf padding)
            let pad: usize = self.kernel_size / 2;
            let padded: Tensor = current.pad_with_zeros(2, pad, pad)?;
            let padded: Tensor = padded.pad_with_zeros(3, pad, pad)?;
            current = padded.max_pool2d_with_stride(self.kernel_size, 1)?;
            pools.push(current.clone());
        }

        let cat: Tensor = Tensor::cat(&pools.iter().collect::<Vec<_>>(), 1)?;
        let out: Tensor = self.cv2.forward(&cat)?;
        if self.has_shortcut {
            input + out
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Attention: Multi-head self-attention via 1×1 Conv QKV + DWConv positional encoding
// ---------------------------------------------------------------------------

pub struct Attention {
    qkv: ConvBlock,
    proj: ConvBlock,
    pe: ConvBlock, // DWConv for positional encoding
    num_heads: usize,
    key_dim: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim: usize = dim / num_heads;
        let key_dim: usize = head_dim / 2; // attn_ratio=0.5
        let nh_kd: usize = num_heads * key_dim;
        let h: usize = dim + nh_kd * 2;

        let qkv: ConvBlock = ConvBlock::load(vb.pp("qkv"), dim, h, 1, 1, 1, false)?;
        let proj: ConvBlock = ConvBlock::load(vb.pp("proj"), dim, dim, 1, 1, 1, false)?;
        let pe: ConvBlock = ConvBlock::load(vb.pp("pe"), dim, dim, 3, 1, dim, false)?;

        let scale: f64 = (key_dim as f64).powf(-0.5);

        Ok(Self {
            qkv,
            proj,
            pe,
            num_heads,
            key_dim,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let n: usize = h * w;

        // QKV projection: [B, C, H, W] → [B, h, H, W] where h = dim + 2*nh_kd
        let qkv: Tensor = self.qkv.forward(x)?;

        // Reshape to [B, num_heads, key_dim*2+head_dim, N]
        let qkv: Tensor = qkv.reshape((b, self.num_heads, self.key_dim * 2 + self.head_dim, n))?;

        // Split into q, k, v
        let q: Tensor = qkv.narrow(2, 0, self.key_dim)?;
        let k: Tensor = qkv.narrow(2, self.key_dim, self.key_dim)?;
        let v: Tensor = qkv.narrow(2, self.key_dim * 2, self.head_dim)?;

        // attn = softmax(q^T @ k * scale)
        let q_t: Tensor = q.transpose(2, 3)?; // [B, nh, N, kd]
        let attn: Tensor = q_t.matmul(&k)?; // [B, nh, N, N]
        let attn: Tensor = (attn * self.scale)?;
        let attn: Tensor = candle_nn::ops::softmax_last_dim(&attn)?;

        // out = v @ attn^T
        let attn_t: Tensor = attn.transpose(2, 3)?; // [B, nh, N, N]
        let out: Tensor = v.matmul(&attn_t)?; // [B, nh, hd, N]
        let out: Tensor = out.reshape((b, c, h, w))?;

        // Add positional encoding: pe is applied to v reshaped as spatial
        let v_spatial: Tensor = v.reshape((b, c, h, w))?;
        let pe: Tensor = self.pe.forward(&v_spatial)?;
        let out: Tensor = (out + pe)?;

        self.proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// PSABlock: Attention + FFN with residual connections
// x = x + attn(x); x = x + ffn(x)
// ---------------------------------------------------------------------------

pub struct PsaBlock {
    attn: Attention,
    ffn_0: ConvBlock, // Conv(c→2c, k=1, act=true)
    ffn_1: ConvBlock, // Conv(2c→c, k=1, act=false)
}

impl PsaBlock {
    pub fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let attn: Attention = Attention::load(vb.pp("attn"), dim, num_heads)?;
        let ffn_0: ConvBlock = ConvBlock::load(vb.pp("ffn").pp("0"), dim, dim * 2, 1, 1, 1, true)?;
        let ffn_1: ConvBlock = ConvBlock::load(vb.pp("ffn").pp("1"), dim * 2, dim, 1, 1, 1, false)?;
        Ok(Self { attn, ffn_0, ffn_1 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = (x + self.attn.forward(x)?)?;
        let ffn_out: Tensor = self.ffn_1.forward(&self.ffn_0.forward(&x)?)?;
        &x + ffn_out
    }
}

// ---------------------------------------------------------------------------
// C2PSA: Conv split → n PSABlock → Conv merge
// ---------------------------------------------------------------------------

pub struct C2psa {
    cv1: ConvBlock,
    cv2: ConvBlock,
    m: Vec<PsaBlock>,
    c_split: usize,
}

impl C2psa {
    pub fn load(vb: VarBuilder, c_in: usize, c_out: usize, n: usize) -> Result<Self> {
        assert_eq!(c_in, c_out, "C2PSA requires c_in == c_out");
        let c_split: usize = c_in / 2; // e=0.5
        let cv1: ConvBlock = ConvBlock::load(vb.pp("cv1"), c_in, 2 * c_split, 1, 1, 1, true)?;
        let cv2: ConvBlock = ConvBlock::load(vb.pp("cv2"), 2 * c_split, c_out, 1, 1, 1, true)?;

        let num_heads: usize = (c_split / 64).max(1);
        let mut m: Vec<PsaBlock> = Vec::with_capacity(n);
        for i in 0..n {
            m.push(PsaBlock::load(
                vb.pp("m").pp(i.to_string()),
                c_split,
                num_heads,
            )?);
        }

        Ok(Self {
            cv1,
            cv2,
            m,
            c_split,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y: Tensor = self.cv1.forward(x)?;
        // Split into two halves: a stays, b goes through PSABlocks
        let a: Tensor = y.narrow(1, 0, self.c_split)?;
        let mut b: Tensor = y.narrow(1, self.c_split, self.c_split)?;
        for psa in &self.m {
            b = psa.forward(&b)?;
        }
        let cat: Tensor = Tensor::cat(&[&a, &b], 1)?;
        self.cv2.forward(&cat)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn test_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    #[test]
    fn test_conv_block_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let block = ConvBlock::load(vb.pp("test"), 16, 32, 3, 2, 1, true).unwrap();
        let x = Tensor::zeros((1, 16, 640, 640), DType::F32, &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 32, 320, 320]);
    }

    #[test]
    fn test_conv_block_no_act() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let block = ConvBlock::load(vb.pp("test"), 32, 64, 1, 1, 1, false).unwrap();
        let x = Tensor::zeros((1, 32, 80, 80), DType::F32, &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 64, 80, 80]);
    }

    #[test]
    fn test_bottleneck_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let b = Bottleneck::load(vb.pp("test"), 64, 64, true, (3, 3)).unwrap();
        let x = Tensor::zeros((1, 64, 40, 40), DType::F32, &device).unwrap();
        let y = b.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 64, 40, 40]);
    }

    #[test]
    fn test_bottleneck_no_shortcut() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let b = Bottleneck::load(vb.pp("test"), 64, 32, false, (3, 3)).unwrap();
        let x = Tensor::zeros((1, 64, 40, 40), DType::F32, &device).unwrap();
        let y = b.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 32, 40, 40]);
    }

    #[test]
    fn test_dwconv_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        // DWConv = ConvBlock with groups=c
        let block = ConvBlock::load(vb.pp("test"), 64, 64, 3, 1, 64, true).unwrap();
        let x = Tensor::zeros((1, 64, 40, 40), DType::F32, &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 64, 40, 40]);
    }

    #[test]
    fn test_c3k_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let c = C3k::load(vb.pp("test"), 128, 128, 2, true, false).unwrap();
        let x = Tensor::zeros((1, 128, 20, 20), DType::F32, &device).unwrap();
        let y = c.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 128, 20, 20]);
    }

    #[test]
    fn test_c3k2_no_c3k() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        // Layer 2: C3k2(32→64, c3k=F, e=0.25, n=1)
        let c = C3k2::load(vb.pp("test"), 32, 64, 1, false, 0.25, true, false).unwrap();
        let x = Tensor::zeros((1, 32, 160, 160), DType::F32, &device).unwrap();
        let y = c.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 64, 160, 160]);
    }

    #[test]
    fn test_c3k2_with_c3k() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        // Layer 6: C3k2(128→128, c3k=T, e=0.5, n=1)
        let c = C3k2::load(vb.pp("test"), 128, 128, 1, true, 0.5, true, false).unwrap();
        let x = Tensor::zeros((1, 128, 40, 40), DType::F32, &device).unwrap();
        let y = c.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 128, 40, 40]);
    }

    #[test]
    fn test_sppf_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let s = Sppf::load(vb.pp("test"), 256, 256, 5, 3, true).unwrap();
        let x = Tensor::zeros((1, 256, 20, 20), DType::F32, &device).unwrap();
        let y = s.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 256, 20, 20]);
    }

    #[test]
    fn test_attention_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let a = Attention::load(vb.pp("test"), 128, 2).unwrap();
        let x = Tensor::zeros((1, 128, 20, 20), DType::F32, &device).unwrap();
        let y = a.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 128, 20, 20]);
    }

    #[test]
    fn test_psa_block_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let p = PsaBlock::load(vb.pp("test"), 128, 2).unwrap();
        let x = Tensor::zeros((1, 128, 20, 20), DType::F32, &device).unwrap();
        let y = p.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 128, 20, 20]);
    }

    #[test]
    fn test_c2psa_shape() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        let c = C2psa::load(vb.pp("test"), 256, 256, 1).unwrap();
        let x = Tensor::zeros((1, 256, 20, 20), DType::F32, &device).unwrap();
        let y = c.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 256, 20, 20]);
    }

    #[test]
    fn test_c3k2_with_attn() {
        let device = Device::Cpu;
        let vb = test_vb(&device);
        // Layer 22: C3k2(384→256, c3k=T, e=0.5, n=1, attn=T)
        let c = C3k2::load(vb.pp("test"), 384, 256, 1, true, 0.5, true, true).unwrap();
        let x = Tensor::zeros((1, 384, 20, 20), DType::F32, &device).unwrap();
        let y = c.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 256, 20, 20]);
    }
}
