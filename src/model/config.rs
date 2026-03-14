/// YOLO26 model scale variants matching ultralytics yolo26.yaml scaling parameters.
///
/// Each scale defines (depth, width, max_channels) that control:
/// - Channel widths: `make_divisible(min(yaml_c, max_channels) * width, 8)`
/// - Repeat counts: `max(round(yaml_n * depth), 1)` for yaml_n > 1
/// - C3k2 branch type: m/l/x override all C3k2 blocks to use C3k branches
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelScale {
    N,
    S,
    M,
    L,
    X,
}

impl ModelScale {
    /// Parse scale from model name: "yolo26n" → N, "yolo26m" → M, etc.
    /// Accepts optional extensions like "yolo26m.safetensors".
    pub fn from_model_name(name: &str) -> Option<Self> {
        let name: &str = name
            .strip_suffix(".safetensors")
            .or_else(|| name.strip_suffix(".pt"))
            .unwrap_or(name);
        let suffix: &str = name.strip_prefix("yolo26")?;
        match suffix {
            "n" => Some(Self::N),
            "s" => Some(Self::S),
            "m" => Some(Self::M),
            "l" => Some(Self::L),
            "x" => Some(Self::X),
            _ => None,
        }
    }

    /// Scaling parameters: (depth, width, max_channels).
    fn params(&self) -> (f32, f32, usize) {
        match self {
            Self::N => (0.50, 0.25, 1024),
            Self::S => (0.50, 0.50, 1024),
            Self::M => (0.50, 1.00, 512),
            Self::L => (1.00, 1.00, 512),
            Self::X => (1.00, 1.50, 512),
        }
    }

    /// Whether ALL C3k2 blocks use C3k branches (overridden for m/l/x in
    /// ultralytics parse_model: `if scale in "mlx": args[3] = True`).
    pub fn c3k_all(&self) -> bool {
        matches!(self, Self::M | Self::L | Self::X)
    }

    /// Scale a YAML channel value: `ceil(min(yaml_c, max_channels) * width / 8) * 8`.
    pub fn channel(&self, yaml_c: usize) -> usize {
        let (_, width, max_ch): (f32, f32, usize) = self.params();
        let raw: f32 = yaml_c.min(max_ch) as f32 * width;
        ((raw / 8.0).ceil() as usize) * 8
    }

    /// Scale a YAML repeat count: `max(round(yaml_n * depth), 1)` for yaml_n > 1.
    pub fn repeat(&self, yaml_n: usize) -> usize {
        if yaml_n > 1 {
            let (depth, _, _): (f32, f32, usize) = self.params();
            (yaml_n as f32 * depth).round().max(1.0) as usize
        } else {
            yaml_n
        }
    }

    /// Detection head input channels: [small(P3), medium(P4), large(P5)].
    /// Corresponds to neck outputs at layers 16, 19, 22 with YAML c2 = [256, 512, 1024].
    pub fn head_input_channels(&self) -> [usize; 3] {
        [self.channel(256), self.channel(512), self.channel(1024)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_model_name() {
        assert_eq!(ModelScale::from_model_name("yolo26n"), Some(ModelScale::N));
        assert_eq!(ModelScale::from_model_name("yolo26s"), Some(ModelScale::S));
        assert_eq!(ModelScale::from_model_name("yolo26m"), Some(ModelScale::M));
        assert_eq!(ModelScale::from_model_name("yolo26l"), Some(ModelScale::L));
        assert_eq!(ModelScale::from_model_name("yolo26x"), Some(ModelScale::X));
        assert_eq!(
            ModelScale::from_model_name("yolo26m.safetensors"),
            Some(ModelScale::M)
        );
        assert_eq!(
            ModelScale::from_model_name("yolo26x.pt"),
            Some(ModelScale::X)
        );
        assert_eq!(ModelScale::from_model_name("yolo26z"), None);
        assert_eq!(ModelScale::from_model_name("yolov8n"), None);
    }

    #[test]
    fn test_c3k_all() {
        assert!(!ModelScale::N.c3k_all());
        assert!(!ModelScale::S.c3k_all());
        assert!(ModelScale::M.c3k_all());
        assert!(ModelScale::L.c3k_all());
        assert!(ModelScale::X.c3k_all());
    }

    // Verify channel scaling matches ultralytics yolo26.yaml parameter summary.
    // Cross-referenced with actual .safetensors weight shapes (model.0.conv.weight, model.1.conv.weight).

    #[test]
    fn test_channel_n() {
        let s: ModelScale = ModelScale::N;
        assert_eq!(s.channel(64), 16);
        assert_eq!(s.channel(128), 32);
        assert_eq!(s.channel(256), 64);
        assert_eq!(s.channel(512), 128);
        assert_eq!(s.channel(1024), 256);
    }

    #[test]
    fn test_channel_s() {
        let s: ModelScale = ModelScale::S;
        assert_eq!(s.channel(64), 32);
        assert_eq!(s.channel(128), 64);
        assert_eq!(s.channel(256), 128);
        assert_eq!(s.channel(512), 256);
        assert_eq!(s.channel(1024), 512);
    }

    #[test]
    fn test_channel_m() {
        let s: ModelScale = ModelScale::M;
        assert_eq!(s.channel(64), 64);
        assert_eq!(s.channel(128), 128);
        assert_eq!(s.channel(256), 256);
        assert_eq!(s.channel(512), 512);
        assert_eq!(s.channel(1024), 512); // capped by max_channels=512
    }

    #[test]
    fn test_channel_l() {
        let s: ModelScale = ModelScale::L;
        // Same as M (same width=1.0, max_ch=512)
        assert_eq!(s.channel(64), 64);
        assert_eq!(s.channel(128), 128);
        assert_eq!(s.channel(256), 256);
        assert_eq!(s.channel(512), 512);
        assert_eq!(s.channel(1024), 512);
    }

    #[test]
    fn test_channel_x() {
        let s: ModelScale = ModelScale::X;
        assert_eq!(s.channel(64), 96);
        assert_eq!(s.channel(128), 192);
        assert_eq!(s.channel(256), 384);
        assert_eq!(s.channel(512), 768);
        assert_eq!(s.channel(1024), 768); // min(1024,512)*1.5=768
    }

    #[test]
    fn test_repeat() {
        // depth=0.5: yaml_n=2 → round(1.0) = 1
        assert_eq!(ModelScale::N.repeat(2), 1);
        assert_eq!(ModelScale::S.repeat(2), 1);
        assert_eq!(ModelScale::M.repeat(2), 1);
        // depth=1.0: yaml_n=2 → round(2.0) = 2
        assert_eq!(ModelScale::L.repeat(2), 2);
        assert_eq!(ModelScale::X.repeat(2), 2);
        // yaml_n=1 is never scaled
        assert_eq!(ModelScale::L.repeat(1), 1);
        assert_eq!(ModelScale::X.repeat(1), 1);
    }

    #[test]
    fn test_head_input_channels() {
        assert_eq!(ModelScale::N.head_input_channels(), [64, 128, 256]);
        assert_eq!(ModelScale::S.head_input_channels(), [128, 256, 512]);
        assert_eq!(ModelScale::M.head_input_channels(), [256, 512, 512]);
        assert_eq!(ModelScale::L.head_input_channels(), [256, 512, 512]);
        assert_eq!(ModelScale::X.head_input_channels(), [384, 768, 768]);
    }
}
