use candle_core::{DType, Device, Result, Tensor};

pub const MODEL_INPUT_SIZE: usize = 640;
const LETTERBOX_PAD_VALUE: f32 = 114.0 / 255.0;

pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Convert RGBA pixels to model input tensor [1, 3, 640, 640] with letterbox transform.
///
/// Steps: RGBA→RGB, nearest-neighbor resize, letterbox pad, normalize [0,1], HWC→CHW.
pub fn preprocess(
    rgba: &[u8],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<(Tensor, LetterboxInfo)> {
    let (w, h) = (width as usize, height as usize);
    let target: usize = MODEL_INPUT_SIZE;

    // Compute letterbox parameters
    let scale: f32 = f32::min(target as f32 / w as f32, target as f32 / h as f32);
    let new_w: usize = (w as f32 * scale).round() as usize;
    let new_h: usize = (h as f32 * scale).round() as usize;
    let pad_x: f32 = (target as f32 - new_w as f32) / 2.0;
    let pad_y: f32 = (target as f32 - new_h as f32) / 2.0;
    let pad_x_int: usize = pad_x.floor() as usize;
    let pad_y_int: usize = pad_y.floor() as usize;

    // Build CHW f32 buffer directly: [3, target, target]
    let channel_size: usize = target * target;
    let mut chw_buf: Vec<f32> = vec![LETTERBOX_PAD_VALUE; 3 * channel_size];

    // Nearest-neighbor resize + RGBA→RGB + normalize, writing directly into CHW layout
    for dst_y in 0..new_h {
        let src_y: usize = (dst_y * h / new_h).min(h - 1);
        for dst_x in 0..new_w {
            let src_x: usize = (dst_x * w / new_w).min(w - 1);
            let src_idx: usize = (src_y * w + src_x) * 4;
            let out_y: usize = dst_y + pad_y_int;
            let out_x: usize = dst_x + pad_x_int;
            let out_pos: usize = out_y * target + out_x;

            // RGB channels normalized to [0, 1]
            chw_buf[out_pos] = rgba[src_idx] as f32 / 255.0; // R
            chw_buf[channel_size + out_pos] = rgba[src_idx + 1] as f32 / 255.0; // G
            chw_buf[2 * channel_size + out_pos] = rgba[src_idx + 2] as f32 / 255.0;
            // B
        }
    }

    // Create tensor [1, 3, 640, 640]
    let tensor: Tensor =
        Tensor::from_vec(chw_buf, (1, 3, target, target), device)?.to_dtype(DType::F32)?;

    let info = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };
    Ok((tensor, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_to_rgb_strips_alpha() {
        // 2x2 RGBA image: red, green, blue, white (with varying alpha)
        let rgba: Vec<u8> = vec![
            255, 0, 0, 128, // red pixel, alpha=128
            0, 255, 0, 255, // green pixel, alpha=255
            0, 0, 255, 0, // blue pixel, alpha=0
            255, 255, 255, 64, // white pixel, alpha=64
        ];
        let device = Device::Cpu;
        let (tensor, _) = preprocess(&rgba, 2, 2, &device).unwrap();
        // Shape must be [1, 3, 640, 640]
        assert_eq!(tensor.dims(), &[1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    }

    #[test]
    fn test_letterbox_square_input() {
        // 640x640 input → no padding needed
        let rgba = vec![128u8; 640 * 640 * 4];
        let device = Device::Cpu;
        let (_, info) = preprocess(&rgba, 640, 640, &device).unwrap();
        assert!((info.scale - 1.0).abs() < 1e-5);
        assert!((info.pad_x - 0.0).abs() < 1e-5);
        assert!((info.pad_y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_letterbox_landscape() {
        // 1280x640 → scale=0.5, new_size=640x320, pad_y=(640-320)/2=160
        let rgba = vec![128u8; 1280 * 640 * 4];
        let device = Device::Cpu;
        let (tensor, info) = preprocess(&rgba, 1280, 640, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 640, 640]);
        assert!((info.scale - 0.5).abs() < 1e-5);
        assert!((info.pad_x - 0.0).abs() < 1e-5);
        assert!((info.pad_y - 160.0).abs() < 1e-5);
    }

    #[test]
    fn test_letterbox_portrait() {
        // 640x1280 → scale=0.5, new_size=320x640, pad_x=(640-320)/2=160
        let rgba = vec![128u8; 640 * 1280 * 4];
        let device = Device::Cpu;
        let (tensor, info) = preprocess(&rgba, 640, 1280, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 640, 640]);
        assert!((info.scale - 0.5).abs() < 1e-5);
        assert!((info.pad_x - 160.0).abs() < 1e-5);
        assert!((info.pad_y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_range() {
        // All-white image: pixel value 255 → should be 1.0 in output (after normalize)
        // All-black image: pixel value 0 → should be 0.0
        let white_rgba = vec![255u8; 4 * 4 * 4]; // 4x4 white
        let device = Device::Cpu;
        let (tensor, _) = preprocess(&white_rgba, 4, 4, &device).unwrap();
        let data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // The top-left pixel area (inside the resized image) should be close to 1.0
        // Pad area should be close to LETTERBOX_PAD_VALUE
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_val - 1.0).abs() < 1e-3,
            "Max value should be ~1.0, got {max_val}"
        );

        let black_rgba = vec![0u8; 4 * 4 * 4]; // 4x4 black (alpha=0 too)
        let (tensor_b, _) = preprocess(&black_rgba, 4, 4, &device).unwrap();
        let data_b = tensor_b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let min_val = data_b.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min_val.abs() < 1e-3,
            "Min value should be ~0.0, got {min_val}"
        );
    }

    #[test]
    fn test_output_tensor_shape() {
        let rgba = vec![100u8; 320 * 240 * 4]; // arbitrary non-square
        let device = Device::Cpu;
        let (tensor, _) = preprocess(&rgba, 320, 240, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    }

    #[test]
    fn test_letterbox_info_nonsquare() {
        // 800x600 → scale=min(640/800, 640/600)=min(0.8, 1.067)=0.8
        // new_w=640, new_h=480, pad_x=0, pad_y=(640-480)/2=80
        let rgba = vec![128u8; 800 * 600 * 4];
        let device = Device::Cpu;
        let (_, info) = preprocess(&rgba, 800, 600, &device).unwrap();
        assert!((info.scale - 0.8).abs() < 1e-5);
        assert!((info.pad_x - 0.0).abs() < 1e-5);
        assert!((info.pad_y - 80.0).abs() < 1e-5);
    }
}
