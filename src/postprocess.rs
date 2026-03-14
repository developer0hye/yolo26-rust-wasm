use candle_core::{Result, Tensor};
use serde::Serialize;

use crate::preprocess::LetterboxInfo;

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
}

#[derive(Debug, Serialize)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
    pub inference_time_ms: f64,
    pub image_width: u32,
    pub image_height: u32,
}

#[rustfmt::skip]
pub const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// Parse model output tensor [1, 300, 6] into detections in original image coordinates.
///
/// Each row: [x1, y1, x2, y2, confidence, class_id] in model (640x640) space.
/// Transforms back to original image coordinates using LetterboxInfo.
pub fn postprocess(
    output: &Tensor,
    letterbox: &LetterboxInfo,
    img_width: u32,
    img_height: u32,
    confidence_threshold: f32,
) -> Result<Vec<Detection>> {
    // Squeeze batch dim: [1, 300, 6] → [300, 6]
    let output: Tensor = output.squeeze(0)?;
    let num_detections: usize = output.dim(0)?;
    let data: Vec<f32> = output.flatten_all()?.to_vec1::<f32>()?;

    let w_f: f32 = img_width as f32;
    let h_f: f32 = img_height as f32;

    let mut detections: Vec<Detection> = Vec::new();

    for i in 0..num_detections {
        let base: usize = i * 6;
        let confidence: f32 = data[base + 4];

        if confidence < confidence_threshold {
            continue;
        }

        let class_id_raw: f32 = data[base + 5];
        let class_id: u32 = class_id_raw.round() as u32;

        // Reverse letterbox transform: model coords → original image coords
        let x1: f32 = (data[base] - letterbox.pad_x) / letterbox.scale;
        let y1: f32 = (data[base + 1] - letterbox.pad_y) / letterbox.scale;
        let x2: f32 = (data[base + 2] - letterbox.pad_x) / letterbox.scale;
        let y2: f32 = (data[base + 3] - letterbox.pad_y) / letterbox.scale;

        // Clamp to image bounds
        let x1_clamped: f32 = x1.max(0.0).min(w_f);
        let y1_clamped: f32 = y1.max(0.0).min(h_f);
        let x2_clamped: f32 = x2.max(0.0).min(w_f);
        let y2_clamped: f32 = y2.max(0.0).min(h_f);

        let class_name: String = if (class_id as usize) < COCO_CLASSES.len() {
            COCO_CLASSES[class_id as usize].to_string()
        } else {
            format!("class_{class_id}")
        };

        detections.push(Detection {
            x: x1_clamped,
            y: y1_clamped,
            width: x2_clamped - x1_clamped,
            height: y2_clamped - y1_clamped,
            confidence,
            class_id,
            class_name,
        });
    }

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Helper: build a [1, 300, 6] tensor with specified rows (rest filled with zeros).
    fn make_output_tensor(rows: &[[f32; 6]], device: &Device) -> Tensor {
        let mut data = vec![0.0f32; 300 * 6];
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[i * 6 + j] = val;
            }
        }
        Tensor::from_vec(data, (1, 300, 6), device).unwrap()
    }

    fn default_letterbox() -> LetterboxInfo {
        LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    #[test]
    fn test_parse_detections() {
        let device = Device::Cpu;
        // [x1, y1, x2, y2, confidence, class_id]
        let tensor = make_output_tensor(
            &[
                [100.0, 50.0, 300.0, 450.0, 0.92, 0.0],  // person
                [500.0, 200.0, 600.0, 350.0, 0.85, 2.0], // car
            ],
            &device,
        );
        let dets = postprocess(&tensor, &default_letterbox(), 640, 640, 0.5).unwrap();
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].class_name, "person");
        assert!((dets[0].confidence - 0.92).abs() < 1e-3);
        assert_eq!(dets[1].class_name, "car");
    }

    #[test]
    fn test_confidence_filter() {
        let device = Device::Cpu;
        let tensor = make_output_tensor(
            &[
                [10.0, 10.0, 50.0, 50.0, 0.9, 0.0],
                [60.0, 60.0, 100.0, 100.0, 0.3, 1.0], // below threshold
                [120.0, 120.0, 200.0, 200.0, 0.7, 2.0],
            ],
            &device,
        );
        let dets = postprocess(&tensor, &default_letterbox(), 640, 640, 0.5).unwrap();
        assert_eq!(dets.len(), 2);
        assert!(dets.iter().all(|d| d.confidence >= 0.5));
    }

    #[test]
    fn test_coordinate_transform() {
        let device = Device::Cpu;
        // Landscape 1280x640: scale=0.5, pad_y=160
        // Model-space box: x1=100, y1=200, x2=300, y2=400
        // Reverse: x = (100-0)/0.5 = 200, y = (200-160)/0.5 = 80
        //          x2 = (300-0)/0.5 = 600, y2 = (400-160)/0.5 = 480
        let letterbox = LetterboxInfo {
            scale: 0.5,
            pad_x: 0.0,
            pad_y: 160.0,
        };
        let tensor = make_output_tensor(&[[100.0, 200.0, 300.0, 400.0, 0.95, 0.0]], &device);
        let dets = postprocess(&tensor, &letterbox, 1280, 640, 0.1).unwrap();
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!((d.x - 200.0).abs() < 1.0);
        assert!((d.y - 80.0).abs() < 1.0);
        assert!((d.width - 400.0).abs() < 1.0); // 600 - 200
        assert!((d.height - 400.0).abs() < 1.0); // 480 - 80
    }

    #[test]
    fn test_no_detections() {
        let device = Device::Cpu;
        let tensor = make_output_tensor(
            &[[10.0, 10.0, 50.0, 50.0, 0.1, 0.0]], // low confidence
            &device,
        );
        let dets = postprocess(&tensor, &default_letterbox(), 640, 640, 0.5).unwrap();
        assert!(dets.is_empty());
    }

    #[test]
    fn test_clamp_bounds() {
        let device = Device::Cpu;
        // Box extends beyond image after reverse transform
        let letterbox = LetterboxInfo {
            scale: 0.5,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let tensor = make_output_tensor(&[[0.0, 0.0, 640.0, 640.0, 0.99, 0.0]], &device);
        // Original image is 100x100, so coords should be clamped
        let dets = postprocess(&tensor, &letterbox, 100, 100, 0.1).unwrap();
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!(d.x >= 0.0);
        assert!(d.y >= 0.0);
        assert!(d.x + d.width <= 100.0 + 1.0); // small tolerance
        assert!(d.y + d.height <= 100.0 + 1.0);
    }

    #[test]
    fn test_coco_class_name() {
        assert_eq!(COCO_CLASSES[0], "person");
        assert_eq!(COCO_CLASSES[2], "car");
        assert_eq!(COCO_CLASSES[79], "toothbrush");
    }

    #[test]
    fn test_json_format() {
        let device = Device::Cpu;
        let tensor = make_output_tensor(
            &[[100.0, 50.0, 200.0, 150.0, 0.88, 15.0]], // cat
            &device,
        );
        let dets = postprocess(&tensor, &default_letterbox(), 640, 640, 0.1).unwrap();
        let result = DetectionResult {
            detections: dets,
            inference_time_ms: 100.0,
            image_width: 640,
            image_height: 640,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["detections"].is_array());
        assert_eq!(parsed["detections"][0]["class_name"], "cat");
        assert!(parsed["inference_time_ms"].is_f64());
    }
}
