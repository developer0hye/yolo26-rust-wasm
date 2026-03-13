pub mod model;
pub mod postprocess;
pub mod preprocess;

use std::sync::OnceLock;

use wasm_bindgen::prelude::*;

use model::Yolo26Model;
use postprocess::{postprocess, DetectionResult};
use preprocess::preprocess;

static MODEL: OnceLock<Yolo26Model> = OnceLock::new();

/// Load SafeTensors model bytes into memory. Called once on page load.
#[wasm_bindgen]
pub fn init_model(weights: &[u8]) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let device = candle_core::Device::Cpu;
    let model = Yolo26Model::load(weights.to_vec(), &device)
        .map_err(|e| JsValue::from_str(&format!("Failed to load model: {e}")))?;
    MODEL
        .set(model)
        .map_err(|_| JsValue::from_str("Model already initialized"))?;
    Ok(())
}

/// Run inference on RGBA pixels. Returns JSON string with detections.
/// The HTML demo passes confidence_threshold=0.0 and filters in JS.
#[wasm_bindgen]
pub fn detect(
    pixels: &[u8],
    width: u32,
    height: u32,
    confidence_threshold: f32,
) -> Result<String, JsValue> {
    let model: &Yolo26Model = MODEL
        .get()
        .ok_or_else(|| JsValue::from_str("Model not loaded yet"))?;
    let device = candle_core::Device::Cpu;

    #[cfg(target_arch = "wasm32")]
    let start: f64 = js_sys::Date::now();
    #[cfg(not(target_arch = "wasm32"))]
    let start: std::time::Instant = std::time::Instant::now();

    let (input, letterbox) = preprocess(pixels, width, height, &device)
        .map_err(|e| JsValue::from_str(&format!("Preprocess error: {e}")))?;
    let output = model
        .forward(&input)
        .map_err(|e| JsValue::from_str(&format!("Inference error: {e}")))?;
    let detections = postprocess(&output, &letterbox, width, height, confidence_threshold)
        .map_err(|e| JsValue::from_str(&format!("Postprocess error: {e}")))?;

    #[cfg(target_arch = "wasm32")]
    let elapsed_ms: f64 = js_sys::Date::now() - start;
    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms: f64 = start.elapsed().as_secs_f64() * 1000.0;

    let result = DetectionResult {
        detections,
        inference_time_ms: elapsed_ms,
        image_width: width,
        image_height: height,
    };
    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&format!("JSON error: {e}")))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use crate::model::Yolo26Model;
    use crate::postprocess::postprocess;
    use crate::preprocess::preprocess;

    /// Full pipeline test with VarMap (random weights) — verifies shapes only.
    #[test]
    fn test_detect_pipeline_shapes() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Build model from VarMap (random weights)
        let backbone = crate::model::backbone::Backbone::load(vb.pp("model")).unwrap();
        let neck = crate::model::neck::Neck::load(vb.pp("model")).unwrap();
        let head =
            crate::model::head::Detect::load(vb.pp("model").pp("23"), &[64, 128, 256], 80).unwrap();
        let model = Yolo26Model::new_from_parts(backbone, neck, head);

        // Synthetic 100x75 RGBA image
        let width: u32 = 100;
        let height: u32 = 75;
        let rgba = vec![128u8; (width * height * 4) as usize];

        let (input, letterbox) = preprocess(&rgba, width, height, &device).unwrap();
        assert_eq!(input.dims(), &[1, 3, 640, 640]);

        let output: Tensor = model.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 300, 6]);

        let dets = postprocess(&output, &letterbox, width, height, 0.5).unwrap();
        // With random weights, detections may or may not pass threshold — just verify no crash
        assert!(dets.len() <= 300);
    }
}
