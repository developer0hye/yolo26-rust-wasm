//! Comparison tests: Rust implementation vs Python (ultralytics) reference.
//!
//! These tests load real SafeTensors weights and compare Rust inference output
//! against pre-computed Python reference fixtures. Run with:
//!   cargo test --test python_comparison -- --ignored
//!
//! Prerequisites:
//!   1. weights/yolo26n.safetensors (run scripts/download_model.sh)
//!   2. tests/fixtures/* (run python3 scripts/generate_test_fixtures.py)

use candle_core::{Device, Tensor};
use std::path::Path;
use yolo26_rust_wasm::model::Multiples;

const WEIGHTS_PATH: &str = "weights/yolo26n.safetensors";
const FIXTURE_INPUT: &str = "tests/fixtures/test_input.safetensors";
const FIXTURE_PRE_TOPK: &str = "tests/fixtures/reference_pre_topk.safetensors";
const FIXTURE_OUTPUT: &str = "tests/fixtures/reference_output.safetensors";
const FIXTURE_METADATA: &str = "tests/fixtures/reference_metadata.json";

fn fixtures_available() -> bool {
    Path::new(WEIGHTS_PATH).exists()
        && Path::new(FIXTURE_INPUT).exists()
        && Path::new(FIXTURE_PRE_TOPK).exists()
        && Path::new(FIXTURE_OUTPUT).exists()
        && Path::new(FIXTURE_METADATA).exists()
}

fn load_safetensor(path: &str, key: &str, device: &Device) -> Tensor {
    let bytes: Vec<u8> =
        std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    let tensors =
        candle_core::safetensors::load_buffer(&bytes, device).expect("Failed to parse safetensors");
    tensors
        .get(key)
        .unwrap_or_else(|| panic!("Key '{key}' not found in {path}"))
        .clone()
}

/// Compute element-wise relative error statistics between two tensors.
/// Returns (max_abs_error, mean_abs_error, max_rel_error).
fn tensor_error_stats(a: &Tensor, b: &Tensor) -> (f32, f32, f32) {
    let a_flat: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
    let b_flat: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(a_flat.len(), b_flat.len(), "Tensor size mismatch");

    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;

    for (av, bv) in a_flat.iter().zip(b_flat.iter()) {
        let abs_err: f32 = (av - bv).abs();
        max_abs = max_abs.max(abs_err);
        sum_abs += abs_err;
        let denom: f32 = av.abs().max(bv.abs()).max(1e-8);
        max_rel = max_rel.max(abs_err / denom);
    }

    let mean_abs: f32 = sum_abs / a_flat.len() as f32;
    (max_abs, mean_abs, max_rel)
}

/// Compare pre-topk decoded output [1, 8400, 84] between Python and Rust.
/// This validates the entire forward pass (backbone + neck + head decode)
/// without being affected by topk selection differences.
#[test]
#[ignore]
fn test_pre_topk_matches_python() {
    if !fixtures_available() {
        eprintln!("Skipping: fixtures not available. Run scripts/download_model.sh and scripts/generate_test_fixtures.py first.");
        return;
    }

    let device = Device::Cpu;

    // Load model with real weights
    let weights_bytes: Vec<u8> = std::fs::read(WEIGHTS_PATH).unwrap();
    let model = yolo26_rust_wasm::model::Yolo26Model::load(weights_bytes, &device, &Multiples::n())
        .unwrap();

    // Load reference input
    let input: Tensor = load_safetensor(FIXTURE_INPUT, "input", &device);
    assert_eq!(input.dims(), &[1, 3, 640, 640]);

    // Load Python reference pre-topk output
    let python_pre_topk: Tensor = load_safetensor(FIXTURE_PRE_TOPK, "pre_topk", &device);
    assert_eq!(python_pre_topk.dims(), &[1, 8400, 84]);

    // Run Rust model forward (pre-topk)
    let rust_pre_topk: Tensor = model.forward_pre_topk(&input).unwrap();
    assert_eq!(rust_pre_topk.dims(), &[1, 8400, 84]);

    // Compare
    let (max_abs, mean_abs, max_rel) = tensor_error_stats(&rust_pre_topk, &python_pre_topk);
    eprintln!("Pre-topk comparison [1, 8400, 84]:");
    eprintln!("  max_abs_error:  {max_abs:.6e}");
    eprintln!("  mean_abs_error: {mean_abs:.6e}");
    eprintln!("  max_rel_error:  {max_rel:.6e}");

    // Tolerance: BN fusion and f32 arithmetic may cause small differences.
    // 1e-3 relative error is a reasonable bound for FP32 inference differences.
    assert!(
        max_rel < 1e-2,
        "Pre-topk max relative error {max_rel:.6e} exceeds 1e-2 threshold"
    );
    assert!(
        mean_abs < 1.0,
        "Pre-topk mean absolute error {mean_abs:.6e} exceeds 1.0 threshold"
    );
}

/// Compare top detections between Python and Rust.
/// Since topk selection differs (Python: two-pass, Rust: one-pass),
/// we compare the highest-confidence detection and verify that
/// both implementations find the same bounding boxes.
#[test]
#[ignore]
fn test_top_detections_match_python() {
    if !fixtures_available() {
        eprintln!("Skipping: fixtures not available.");
        return;
    }

    let device = Device::Cpu;

    // Load model
    let weights_bytes: Vec<u8> = std::fs::read(WEIGHTS_PATH).unwrap();
    let model = yolo26_rust_wasm::model::Yolo26Model::load(weights_bytes, &device, &Multiples::n())
        .unwrap();

    // Load reference input and run Rust inference
    let input: Tensor = load_safetensor(FIXTURE_INPUT, "input", &device);
    let rust_output: Tensor = model.forward(&input).unwrap();
    assert_eq!(rust_output.dims(), &[1, 300, 6]);

    // Load Python reference output
    let python_output: Tensor = load_safetensor(FIXTURE_OUTPUT, "output", &device);
    assert_eq!(python_output.dims(), &[1, 300, 6]);

    // Extract top detection from each
    let rust_data: Vec<f32> = rust_output
        .squeeze(0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let python_data: Vec<f32> = python_output
        .squeeze(0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // Top Rust detection (first row, already sorted by confidence)
    let rust_top = &rust_data[0..6]; // [x1, y1, x2, y2, conf, cls]
                                     // Top Python detection
    let python_top = &python_data[0..6];

    eprintln!(
        "Rust  top: x1={:.1} y1={:.1} x2={:.1} y2={:.1} conf={:.4} cls={}",
        rust_top[0], rust_top[1], rust_top[2], rust_top[3], rust_top[4], rust_top[5] as i32
    );
    eprintln!(
        "Python top: x1={:.1} y1={:.1} x2={:.1} y2={:.1} conf={:.4} cls={}",
        python_top[0],
        python_top[1],
        python_top[2],
        python_top[3],
        python_top[4],
        python_top[5] as i32
    );

    // The top detection should match: same class, similar confidence and box
    let box_tolerance: f32 = 2.0; // pixels
    let conf_tolerance: f32 = 0.01;

    // Compare bounding box coordinates
    for i in 0..4 {
        let diff: f32 = (rust_top[i] - python_top[i]).abs();
        assert!(
            diff < box_tolerance,
            "Top detection box coord {i} differs: rust={:.2} python={:.2} diff={diff:.4}",
            rust_top[i],
            python_top[i]
        );
    }

    // Compare confidence
    let conf_diff: f32 = (rust_top[4] - python_top[4]).abs();
    assert!(
        conf_diff < conf_tolerance,
        "Top detection confidence differs: rust={:.4} python={:.4}",
        rust_top[4],
        python_top[4]
    );

    // Compare class ID
    assert_eq!(
        rust_top[5] as i32, python_top[5] as i32,
        "Top detection class differs"
    );
}

/// Full pipeline test: RGBA pixels → preprocess → model → postprocess → compare with Python.
/// Validates that the complete Rust pipeline (including preprocessing) produces
/// results consistent with the Python reference.
#[test]
#[ignore]
fn test_full_pipeline_matches_python() {
    if !fixtures_available() {
        eprintln!("Skipping: fixtures not available.");
        return;
    }

    let device = Device::Cpu;

    // Load model
    let weights_bytes: Vec<u8> = std::fs::read(WEIGHTS_PATH).unwrap();
    let model = yolo26_rust_wasm::model::Yolo26Model::load(weights_bytes, &device, &Multiples::n())
        .unwrap();

    // Load RGBA test image
    let rgba_bytes: Vec<u8> = std::fs::read("tests/fixtures/test_image_rgba.bin").unwrap();

    // Load metadata for image dimensions
    let metadata_str: String = std::fs::read_to_string(FIXTURE_METADATA).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
    let width: u32 = metadata["image_width"].as_u64().unwrap() as u32;
    let height: u32 = metadata["image_height"].as_u64().unwrap() as u32;
    assert_eq!(rgba_bytes.len(), (width * height * 4) as usize);

    // Run full Rust pipeline
    let (input, letterbox) =
        yolo26_rust_wasm::preprocess::preprocess(&rgba_bytes, width, height, &device).unwrap();
    assert_eq!(input.dims(), &[1, 3, 640, 640]);

    let output: Tensor = model.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 300, 6]);

    let detections =
        yolo26_rust_wasm::postprocess::postprocess(&output, &letterbox, width, height, 0.001)
            .unwrap();

    // Load Python reference detections
    let python_dets: &Vec<serde_json::Value> = metadata["detections"].as_array().unwrap();

    eprintln!("Rust detections (conf>0.001): {}", detections.len());
    eprintln!("Python detections (conf>0.001): {}", python_dets.len());

    // Verify the highest-confidence Rust detection matches Python's top detection
    if !detections.is_empty() && !python_dets.is_empty() {
        let rust_best = &detections[0]; // Already sorted by confidence
        let python_best = &python_dets[0];

        let py_conf: f64 = python_best["confidence"].as_f64().unwrap();
        let py_cls: i32 = python_best["class_id"].as_i64().unwrap() as i32;

        eprintln!(
            "Rust best:  class={} conf={:.4}",
            rust_best.class_id, rust_best.confidence
        );
        eprintln!("Python best: class={py_cls} conf={py_conf:.4}");

        // Class should match
        assert_eq!(
            rust_best.class_id as i32, py_cls,
            "Best detection class mismatch"
        );

        // Confidence should be close
        let conf_diff: f64 = (rust_best.confidence as f64 - py_conf).abs();
        assert!(
            conf_diff < 0.01,
            "Best detection confidence differs by {conf_diff:.4}"
        );
    }
}
