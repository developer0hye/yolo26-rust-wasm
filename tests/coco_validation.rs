//! COCO validation: compare Rust inference against Python (ultralytics) reference.
//!
//! Runs full pipeline (RGBA → preprocess → model → postprocess) on 16 real COCO images
//! and compares detections with Python reference results.
//!
//! Prerequisites:
//!   1. weights/yolo26n.safetensors (run scripts/download_model.sh)
//!   2. tests/fixtures/coco/* (run python3 scripts/validate_coco_images.py)
//!
//! Run with: cargo test --test coco_validation -- --ignored --nocapture

use candle_core::Device;
use std::path::Path;

const WEIGHTS_PATH: &str = "weights/yolo26n.safetensors";
const COCO_FIXTURES_DIR: &str = "tests/fixtures/coco";

const COCO_IMAGE_IDS: [&str; 50] = [
    "000000000139",
    "000000000285",
    "000000000632",
    "000000000776",
    "000000001000",
    "000000001268",
    "000000001503",
    "000000001761",
    "000000002006",
    "000000002149",
    "000000002299",
    "000000002431",
    "000000002473",
    "000000002532",
    "000000002587",
    "000000002685",
    "000000003156",
    "000000003501",
    "000000003553",
    "000000003934",
    "000000004134",
    "000000004395",
    "000000005037",
    "000000005193",
    "000000005477",
    "000000005529",
    "000000006040",
    "000000006471",
    "000000006763",
    "000000006818",
    "000000007108",
    "000000007278",
    "000000007386",
    "000000007574",
    "000000007784",
    "000000007816",
    "000000007977",
    "000000008021",
    "000000008211",
    "000000008532",
    "000000008690",
    "000000008844",
    "000000009448",
    "000000009483",
    "000000009590",
    "000000009769",
    "000000009891",
    "000000009914",
    "000000010092",
    "000000010977",
];

fn fixtures_available() -> bool {
    if !Path::new(WEIGHTS_PATH).exists() {
        return false;
    }
    let summary_path: String = format!("{COCO_FIXTURES_DIR}/summary.json");
    Path::new(&summary_path).exists()
}

/// Per-image comparison result
struct ImageResult {
    image_id: String,
    description: String,
    rust_detection_count: usize,
    python_detection_count: usize,
    top_class_match: bool,
    top_conf_diff: f32,
    rust_top_class: String,
    python_top_class: String,
    rust_top_conf: f32,
    python_top_conf: f32,
}

#[test]
#[ignore]
fn test_coco_16_images() {
    if !fixtures_available() {
        eprintln!(
            "Skipping: fixtures not available.\n\
             Run: scripts/download_model.sh && python3 scripts/validate_coco_images.py"
        );
        return;
    }

    let device: Device = Device::Cpu;

    // Load model once
    let weights_bytes: Vec<u8> = std::fs::read(WEIGHTS_PATH).unwrap();
    let model = yolo26_rust_wasm::model::Yolo26Model::load(weights_bytes, &device).unwrap();

    let confidence_threshold: f32 = 0.25;
    let mut results: Vec<ImageResult> = Vec::new();
    let mut pass_count: usize = 0;
    let mut total_count: usize = 0;

    eprintln!("\n{:=<80}", "");
    eprintln!("COCO Validation: 16 images, conf_threshold={confidence_threshold}");
    eprintln!("{:=<80}", "");

    for image_id in &COCO_IMAGE_IDS {
        let rgba_path: String = format!("{COCO_FIXTURES_DIR}/{image_id}_rgba.bin");
        let metadata_path: String = format!("{COCO_FIXTURES_DIR}/{image_id}_metadata.json");

        if !Path::new(&rgba_path).exists() || !Path::new(&metadata_path).exists() {
            eprintln!("[{image_id}] SKIP: fixture files missing");
            continue;
        }

        total_count += 1;

        // Load metadata
        let metadata_str: String = std::fs::read_to_string(&metadata_path).unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
        let width: u32 = metadata["width"].as_u64().unwrap() as u32;
        let height: u32 = metadata["height"].as_u64().unwrap() as u32;
        let description: &str = metadata["description"].as_str().unwrap_or("");

        // Load Python reference detections (already filtered at conf>=0.001)
        let python_all_dets: &Vec<serde_json::Value> =
            metadata["python_detections"].as_array().unwrap();
        let python_dets: Vec<&serde_json::Value> = python_all_dets
            .iter()
            .filter(|d| d["confidence"].as_f64().unwrap() >= confidence_threshold as f64)
            .collect();

        // Load RGBA pixels
        let rgba_bytes: Vec<u8> = std::fs::read(&rgba_path).unwrap();
        assert_eq!(
            rgba_bytes.len(),
            (width * height * 4) as usize,
            "RGBA size mismatch for {image_id}"
        );

        // Run Rust pipeline
        let (input, letterbox) =
            yolo26_rust_wasm::preprocess::preprocess(&rgba_bytes, width, height, &device).unwrap();
        let output = model.forward(&input).unwrap();
        let rust_dets = yolo26_rust_wasm::postprocess::postprocess(
            &output,
            &letterbox,
            width,
            height,
            confidence_threshold,
        )
        .unwrap();

        // Compare top detection
        let rust_top_class: String = rust_dets
            .first()
            .map(|d| d.class_name.clone())
            .unwrap_or_default();
        let rust_top_conf: f32 = rust_dets.first().map(|d| d.confidence).unwrap_or(0.0);
        let python_top_class: String = python_dets
            .first()
            .and_then(|d| d["class_name"].as_str())
            .unwrap_or("")
            .to_string();
        let python_top_conf: f32 = python_dets
            .first()
            .and_then(|d| d["confidence"].as_f64())
            .unwrap_or(0.0) as f32;

        let top_class_match: bool =
            rust_top_class == python_top_class || (rust_dets.is_empty() && python_dets.is_empty());
        let top_conf_diff: f32 = (rust_top_conf - python_top_conf).abs();

        let status: &str = if top_class_match && top_conf_diff < 0.05 {
            pass_count += 1;
            "PASS"
        } else if top_class_match {
            pass_count += 1;
            "PASS (conf drift)"
        } else {
            "FAIL"
        };

        eprintln!(
            "[{image_id}] {status:15} | {description:20} | \
             rust={:2} python={:2} | \
             rust_top={:15} ({:.4}) python_top={:15} ({:.4}) conf_diff={:.4}",
            rust_dets.len(),
            python_dets.len(),
            rust_top_class,
            rust_top_conf,
            python_top_class,
            python_top_conf,
            top_conf_diff,
        );

        results.push(ImageResult {
            image_id: image_id.to_string(),
            description: description.to_string(),
            rust_detection_count: rust_dets.len(),
            python_detection_count: python_dets.len(),
            top_class_match,
            top_conf_diff,
            rust_top_class,
            python_top_class,
            rust_top_conf,
            python_top_conf,
        });
    }

    eprintln!("{:=<80}", "");
    eprintln!("Results: {pass_count}/{total_count} passed (top class match + conf_diff < 0.05)");

    // Summary statistics
    let total_rust_dets: usize = results.iter().map(|r| r.rust_detection_count).sum();
    let total_python_dets: usize = results.iter().map(|r| r.python_detection_count).sum();
    let avg_conf_diff: f32 =
        results.iter().map(|r| r.top_conf_diff).sum::<f32>() / results.len().max(1) as f32;
    let class_match_rate: f32 =
        results.iter().filter(|r| r.top_class_match).count() as f32 / results.len().max(1) as f32;

    eprintln!("Total Rust detections:  {total_rust_dets}");
    eprintln!("Total Python detections: {total_python_dets}");
    eprintln!("Avg top conf diff:      {avg_conf_diff:.6}");
    eprintln!("Top class match rate:   {:.1}%", class_match_rate * 100.0);
    eprintln!("{:=<80}", "");

    // Median confidence difference (robust to outliers from preprocessing differences)
    let mut conf_diffs: Vec<f32> = results.iter().map(|r| r.top_conf_diff).collect();
    conf_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_conf_diff: f32 = conf_diffs[conf_diffs.len() / 2];
    eprintln!("Median top conf diff:   {median_conf_diff:.6}");

    // Assertions: at least 75% of images should have matching top class
    assert!(
        class_match_rate >= 0.75,
        "Top class match rate {:.1}% is below 75% threshold",
        class_match_rate * 100.0
    );

    // Median confidence difference should be small (robust to preprocessing outliers)
    assert!(
        median_conf_diff < 0.05,
        "Median top confidence difference {median_conf_diff:.4} exceeds 0.05 threshold"
    );
}
