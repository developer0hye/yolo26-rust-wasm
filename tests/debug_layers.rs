//! Debug test: compare Rust intermediate layer outputs with Python.
//! Run with: cargo test --test debug_layers -- --ignored --nocapture

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

fn load_vb(device: &Device) -> VarBuilder<'static> {
    let weights: Vec<u8> = std::fs::read("weights/yolo26n.safetensors").unwrap();
    candle_core::DType::F32;
    VarBuilder::from_buffered_safetensors(weights, candle_core::DType::F32, device).unwrap()
}

fn load_input(device: &Device) -> Tensor {
    let bytes: Vec<u8> = std::fs::read("tests/fixtures/test_input.safetensors").unwrap();
    let tensors = candle_core::safetensors::load_buffer(&bytes, device).unwrap();
    tensors.get("input").unwrap().clone()
}

fn first_n(t: &Tensor, n: usize) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[..n].to_vec()
}

fn print_first(label: &str, t: &Tensor, n: usize) {
    let vals = first_n(t, n);
    let s: Vec<String> = vals.iter().map(|v| format!("{v:.6}")).collect();
    eprintln!("{label}: shape={:?}, [{s}]", t.dims(), s = s.join(", "));
}

#[test]
#[ignore]
fn test_layer0_conv() {
    let device = Device::Cpu;
    let vb = load_vb(&device);
    let input = load_input(&device);

    let layer0 = yolo26_rust_wasm::model::blocks::ConvBlock::load(
        vb.pp("model").pp("0"),
        3,
        16,
        3,
        2,
        1,
        true,
    )
    .unwrap();
    let out = layer0.forward(&input).unwrap();
    print_first("Rust layer 0", &out, 5);

    let expected = [16.585279_f32, 1.210977, 1.210977, 1.210977, 1.210977];
    let vals = first_n(&out, 5);
    for (i, (got, exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "Layer 0 val {i}: got={got}, exp={exp}"
        );
    }
    eprintln!("Layer 0: PASS");
}

/// Test layers 7-10 (where divergence starts) step by step
#[test]
#[ignore]
fn test_layers_7_to_10() {
    use yolo26_rust_wasm::model::blocks::{C2psa, C3k2, ConvBlock, Sppf};

    let device = Device::Cpu;
    let vb = load_vb(&device);
    let input = load_input(&device);

    // Run through layers 0-6 to get p4 (which we know is correct)
    let l0 = ConvBlock::load(vb.pp("model").pp("0"), 3, 16, 3, 2, 1, true).unwrap();
    let l1 = ConvBlock::load(vb.pp("model").pp("1"), 16, 32, 3, 2, 1, true).unwrap();
    let l2 = C3k2::load(vb.pp("model").pp("2"), 32, 64, 1, false, 0.25, true, false).unwrap();
    let l3 = ConvBlock::load(vb.pp("model").pp("3"), 64, 64, 3, 2, 1, true).unwrap();
    let l4 = C3k2::load(vb.pp("model").pp("4"), 64, 128, 1, false, 0.25, true, false).unwrap();
    let l5 = ConvBlock::load(vb.pp("model").pp("5"), 128, 128, 3, 2, 1, true).unwrap();
    let l6 = C3k2::load(vb.pp("model").pp("6"), 128, 128, 1, true, 0.5, true, false).unwrap();
    let l7 = ConvBlock::load(vb.pp("model").pp("7"), 128, 256, 3, 2, 1, true).unwrap();
    let l8 = C3k2::load(vb.pp("model").pp("8"), 256, 256, 1, true, 0.5, true, false).unwrap();
    let l9 = Sppf::load(vb.pp("model").pp("9"), 256, 256, 5, 3, true).unwrap();
    let l10 = C2psa::load(vb.pp("model").pp("10"), 256, 256, 1).unwrap();

    let x = l0.forward(&input).unwrap();
    let x = l1.forward(&x).unwrap();
    let x = l2.forward(&x).unwrap();
    let x = l3.forward(&x).unwrap();
    let x = l4.forward(&x).unwrap();
    let x = l5.forward(&x).unwrap();
    let x = l6.forward(&x).unwrap();

    // Layer 7: Conv(128→256, k=3, s=2)
    let x = l7.forward(&x).unwrap();
    print_first("Layer 7 (Conv)", &x, 5);
    // Python: [-0.246528, -0.257104, -0.207448, -0.219650, -0.234363]

    // Layer 8: C3k2(256→256, c3k=T, e=0.5)
    let x = l8.forward(&x).unwrap();
    print_first("Layer 8 (C3k2)", &x, 5);
    // Python: [-0.150028, -0.119362, 0.070899, 0.247476, 0.025443]

    // Layer 9: SPPF(256→256, k=5)
    let x = l9.forward(&x).unwrap();
    print_first("Layer 9 (SPPF)", &x, 5);
    // Python: [-0.209128, -0.185892, 0.009064, 0.132908, -0.104251]

    // Layer 10: C2PSA(256→256, n=1)
    let x = l10.forward(&x).unwrap();
    print_first("Layer 10 (C2PSA)", &x, 5);
    // Python: [-0.245049, -0.248769, -0.246283, -0.247148, -0.244466]
}

/// Test backbone output shapes and values
#[test]
#[ignore]
fn test_backbone_output() {
    let device = Device::Cpu;
    let vb = load_vb(&device);
    let input = load_input(&device);

    let backbone = yolo26_rust_wasm::model::backbone::Backbone::load(vb.pp("model")).unwrap();
    let bb = backbone.forward(&input).unwrap();
    print_first("Rust p3 (layer 4)", &bb.p3, 5);
    print_first("Rust p4 (layer 6)", &bb.p4, 5);
    print_first("Rust p5 (layer 10)", &bb.p5, 5);
}
