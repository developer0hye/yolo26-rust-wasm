"""Export YOLO26 weights for all model sizes (n/s/m/l/x) as FP16 SafeTensors.

FP16 halves file size with zero inference accuracy loss — candle auto-converts
to FP32 at load time (absorb_bn fuses Conv+BN after the cast).

Usage:
    pip install ultralytics safetensors
    python scripts/export_all_sizes.py

Outputs yolo26{n,s,m,l,x}.safetensors to the current directory.
"""

from ultralytics import YOLO
from safetensors.torch import save_file

SIZES = ["n", "s", "m", "l", "x"]

for size in SIZES:
    model_name = f"yolo26{size}.pt"
    out_name = f"yolo26{size}.safetensors"
    print(f"Exporting {model_name} -> {out_name}")
    model = YOLO(model_name)
    sd = model.model.state_dict()
    sd_fp16 = {k: v.half() for k, v in sd.items()}
    save_file(sd_fp16, out_name)
    print(f"  Saved {len(sd_fp16)} tensors (FP16)")

print("Done. Upload all .safetensors files to HuggingFace Hub.")
