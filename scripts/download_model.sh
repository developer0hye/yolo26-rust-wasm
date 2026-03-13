#!/bin/bash
set -euo pipefail

mkdir -p weights

python3 -c "
from ultralytics import YOLO
from safetensors.torch import save_file

model = YOLO('yolo26n.pt')
sd = model.model.state_dict()

# Convert all tensors to float32 for candle compatibility
sd_f32 = {k: v.float() if v.is_floating_point() else v.float() for k, v in sd.items()}
save_file(sd_f32, 'weights/yolo26n.safetensors')

print(f'Saved {len(sd_f32)} tensors to weights/yolo26n.safetensors')
for k, v in sorted(sd_f32.items())[:20]:
    print(f'  {k}: {list(v.shape)}')
if len(sd_f32) > 20:
    print(f'  ... ({len(sd_f32) - 20} more)')
"

echo "Done: weights/yolo26n.safetensors"
