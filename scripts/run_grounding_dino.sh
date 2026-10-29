#!/bin/bash
set -e

cd /workspace/GroundingDINO

python demo/inference_on_a_image.py \
  -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p weights/groundingdino_swint_ogc.pth \
  -i /workspace/synthetic_fridge_001.png \
  -o /workspace/main/outputs/grounding_dino/synthetic_fridge \
  -t "egg . tomato . cucumber . carrot ." \
  --box_threshold 0.35 \
  --text_threshold 0.25
