#!/bin/bash

python scripts/inference.py \
    --model "SD3" \
    --guidance_type "constant" \
    --model_path "/scratch/cvlab/home/schiesser/models/sd3_5_medium" \
    --prompt "a beautiful sunset over the mountains" \
    --height 512 \
    --width 512 \
    --num_inference_steps 28 \
    --guidance_scale 7.0 \
    --output "output.png" \
    --plot False