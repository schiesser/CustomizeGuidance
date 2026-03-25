#!/bin/bash

MODEL="SD3"
MODEL_PATH="/home/schiesser/models/sd35_medium"

PROMPT="a beautiful sunset over the mountains"

HEIGHT=512
WIDTH=512

NUM_STEPS=28
GUIDANCE_SCALE=7.0
GUIDANCE_TYPE="APG"
GUIDANCE_PARAMETERS='{"momentum_value": 0.0, "eta": -0.75, "norm_threshold": 15.0}'

OUTPUT="outputs/output.png"
PLOT=false

python scripts/inference.py \
    --model "$MODEL" \
    --guidance_type "$GUIDANCE_TYPE" \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --guidance_parameters "$GUIDANCE_PARAMETERS" \
    --output "$OUTPUT" \
    --plot $PLOT