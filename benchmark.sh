#!/bin/bash

#model
MODEL="SD3"

#path
MODEL_PATH="/home/schiesser/models/sd35_medium"
DATA_IMAGES_PATH="/home/schiesser/datasets/MS_COCO/val2017"
DATA_ANNOTATIONS_PATH="/home/schiesser/datasets/MS_COCO/annotations/captions_val2017.json"

CLIP_MODEL_PATH="/home/schiesser/models/clip"
BLIP_MODEL_PATH="/home/schiesser/models/blip"

# images size / number
NUM_IMAGES=20
HEIGHT=256
WIDTH=256

# steps and guidance scale
NUM_STEPS=35
GUIDANCE_SCALE=7.0
GUIDANCE_TYPES=("constant" "linear" "exponential" "APG")
LIST_GUIDANCE_PARAMS=("" "" "" '{"momentum_value": 0.0, "eta": -0.75, "norm_threshold": 15.0}')

# score 
SCORES=("FID" "CLIP" "IS" "BLIP")

# reproductibility
RUN_ID="test_run"
SEED=13

python scripts/benchmark.py \
    --model "$MODEL" \
    --guidance_types "${GUIDANCE_TYPES[@]}" \
    --guidance_parameters "${LIST_GUIDANCE_PARAMS[@]}" \
    --model_path "$MODEL_PATH" \
    --data_annotations_path "$DATA_ANNOTATIONS_PATH" \
    --data_images_path "$DATA_IMAGES_PATH" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --score_list "${SCORES[@]}" \
    --number_of_images $NUM_IMAGES \
    --run_id "$RUN_ID" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --blip_model_path "$BLIP_MODEL_PATH" \
    --seed $SEED 