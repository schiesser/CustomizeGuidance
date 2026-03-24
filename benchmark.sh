#!/bin/bash

#model
MODEL="SD3"

#path
MODEL_PATH="/scratch/cvlab/home/schiesser/models/sd35_medium"
DATA_IMAGES_PATH="/scratch/cvlab/home/schiesser/datasets/MS_COCO/annotations/val2017"
DATA_ANNOTATIONS_PATH="/scratch/cvlab/home/schiesser/datasets/MS_COCO/annotations/caption_val2017.json"

CLIP_MODEL_PATH="/scratch/cvlab/home/schiesser/models/clip"
BLIP_MODEL_PATH="/scratch/cvlab/home/schiesser/models/blip"

# images size / number
NUM_IMAGES=10
HEIGHT=256
WIDTH=256

# steps and guidance scale
NUM_STEPS=28
GUIDANCE_SCALE=7.0
GUIDANCE_TYPE="constant"

# score 
SCORES=("FID" "CLIP" "IS" "BLIP")

# reproductibility
RUN_ID="test_run"
SEED=13

python -m scripts/benchmark.py \
    --model "$MODEL" \
    --guidance_types "$GUIDANCE_TYPE" \
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