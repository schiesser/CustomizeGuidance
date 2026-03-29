#!/bin/bash

#model
MODEL="SD3"
#MODEL="flux2Klein"

#path
MODEL_PATH="/home/schiesser/models/flux2_klein_base_4B"
DATA_IMAGES_PATH="/home/schiesser/datasets/MS_COCO/val2017"
DATA_ANNOTATIONS_PATH="/home/schiesser/datasets/MS_COCO/annotations/captions_val2017.json"

CLIP_MODEL_PATH="/home/schiesser/models/clip"
BLIP_MODEL_PATH="/home/schiesser/models/blip"

SAVE_PATH="/home/schiesser"

# images size / number
NUM_IMAGES=10
HEIGHT=1024
WIDTH=1024

# steps and guidance scale
NUM_STEPS=40
GUIDANCE_SCALE=4.5
#GUIDANCE_SCALE=4.0
GUIDANCE_TYPES=("constant" "linear")
LIST_GUIDANCE_PARAMS=("" "" "" '{"momentum_value": 0.3, "eta": -0.75, "norm_threshold": 15.0}' '{"zero_steps": 1, "use_zero_init": 1 '} '{"lambda_max": 1.3 , "gamma": 2.0 }' '{"lambda_param": 1.5 , "k": 0.2}')

# score 
SCORES=("FID" "CLIP" "IS" "BLIP")
KEEP_IMAGES=true

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
    --seed $SEED \
    --keep_images $KEEP_IMAGES \
    --save_path $SAVE_PATH