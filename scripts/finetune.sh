#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

deepspeed --num_gpus=8 llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_path data/attack/DPA/data_llava/poison_data/sst2/badnet/llava_backdoor_dpa_badnet.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --freeze_vision_tower True \
    --freeze_mm_mlp_adapter True \
    --bf16 True \
    --output_dir outputs/sst2/llava-sst2-badnet-full-1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps .2 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to 'none' 
