#!/bin/bash

# set your environment variables here
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH="${SCRIPT_DIR}/../.."
DATASET_DIR="${ROOT_PATH}/datasets"

N_GPUS=8
BATCH_SIZE_PER_GPU=16

cd $ROOT_PATH

EFFECTIVE_BATCH_SIZE=256
EFFECTIVE_BATCH_SIZE_PER_GPU=$((EFFECTIVE_BATCH_SIZE / N_GPUS))
GRAD_STEPS=$((EFFECTIVE_BATCH_SIZE_PER_GPU / BATCH_SIZE_PER_GPU))

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=$N_GPUS --standalone train.py \
    --deepspeed configs/deepspeed/zero1.json \
    --model_variant providellm_8b --stage 1 \
    --train_datasets egoclip_stage1 \
    --dataset_dir $DATASET_DIR \
    --num_train_epochs 2 --per_device_train_batch_size $BATCH_SIZE_PER_GPU --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_STEPS --gradient_checkpointing False \
    --eval_strategy no --prediction_loss_only False \
    --save_strategy steps --save_steps 5000 \
    --learning_rate 1e-3 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --logging_steps 10 --dataloader_num_workers 6 \
    --bf16 False --tf32 False --fp16 True \
    --report_to wandb \
    --fine_tune connector \
    --num_samples 4