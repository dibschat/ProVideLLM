#!/bin/bash

# set your environment variables here
ROOT_PATH="../../"
DATASET_DIR="../../datasets"

N_GPUS=8
BATCH_SIZE_PER_GPU=16

cd $ROOT_PATH

EFFECTIVE_BATCH_SIZE=128
EFFECTIVE_BATCH_SIZE_PER_GPU=$((EFFECTIVE_BATCH_SIZE / N_GPUS))
GRAD_STEPS=$((EFFECTIVE_BATCH_SIZE_PER_GPU / BATCH_SIZE_PER_GPU))

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=$N_GPUS --standalone train.py \
    --deepspeed configs/deepspeed/zero1.json \
    --model_variant providellm_8b \
    --train_datasets coin_step_train coin_task_train coin_next_train coin_procedure_train coin_taskprocedure_train --eval_datasets coin_step_test coin_task_test coin_next_test coin_procedure_test coin_taskprocedure_test \
    --dataset_dir $DATASET_DIR \
    --num_train_epochs 20 --per_device_train_batch_size $BATCH_SIZE_PER_GPU --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_STEPS --gradient_checkpointing False \
    --eval_strategy no --prediction_loss_only False \
    --save_strategy no --save_steps 1000 \
    --learning_rate 1.5e-4 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --logging_steps 10 --dataloader_num_workers 6 \
    --bf16 False --tf32 False --fp16 True \
    --report_to wandb \
    --fine_tune "lora" --lora_r 128 \
    --num_samples 16