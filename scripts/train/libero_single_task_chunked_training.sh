#!/bin/bash

set -euo pipefail

export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"$REPO_ROOT/data/libero_goal_single_task_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/checkpoints/dreamzero_libero_single_task_chunked"}
if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-4}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$REPO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$REPO_ROOT/checkpoints/umt5-xxl"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-}
RESET_LIBERO_HEADS=${RESET_LIBERO_HEADS:-false}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-500}
REPORT_TO=${REPORT_TO:-wandb}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
FRAME_SEQLEN=${FRAME_SEQLEN:-512}
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-4}
NUM_FRAME_PER_BLOCK=${NUM_FRAME_PER_BLOCK:-2}
NUM_SHARDS_TO_SAMPLE=${NUM_SHARDS_TO_SAMPLE:-1048576}

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: Converted LIBERO dataset not found at $LIBERO_DATA_ROOT"
    echo "Create it with:"
    echo "  LIBERO_HDF5_PATH=/path/to/task_demo.hdf5 OUTPUT_PATH=$LIBERO_DATA_ROOT bash scripts/data/convert_libero_single_task.sh"
    exit 1
fi

if [ "$RESET_LIBERO_HEADS" = "true" ] && [ -z "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: RESET_LIBERO_HEADS=true requires PRETRAINED_MODEL_PATH to be set"
    exit 1
fi

TORCHRUN_ARGS=(
    --nproc_per_node "$NUM_GPUS"
    --standalone
    groot/vla/experiment/experiment.py
    "report_to=$REPORT_TO"
    data=dreamzero/libero_spatial_chunked
    wandb_project=dreamzero
    train_architecture=lora
    num_frames=33
    action_horizon=24
    num_views=2
    model=dreamzero/vla
    model/dreamzero/action_head=wan_flow_matching_action_tf
    model/dreamzero/transform=dreamzero_cotrain
    "num_frame_per_block=$NUM_FRAME_PER_BLOCK"
    num_action_per_block=24
    num_state_per_block=1
    seed=42
    training_args.learning_rate=1e-5
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json"
    "save_steps=$SAVE_STEPS"
    training_args.warmup_ratio=0.05
    "output_dir=$OUTPUT_DIR"
    per_device_train_batch_size=1
    gradient_accumulation_steps=4
    "max_steps=$MAX_STEPS"
    weight_decay=1e-5
    save_total_limit=10
    upload_checkpoints=false
    bf16=true
    tf32=true
    eval_bf16=true
    dataloader_pin_memory=false
    dataloader_num_workers=1
    image_resolution_width_single_frame=256
    image_resolution_height_single_frame=256
    save_lora_only=true
    "max_chunk_size=$MAX_CHUNK_SIZE"
    "num_shards_to_sample=$NUM_SHARDS_TO_SAMPLE"
    "frame_seqlen=$FRAME_SEQLEN"
    save_strategy=steps
    "libero_data_root=$LIBERO_DATA_ROOT"
    "dit_version=$WAN_CKPT_DIR"
    "text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    "image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth"
    "tokenizer_path=$TOKENIZER_DIR"
)

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    TORCHRUN_ARGS+=("resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

if [ -n "$PRETRAINED_MODEL_PATH" ]; then
    TORCHRUN_ARGS+=(
        "pretrained_model_path=$PRETRAINED_MODEL_PATH"
        "++action_head_cfg.config.skip_component_loading=true"
        "++action_head_cfg.config.defer_lora_injection=true"
    )
fi

if [ "$RESET_LIBERO_HEADS" = "true" ]; then
    TORCHRUN_ARGS+=("reset_libero_heads=true")
fi

cd "$REPO_ROOT"
torchrun "${TORCHRUN_ARGS[@]}"
