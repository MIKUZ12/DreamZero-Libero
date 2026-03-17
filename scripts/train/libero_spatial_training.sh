#!/bin/bash

export HYDRA_FULL_ERROR=1

LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"./data/libero_spatial_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_libero_spatial_lora_5k"}
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-}
RESET_LIBERO_HEADS=${RESET_LIBERO_HEADS:-false}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_STEPS=${SAVE_STEPS:-500}
REPORT_TO=${REPORT_TO:-none}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

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
    echo "Create it with: python scripts/data/convert_libero.py --suite-path <LIBERO/libero/datasets/libero_spatial> --output-path $LIBERO_DATA_ROOT"
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
    data=dreamzero/libero_spatial \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=25 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    "save_steps=$SAVE_STEPS" \
    training_args.warmup_ratio=0.05 \
    "output_dir=$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=4 \
    "max_steps=$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width_single_frame=256 \
    image_resolution_height_single_frame=256 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=1024 \
    save_strategy=steps \
    "libero_data_root=$LIBERO_DATA_ROOT" \
    "dit_version=$WAN_CKPT_DIR" \
    "text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    "image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
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

torchrun "${TORCHRUN_ARGS[@]}"
