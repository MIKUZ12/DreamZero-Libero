#!/bin/bash

export HYDRA_FULL_ERROR=1

LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"./data/libero_goal_single_task_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_libero_goal_single_task_5b_full"}
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-5B-480P"}
WAN_HF_REPO=${WAN_HF_REPO:-"Wan-AI/Wan2.1-I2V-5B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_STEPS=${SAVE_STEPS:-500}
REPORT_TO=${REPORT_TO:-none}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan checkpoint not found at $WAN_CKPT_DIR. Downloading from HuggingFace repo: $WAN_HF_REPO ..."
    huggingface-cli download "$WAN_HF_REPO" --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: Converted LIBERO dataset not found at $LIBERO_DATA_ROOT"
    echo "Convert with:"
    echo "python scripts/data/convert_libero.py --hdf5-path /local/yangshuo/fyhong/LIBERO/libero/datasets/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5 --output-path $LIBERO_DATA_ROOT"
    exit 1
fi

WAN_CONFIG_JSON="$WAN_CKPT_DIR/config.json"
if [ ! -f "$WAN_CONFIG_JSON" ]; then
    echo "ERROR: Missing Wan config file: $WAN_CONFIG_JSON"
    exit 1
fi

REQUIRED_WAN_FILES=(
    "models_t5_umt5-xxl-enc-bf16.pth"
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "Wan2.1_VAE.pth"
)
for rel_path in "${REQUIRED_WAN_FILES[@]}"; do
    if [ ! -f "$WAN_CKPT_DIR/$rel_path" ]; then
        echo "ERROR: Missing required Wan file: $WAN_CKPT_DIR/$rel_path"
        exit 1
    fi
done

if [ ! -f "$WAN_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json" ] && [ ! -f "$WAN_CKPT_DIR/diffusion_pytorch_model.safetensors" ]; then
    echo "ERROR: Missing diffusion weights under $WAN_CKPT_DIR"
    echo "Expected either diffusion_pytorch_model.safetensors.index.json (sharded) or diffusion_pytorch_model.safetensors (single file)."
    exit 1
fi

WAN_CFG_VALUES=$(python - "$WAN_CONFIG_JSON" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
keys = ["dim", "num_layers", "num_heads", "ffn_dim", "freq_dim", "in_dim", "out_dim", "model_type"]
print(" ".join(str(cfg[k]) for k in keys))
PY
)

read -r WAN_DIM WAN_NUM_LAYERS WAN_NUM_HEADS WAN_FFN_DIM WAN_FREQ_DIM WAN_IN_DIM WAN_OUT_DIM WAN_MODEL_TYPE <<< "$WAN_CFG_VALUES"
echo "Using Wan diffusion config from $WAN_CONFIG_JSON: dim=$WAN_DIM layers=$WAN_NUM_LAYERS heads=$WAN_NUM_HEADS ffn_dim=$WAN_FFN_DIM model_type=$WAN_MODEL_TYPE"

TORCHRUN_ARGS=(
    --nproc_per_node "$NUM_GPUS"
    --standalone
    groot/vla/experiment/experiment.py
    "report_to=$REPORT_TO"
    data=dreamzero/libero_spatial
    wandb_project=dreamzero
    train_architecture=full
    num_frames=25
    action_horizon=24
    num_views=2
    model=dreamzero/vla
    model/dreamzero/action_head=wan_flow_matching_action_tf
    model/dreamzero/transform=dreamzero_cotrain
    num_frame_per_block=2
    num_action_per_block=24
    num_state_per_block=1
    seed=42
    training_args.learning_rate=1e-5
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json"
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
    save_lora_only=false
    max_chunk_size=4
    frame_seqlen=1024
    save_strategy=steps
    "libero_data_root=$LIBERO_DATA_ROOT"
    "dit_version=$WAN_CKPT_DIR"
    "text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    "image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth"
    "tokenizer_path=$TOKENIZER_DIR"
    "action_head_cfg.config.diffusion_model_cfg.dim=$WAN_DIM"
    "action_head_cfg.config.diffusion_model_cfg.num_layers=$WAN_NUM_LAYERS"
    "action_head_cfg.config.diffusion_model_cfg.num_heads=$WAN_NUM_HEADS"
    "action_head_cfg.config.diffusion_model_cfg.ffn_dim=$WAN_FFN_DIM"
    "action_head_cfg.config.diffusion_model_cfg.freq_dim=$WAN_FREQ_DIM"
    "action_head_cfg.config.diffusion_model_cfg.in_dim=$WAN_IN_DIM"
    "action_head_cfg.config.diffusion_model_cfg.out_dim=$WAN_OUT_DIM"
    "action_head_cfg.config.diffusion_model_cfg.model_type=$WAN_MODEL_TYPE"
)

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    TORCHRUN_ARGS+=("resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

torchrun "${TORCHRUN_ARGS[@]}"
