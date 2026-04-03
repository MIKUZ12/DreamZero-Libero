# DreamZero on LIBERO

This README documents the current LIBERO path in this checkout of DreamZero.

The authoritative reference for the training recipe is:

That checkpoint corresponds to the suite-level `libero_spatial` chunked recipe:

- dataset: `data=dreamzero/libero_spatial_chunked`
- data root: `data/libero_spatial_lerobot_chunked_ee`
- views: `video.agentview_rgb`, `video.eye_in_hand_rgb`
- state: `state.eef_state`, `state.gripper_state`
- action: `action.pose_delta`, `action.gripper_position`
- `num_frames=33`
- `action_horizon=24`
- `num_views=2`
- `num_frame_per_block=2`
- `num_action_per_block=24`
- `num_state_per_block=1`
- `max_chunk_size=4`
- `frame_seqlen=512`
- `train_architecture=lora`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `learning_rate=1e-4`
- `weight_decay=1e-5`
- `warmup_ratio=0.05`
- `bf16=true`
- `tf32=true`
- `save_lora_only=true`
- `dataset_shard_sampling_rate=0.1`
- `num_shards_to_sample=1048576`
- `max_steps=10000`
- `save_steps=500`

## Current Pipeline

```text
LIBERO suite demo.hdf5 files
  -> scripts/data/convert_libero.py
  -> data/libero_spatial_lerobot_chunked_ee
  -> scripts/train/libero_training.sh
  -> checkpoints/dreamzero_libero_spatial/checkpoint-*
  -> scripts/eval/libero_single_task_server.sh
  -> scripts/eval/libero_single_task_client.sh
  -> official LIBERO rollout
```

## Data Preparation

Convert the full `libero_spatial` suite into the dataset layout used by `checkpoint-10000`:

```bash
cd dreamzero

python scripts/data/convert_libero.py \
  --suite-path <LIBERO-root>/libero/libero/datasets/libero_spatial \
  --output-path ./data/libero_spatial_lerobot_chunked_ee
```

This generates:

- LeRobot-style episode parquet files
- encoded videos under `videos/`
- DreamZero / GEAR metadata under `meta/`

Important note:

- `checkpoint-10000` was trained from the suite-level dataset `data/libero_spatial_lerobot_chunked_ee`
- it was not trained from the older single-task default path

## Training

The maintained shell entry point in this repo is:

- [`scripts/train/libero_training.sh`](/local/yangshuo/fyhong/dreamzero/scripts/train/libero_training.sh)

However, if you want to match `checkpoint-10000` literally, use the command below as the reference recipe. This is the safest option because the shell wrapper may drift from the saved checkpoint config over time.

### Exact `checkpoint-10000` recipe

```bash
cd dreamzero

export LIBERO_DATA_ROOT=./data/libero_spatial_lerobot_chunked_ee
export OUTPUT_DIR=./checkpoints/dreamzero_libero_spatial
export WAN_CKPT_DIR=./checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=./checkpoints/umt5-xxl
export NUM_GPUS=8

torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
  report_to=wandb \
  data=dreamzero/libero_spatial_chunked \
  wandb_project=dreamzero \
  train_architecture=lora \
  num_frames=33 \
  action_horizon=24 \
  num_views=2 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate=1e-4 \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json \
  save_steps=500 \
  training_args.warmup_ratio=0.05 \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=4 \
  max_steps=10000 \
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
  dataset_shard_sampling_rate=0.1 \
  num_shards_to_sample=1048576 \
  frame_seqlen=512 \
  save_strategy=steps \
  libero_data_root="$LIBERO_DATA_ROOT" \
  dit_version="$WAN_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR"
```

### About `scripts/train/libero_training.sh`

`scripts/train/libero_training.sh` is still useful as a wrapper, but treat the checkpoint `conf.yaml` as the source of truth.

If you use the wrapper, make sure at minimum that these match the checkpoint recipe:

- `LIBERO_DATA_ROOT=./data/libero_spatial_lerobot_chunked_ee`
- `OUTPUT_DIR=./checkpoints/dreamzero_libero_spatial`
- `FRAME_SEQLEN=512`
- `MAX_CHUNK_SIZE=4`
- `NUM_FRAME_PER_BLOCK=2`

If you need a bit-for-bit recipe match, prefer the explicit `torchrun` command above.

### Continuing Training

For weight initialization from an existing checkpoint:

```bash
export PRETRAINED_MODEL_PATH=<path-to-checkpoint>
export RESET_LIBERO_HEADS=false
```

For true optimizer / scheduler / step resume:

```bash
export RESUME_FROM_CHECKPOINT=<path-to-checkpoint-dir>
```

Use:

- `PRETRAINED_MODEL_PATH` for initialization only
- `RESUME_FROM_CHECKPOINT` for true resume

## Evaluation

The current rollout path uses:

- [`scripts/eval/libero_single_task_server.sh`](/local/yangshuo/fyhong/dreamzero/scripts/eval/libero_single_task_server.sh)
- [`scripts/eval/libero_single_task_client.sh`](/local/yangshuo/fyhong/dreamzero/scripts/eval/libero_single_task_client.sh)

Despite the script names, they are also used for the `libero_spatial` suite.

### 1. Start the policy server

```bash
cd dreamzero

MODEL_PATH=./checkpoints/dreamzero_libero_spatial/checkpoint-10000 \
METADATA_DATASET_PATH=./data/libero_spatial_lerobot_chunked_ee \
bash scripts/eval/libero_single_task_server.sh
```

### 2. Run official LIBERO rollout

```bash
cd dreamzero

CHECKPOINT_PATH=./checkpoints/dreamzero_libero_spatial/checkpoint-10000 \
LIBERO_ROOT=../LIBERO \
SERVER_HOST=127.0.0.1 \
SERVER_PORT=8000 \
BENCHMARK_NAME=libero_spatial \
TASK_IDS="2" \
OPEN_LOOP_HORIZON=4 \
N_EVAL=1 \
OUTPUT_DIR=./runs/libero_spatial_task_rollout_run1 \
SAVE_VIDEO=true \
SAVE_VIDEO_PRED=true \
bash scripts/eval/libero_single_task_client.sh
```

Important notes:

- the current eval path uses DROID-style server-side frame buffering
- `HISTORY_FRAMES` is kept only for compatibility / logging and no longer controls the real client input window
- use a fresh `OUTPUT_DIR` if you want to preserve previous `results.json`, `results.csv`, rollout videos, and `video_pred` clips

## Teacher-forced Diagnostics

For offline teacher-forced checks against a raw LIBERO demo file:

```bash
cd dreamzero

MODEL_PATH=./checkpoints/dreamzero_libero_spatial/checkpoint-10000 \
DEMO_FILE=<LIBERO-demo-file>.hdf5 \
METADATA_DATASET_PATH=./data/libero_spatial_lerobot_chunked_ee \
bash scripts/eval/libero_teacher_forced_open_loop.sh
```

Here:

- `DEMO_FILE` is the raw LIBERO `*.hdf5` demo file used for teacher-forced offline comparison
- `METADATA_DATASET_PATH` is the converted DreamZero dataset used for metadata and normalization

## Important Files

- [`scripts/data/convert_libero.py`](/local/yangshuo/fyhong/dreamzero/scripts/data/convert_libero.py)
- [`scripts/train/libero_training.sh`](/local/yangshuo/fyhong/dreamzero/scripts/train/libero_training.sh)
- [`groot/vla/configs/data/dreamzero/libero_spatial_chunked.yaml`](/local/yangshuo/fyhong/dreamzero/groot/vla/configs/data/dreamzero/libero_spatial_chunked.yaml)
- [`groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`](/local/yangshuo/fyhong/dreamzero/groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml)
- [`groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`](/local/yangshuo/fyhong/dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py)
- [`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`](/local/yangshuo/fyhong/dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py)
- [`eval_utils/run_libero_server.py`](/local/yangshuo/fyhong/dreamzero/eval_utils/run_libero_server.py)
- [`eval_utils/run_libero_eval.py`](/local/yangshuo/fyhong/dreamzero/eval_utils/run_libero_eval.py)
- [`checkpoints/dreamzero_libero_spatial/checkpoint-10000/experiment_cfg/conf.yaml`](/local/yangshuo/fyhong/dreamzero/checkpoints/dreamzero_libero_spatial/checkpoint-10000/experiment_cfg/conf.yaml)
