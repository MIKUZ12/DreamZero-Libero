# DreamZero on LIBERO

This repository tracks my current DreamZero adaptation for LIBERO training and evaluation.

The codebase now supports:

- converting LIBERO raw `hdf5` demonstrations into DreamZero-compatible LeRobot datasets
- single-task and suite-level LIBERO training
- chunked LIBERO training aligned with DreamZero's DROID-style temporal sampling
- checkpoint initialization and true resume from an existing checkpoint
- official LIBERO closed-loop rollout evaluation through a DreamZero websocket policy server

## Current Recommended Pipeline

The most actively maintained LIBERO path in the current code is the single-task chunked workflow:

```text
LIBERO single-task demo.hdf5
  -> dreamzero/scripts/data/convert_libero_single_task.sh
  -> data/libero_goal_single_task_lerobot
  -> dreamzero/scripts/train/libero_single_task_chunked_training.sh
  -> checkpoint
  -> dreamzero/scripts/eval/libero_single_task_server.sh
  -> dreamzero/scripts/eval/libero_single_task_client.sh
  -> official LIBERO rollout
```

There are still older or alternative paths in the repo, including:

- `dreamzero/scripts/train/libero_spatial_training.sh`
- `dreamzero/scripts/train/libero_single_task_training.sh`

Those scripts use the older 25-frame LIBERO setup. The chunked single-task script is the path that is currently closest to the DROID-style training logic.

## What Is Current In This Repo

### Training-side LIBERO setup

For the chunked LIBERO single-task path, the important defaults are:

- benchmark family: `libero_spatial` / LIBERO single-task data
- observation views:
  - `video.agentview_rgb`
  - `video.eye_in_hand_rgb`
- state:
  - `state.eef_state`
  - `state.gripper_state`
- action:
  - `action.pose_delta`
  - `action.gripper_position`
- action space: `osc_pose`
- control frequency: `20 Hz`
- chunked temporal setup:
  - `num_frames=33`
  - `action_horizon=24`
  - `num_frame_per_block=2`
  - `max_chunk_size=4`
  - `frame_seqlen=512`

The chunked dataset implementation for LIBERO explicitly reuses the DROID-style sharded chunk sampler while keeping LIBERO's own state/action semantics.

### Evaluation-side LIBERO setup

The current LIBERO evaluation path now follows a DROID-style online protocol:

- the client sends the current observation frame instead of sending a full 33-frame history window
- the server buffers frames and continues autoregressive inference across requests
- by default, the server reuses cache across requests
- the old `history_frames` argument is kept only for compatibility / logging and no longer controls actual policy input construction

In practice:

- first request: server uses 1 frame
- later requests: server accumulates and uses 4-frame chunks
- action chunks are still executed with `open_loop_horizon`

## Repository Layout

```text
.
|-- dreamzero/
|   |-- scripts/data/convert_libero.py
|   |-- scripts/data/convert_libero_single_task.sh
|   |-- scripts/train/libero_single_task_chunked_training.sh
|   |-- scripts/eval/libero_single_task_server.sh
|   |-- scripts/eval/libero_single_task_client.sh
|   |-- eval_utils/run_libero_server.py
|   |-- eval_utils/run_libero_eval.py
|   |-- groot/vla/...
|-- LIBERO/
|   |-- official LIBERO codebase
|-- Dreamzero2libero.md
|-- Dreamzero2libero_experience.md
|-- Dreamzero2libero_training_reference.md
```

## Environment Setup

This workflow usually involves two environments:

- a DreamZero training / policy-server environment
- a LIBERO rollout environment that can import `libero` correctly

Useful references:

- [dreamzero/README.md](./dreamzero/README.md)
- [dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md](./dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md)

## Data Preparation

### Single-task LIBERO conversion

The simplest current data entry point is:

- [dreamzero/scripts/data/convert_libero_single_task.sh](./dreamzero/scripts/data/convert_libero_single_task.sh)

Example:

```bash
cd dreamzero

LIBERO_HDF5_PATH=/path/to/task_demo.hdf5 \
OUTPUT_PATH=./data/libero_goal_single_task_lerobot \
bash scripts/data/convert_libero_single_task.sh
```

This wraps [dreamzero/scripts/data/convert_libero.py](./dreamzero/scripts/data/convert_libero.py) and generates:

- a LeRobot-format dataset
- GEAR metadata such as `meta/modality.json` and `meta/stats.json`

### Suite-level conversion

If you want a suite-style converted dataset rather than a single-task dataset, use:

```bash
cd dreamzero

python scripts/data/convert_libero.py \
  --suite-path <LIBERO-dataset-root>/libero_spatial \
  --output-path ./data/libero_spatial_lerobot
```

## Training

### Recommended current training entry

The recommended current training entry is:

- [dreamzero/scripts/train/libero_single_task_chunked_training.sh](./dreamzero/scripts/train/libero_single_task_chunked_training.sh)

Example:

```bash
cd dreamzero

export LIBERO_DATA_ROOT=./data/libero_goal_single_task_lerobot
export OUTPUT_DIR=./checkpoints/dreamzero_libero_single_task_chunked
export NUM_GPUS=4

bash scripts/train/libero_single_task_chunked_training.sh
```

Key defaults in that script include:

- `num_frames=33`
- `action_horizon=24`
- `num_views=2`
- `num_frame_per_block=2`
- `max_chunk_size=4`
- `frame_seqlen=512`
- `learning_rate=1e-5`
- `warmup_ratio=0.05`
- `weight_decay=1e-5`
- `bf16=true`
- `train_architecture=lora`

### Continuing training from an existing checkpoint

For initialization from an existing checkpoint:

```bash
export PRETRAINED_MODEL_PATH=<path-to-checkpoint>
export RESET_LIBERO_HEADS=false
```

If the checkpoint comes from a non-LIBERO setup and you intentionally want fresh LIBERO heads, set:

```bash
export RESET_LIBERO_HEADS=true
```

### True resume

The repo now supports explicit true resume through:

```bash
export RESUME_FROM_CHECKPOINT=<path-to-checkpoint-dir>
```

Use `RESUME_FROM_CHECKPOINT` for optimizer/scheduler/global-step resume.
Use `PRETRAINED_MODEL_PATH` for weight initialization only.

## Evaluation

### Recommended current evaluation entry

The easiest current evaluation path uses the shell wrappers:

- [dreamzero/scripts/eval/libero_single_task_server.sh](./dreamzero/scripts/eval/libero_single_task_server.sh)
- [dreamzero/scripts/eval/libero_single_task_client.sh](./dreamzero/scripts/eval/libero_single_task_client.sh)

### 1. Start the policy server

```bash
cd dreamzero

MODEL_PATH=./checkpoints/dreamzero_libero_spatial/checkpoint-10000 \
METADATA_DATASET_PATH=./data/libero_goal_single_task_lerobot \
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

Important notes for the current code:

- do not rely on `HISTORY_FRAMES` to control actual inference input any more
- use a fresh `OUTPUT_DIR` if you want to preserve previous `results.json`, `results.csv`, rollout videos, and `video_pred` clips
- `VIDEO_EPISODES_PER_TASK` controls how many episodes per task will save rollout videos and decoded `video_pred` clips

### Outputs

The evaluation script writes:

- `results.json`
- `results.csv`
- rollout videos under `videos/...`
- decoded `video_pred` clips under `video_pred/...`

## Important Files

Useful code entry points:

- [dreamzero/scripts/data/convert_libero.py](./dreamzero/scripts/data/convert_libero.py)
- [dreamzero/scripts/data/convert_libero_single_task.sh](./dreamzero/scripts/data/convert_libero_single_task.sh)
- [dreamzero/scripts/train/libero_single_task_chunked_training.sh](./dreamzero/scripts/train/libero_single_task_chunked_training.sh)
- [dreamzero/scripts/eval/libero_single_task_server.sh](./dreamzero/scripts/eval/libero_single_task_server.sh)
- [dreamzero/scripts/eval/libero_single_task_client.sh](./dreamzero/scripts/eval/libero_single_task_client.sh)
- [dreamzero/groot/vla/configs/data/dreamzero/libero_spatial_chunked.yaml](./dreamzero/groot/vla/configs/data/dreamzero/libero_spatial_chunked.yaml)
- [dreamzero/groot/vla/data/dataset/lerobot_sharded.py](./dreamzero/groot/vla/data/dataset/lerobot_sharded.py)
- [dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py](./dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py)
- [dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py](./dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py)
- [dreamzero/eval_utils/run_libero_server.py](./dreamzero/eval_utils/run_libero_server.py)
- [dreamzero/eval_utils/run_libero_eval.py](./dreamzero/eval_utils/run_libero_eval.py)

## Notes

- The repo still contains older 25-frame LIBERO scripts. They are useful for reference, but they are not the best representation of the current chunked single-task path.
- The current evaluation path is aligned with the local implementation, but a smoke test is still recommended before large-scale runs.
- If you publish this repository, please clearly state that it is a DreamZero/LIBERO adaptation and that upstream datasets, checkpoints, and licenses remain governed by the original projects.

## Acknowledgements

This repository builds on:

- [DreamZero](https://github.com/dreamzero0/dreamzero)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

