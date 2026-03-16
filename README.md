# DreamZero on LIBERO

This repository contains my reproduction and adaptation of **DreamZero** on the **LIBERO** benchmark.

The current goal of this codebase is to connect the DreamZero training and inference pipeline to LIBERO data and the official LIBERO benchmark, including:

- converting raw LIBERO `hdf5` demonstrations into the format required by DreamZero
- training DreamZero on `libero_spatial`
- evaluating DreamZero with the official LIBERO closed-loop rollout pipeline
- continuing training from an existing DreamZero checkpoint with LoRA / continued training on LIBERO

This repository is intended for:

- reproducing DreamZero training on LIBERO
- studying DreamZero's joint video-action modeling behavior in simulation tasks
- serving as a starting point for extending to other LIBERO suites or other embodied benchmarks

## Current Status

The main working pipeline is:

```text
LIBERO raw hdf5
  -> dreamzero/scripts/data/convert_libero.py
  -> LeRobot-style dataset + GEAR metadata
  -> dreamzero/scripts/train/libero_spatial_training.sh
  -> DreamZero training
  -> dreamzero/eval_utils/run_libero_server.py
  -> dreamzero/eval_utils/run_libero_eval.py
  -> official LIBERO closed-loop rollout
```

The default setup currently corresponds to:

- benchmark: `libero_spatial`
- observation:
  - `agentview_rgb`
  - `eye_in_hand_rgb`
  - `joint_position`
  - `gripper_position`
  - `language_instruction`
- raw `state_dim = 9`
- raw `action_dim = 7`
- action space: `osc_pose` / ee-space delta
- control frequency: `20 Hz`
- training window:
  - `num_frames = 25`
  - `action_horizon = 24`

## Repository Layout

```text
.
|-- dreamzero/
|   |-- scripts/data/convert_libero.py
|   |-- scripts/train/libero_spatial_training.sh
|   |-- eval_utils/run_libero_server.py
|   |-- eval_utils/run_libero_eval.py
|   |-- groot/vla/...
|-- LIBERO/
|   |-- official LIBERO codebase
|-- Dreamzero2libero.md
|-- Dreamzero2libero_experience.md
|-- Dreamzero2libero_training_reference.md
```

Notes:

- `dreamzero/` contains the main training, inference, and evaluation bridge code.
- `LIBERO/` is the local official LIBERO codebase used together with this repo.
- The `Dreamzero2libero*.md` files in the repository root are engineering notes from the integration and reproduction process.

## Environment Setup

This workflow usually involves at least two kinds of environments:

- a DreamZero training / policy server environment
- a LIBERO evaluation environment with `libero` installed correctly

If you only want to train and export checkpoints, the `dreamzero/` environment is the main one.
If you want to run official LIBERO rollouts, it is recommended to prepare a separate environment that can import `libero` cleanly.

For the base installation, refer to:

- [dreamzero/README.md](./dreamzero/README.md)
- [dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md](./dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md)

## Data Preparation

### 1. Prepare the raw LIBERO dataset

You first need the official LIBERO demonstrations. The current workflow expects raw `*_demo.hdf5` files.

### 2. Convert the dataset into the DreamZero training format

Run the following under `dreamzero/`:

```bash
cd dreamzero

python scripts/data/convert_libero.py \
  --suite-path <path-to-libero-suite> \
  --output-path ./data/libero_spatial_lerobot
```

For the current main setup, this is typically:

```bash
python scripts/data/convert_libero.py \
  --suite-path <LIBERO-dataset-root>/libero_spatial \
  --output-path ./data/libero_spatial_lerobot
```

This step does two things:

- converts raw LIBERO `hdf5` files into a LeRobot-style dataset
- generates the metadata required by DreamZero/GEAR training, such as `meta/modality.json` and `meta/stats.json`

The converted dataset should look like:

```text
dreamzero/data/libero_spatial_lerobot/
|-- data/
|-- videos/
|-- meta/
|   |-- info.json
|   |-- modality.json
|   |-- episodes.jsonl
|   |-- tasks.jsonl
|   |-- stats.json
```

## Training

The current training entry script is:

- [dreamzero/scripts/train/libero_spatial_training.sh](./dreamzero/scripts/train/libero_spatial_training.sh)

Run:

```bash
cd dreamzero

export LIBERO_DATA_ROOT=./data/libero_spatial_lerobot
export OUTPUT_DIR=./checkpoints/dreamzero_libero_spatial
export NUM_GPUS=8

bash scripts/train/libero_spatial_training.sh
```

By default, the script will automatically check for or download:

- `Wan2.1-I2V-14B-480P`
- `umt5-xxl`

Key training defaults currently include:

- `num_frames=25`
- `action_horizon=24`
- `num_views=2`
- `learning_rate=1e-5`
- `warmup_ratio=0.05`
- `weight_decay=1e-5`
- `bf16=true`
- `train_architecture=lora`

If you want to continue training from an existing DreamZero checkpoint, you can additionally set:

```bash
export PRETRAINED_MODEL_PATH=<path-to-checkpoint>
export RESET_LIBERO_HEADS=true
```

## Evaluation

The current evaluation setup uses a DreamZero websocket policy server connected to the official LIBERO environment.

### 1. Start the DreamZero policy server

Under `dreamzero/`:

```bash
cd dreamzero

python eval_utils/run_libero_server.py \
  --model-path <path-to-checkpoint> \
  --metadata-dataset-path ./data/libero_spatial_lerobot \
  --port 8000
```

### 2. Run official LIBERO rollout evaluation

Then, in a working LIBERO environment:

```bash
cd dreamzero

python eval_utils/run_libero_eval.py \
  --libero-root ../LIBERO \
  --host localhost \
  --port 8000 \
  --benchmark-name libero_spatial \
  --checkpoint-path <path-to-checkpoint> \
  --output-dir ./runs/libero_eval
```

Evaluation results are saved as:

- `results.json`
- `results.csv`
- optional rollout videos

### Evaluation Protocol

The current evaluation pipeline is:

- closed-loop rollout at the episode level
- but action chunks are executed in a segmented open-loop manner

In practice, the model predicts an action chunk at once, and the client executes several actions before requesting the next inference call. This is the actual DreamZero-on-LIBERO evaluation behavior in this repository.

## Important Files

If you want to read the code efficiently, these are the best entry points:

- [dreamzero/scripts/data/convert_libero.py](./dreamzero/scripts/data/convert_libero.py)
- [dreamzero/scripts/train/libero_spatial_training.sh](./dreamzero/scripts/train/libero_spatial_training.sh)
- [dreamzero/groot/vla/configs/data/dreamzero/libero_spatial.yaml](./dreamzero/groot/vla/configs/data/dreamzero/libero_spatial.yaml)
- [dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py](./dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py)
- [dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py](./dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py)
- [dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py](./dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py)
- [dreamzero/eval_utils/run_libero_server.py](./dreamzero/eval_utils/run_libero_server.py)
- [dreamzero/eval_utils/run_libero_eval.py](./dreamzero/eval_utils/run_libero_eval.py)

## Additional Notes

For more detailed engineering notes and design references, see:

- [Dreamzero2libero.md](./Dreamzero2libero.md)
- [Dreamzero2libero_experience.md](./Dreamzero2libero_experience.md)
- [Dreamzero2libero_training_reference.md](./Dreamzero2libero_training_reference.md)
- [Dreamzero2libero_continued_training_plan.md](./Dreamzero2libero_continued_training_plan.md)

## Known Notes

- The current main workflow is focused on `libero_spatial`.
- The data, training, and evaluation scripts are aligned with the current local implementation, but a small smoke test is still recommended before large-scale runs.
- If you upload this repository publicly, it is a good idea to clearly state in the release notes that:
  - this is a reproduction / adaptation project built on top of DreamZero and LIBERO
  - checkpoint usage, dataset usage, and upstream licenses must follow the original projects

## Acknowledgements

This repository is built on top of:

- [DreamZero](https://github.com/dreamzero0/dreamzero)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

If this repository is useful for your work, please also cite the corresponding upstream projects and papers.
