#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/checkpoints/dreamzero_libero_spatial/checkpoint-10000"}
DEMO_FILE=${DEMO_FILE:-}
METADATA_DATASET_PATH=${METADATA_DATASET_PATH:-"$REPO_ROOT/data/libero_spatial_lerobot"}
LIBERO_ROOT=${LIBERO_ROOT:-"/local/yangshuo/fyhong/LIBERO"}
BENCHMARK_NAME=${BENCHMARK_NAME:-"libero_spatial"}
TASK_ORDER_INDEX=${TASK_ORDER_INDEX:-0}
TASK_ID=${TASK_ID:-}
DEVICE=${DEVICE:-"cuda:0"}
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-1}
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-43200}
HISTORY_FRAMES=${HISTORY_FRAMES:-33}
OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON:-1}
DEMO_IDS=${DEMO_IDS:-"0"}
MAX_OFFLINE_STEPS=${MAX_OFFLINE_STEPS:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/runs/libero_teacher_forced_open_loop"}
SAVE_VIDEO_PRED=${SAVE_VIDEO_PRED:-false}

if [ -z "$DEMO_FILE" ]; then
    echo "ERROR: DEMO_FILE is required."
    echo "Example:"
    echo "  DEMO_FILE=/local/yangshuo/fyhong/LIBERO/libero/datasets/libero_spatial/..."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model checkpoint directory not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DEMO_FILE" ]; then
    echo "ERROR: demo file not found at $DEMO_FILE"
    exit 1
fi

if [ ! -d "$METADATA_DATASET_PATH" ]; then
    echo "ERROR: metadata dataset not found at $METADATA_DATASET_PATH"
    exit 1
fi

ARGS=(
    eval_utils/diagnose_libero_fit.py
    --model-path "$MODEL_PATH"
    --demo-file "$DEMO_FILE"
    --metadata-dataset-path "$METADATA_DATASET_PATH"
    --libero-root "$LIBERO_ROOT"
    --benchmark-name "$BENCHMARK_NAME"
    --task-order-index "$TASK_ORDER_INDEX"
    --alignment teacher_forced_open_loop
    --offline-only
    --device "$DEVICE"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --history-frames "$HISTORY_FRAMES"
    --open-loop-horizon "$OPEN_LOOP_HORIZON"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$TASK_ID" ]; then
    ARGS+=(--task-id "$TASK_ID")
fi

if [ -n "$MAX_OFFLINE_STEPS" ]; then
    ARGS+=(--max-offline-steps "$MAX_OFFLINE_STEPS")
fi

if [ "$SAVE_VIDEO_PRED" = "true" ]; then
    ARGS+=(--save-video-pred)
fi

if [ -n "$DEMO_IDS" ]; then
    # shellcheck disable=SC2206
    DEMO_ID_ARRAY=($DEMO_IDS)
    ARGS+=(--demo-ids "${DEMO_ID_ARRAY[@]}")
fi

cd "$REPO_ROOT"
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting multi-process teacher-forced LIBERO diagnostics with torchrun on $NUM_GPUS GPUs"
    echo "Note: this shards demos across ranks; it does not tensor-parallelize a single demo inference."
    torchrun --nproc_per_node "$NUM_GPUS" --standalone "${ARGS[@]}" "$@"
else
    echo "Starting single-process teacher-forced LIBERO diagnostics on $DEVICE"
    python "${ARGS[@]}" "$@"
fi
