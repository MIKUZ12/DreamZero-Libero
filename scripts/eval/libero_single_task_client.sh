#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"$REPO_ROOT/checkpoints/dreamzero_libero_single_task"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
LIBERO_ROOT=${LIBERO_ROOT:-"/local/yangshuo/fyhong/LIBERO"}
SERVER_HOST=${SERVER_HOST:-"localhost"}
SERVER_PORT=${SERVER_PORT:-8000}
BENCHMARK_NAME=${BENCHMARK_NAME:-"libero_spatial"}
TASK_IDS=${TASK_IDS:-}
N_EVAL=${N_EVAL:-20}
MAX_STEPS=${MAX_STEPS:-400}
OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON:-8}
HISTORY_FRAMES=${HISTORY_FRAMES:-33}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-128}
CAMERA_WIDTH=${CAMERA_WIDTH:-128}
TASK_ORDER_INDEX=${TASK_ORDER_INDEX:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/runs/libero_goal_single_task_eval"}
SAVE_VIDEO=${SAVE_VIDEO:-false}
SAVE_VIDEO_PRED=${SAVE_VIDEO_PRED:-false}
DEBUG_OPEN_LOOP=${DEBUG_OPEN_LOOP:-false}
VIDEO_EPISODES_PER_TASK=${VIDEO_EPISODES_PER_TASK:-1}
REUSE_SERVER_CACHE_ACROSS_REQUESTS=${REUSE_SERVER_CACHE_ACROSS_REQUESTS:-true}
RESET_SERVER_EACH_REQUEST=${RESET_SERVER_EACH_REQUEST:-false}

if [ -z "$CHECKPOINT_PATH" ]; then
    LATEST_CHECKPOINT=$(find "$TRAIN_OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_PATH="$LATEST_CHECKPOINT"
    fi
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH is not set and no checkpoint-* directory was found under $TRAIN_OUTPUT_DIR"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: checkpoint directory not found at $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$LIBERO_ROOT" ]; then
    echo "ERROR: LIBERO_ROOT not found at $LIBERO_ROOT"
    exit 1
fi

CLIENT_ARGS=(
    eval_utils/run_libero_eval.py
    --libero-root "$LIBERO_ROOT"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --benchmark-name "$BENCHMARK_NAME"
    --task-order-index "$TASK_ORDER_INDEX"
    --n-eval "$N_EVAL"
    --max-steps "$MAX_STEPS"
    --open-loop-horizon "$OPEN_LOOP_HORIZON"
    --history-frames "$HISTORY_FRAMES"
    --camera-height "$CAMERA_HEIGHT"
    --camera-width "$CAMERA_WIDTH"
    --checkpoint-path "$CHECKPOINT_PATH"
    --output-dir "$OUTPUT_DIR"
    --video-episodes-per-task "$VIDEO_EPISODES_PER_TASK"
)

if [ "$SAVE_VIDEO" = "true" ]; then
    CLIENT_ARGS+=(--save-video)
fi

if [ "$SAVE_VIDEO_PRED" = "true" ]; then
    CLIENT_ARGS+=(--save-video-pred)
fi

if [ "$DEBUG_OPEN_LOOP" = "true" ]; then
    CLIENT_ARGS+=(--debug-open-loop)
fi

if [ "$REUSE_SERVER_CACHE_ACROSS_REQUESTS" = "true" ]; then
    CLIENT_ARGS+=(--reuse-server-cache-across-requests)
fi

if [ "$RESET_SERVER_EACH_REQUEST" = "true" ]; then
    CLIENT_ARGS+=(--reset-server-each-request)
fi

if [ -n "$TASK_IDS" ]; then
    # shellcheck disable=SC2206
    TASK_ID_ARRAY=($TASK_IDS)
    CLIENT_ARGS+=(--task-ids "${TASK_ID_ARRAY[@]}")
fi

if [ "$#" -gt 0 ]; then
    CLIENT_ARGS+=("$@")
fi

cd "$REPO_ROOT"
echo "Starting LIBERO eval client (benchmark=${BENCHMARK_NAME}, host=${SERVER_HOST}:${SERVER_PORT}, history_frames=${HISTORY_FRAMES}, open_loop_horizon=${OPEN_LOOP_HORIZON}, reuse_server_cache_across_requests=${REUSE_SERVER_CACHE_ACROSS_REQUESTS}, reset_server_each_request=${RESET_SERVER_EACH_REQUEST})"
python "${CLIENT_ARGS[@]}"
