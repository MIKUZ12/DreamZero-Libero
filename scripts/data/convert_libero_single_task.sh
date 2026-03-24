#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

LIBERO_HDF5_PATH=${LIBERO_HDF5_PATH:-"/local/yangshuo/fyhong/LIBERO/libero/datasets/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5"}
OUTPUT_PATH=${OUTPUT_PATH:-"$REPO_ROOT/data/libero_goal_single_task_lerobot"}
FPS=${FPS:-20}
LIMIT_DEMOS=${LIMIT_DEMOS:-}
SKIP_GEAR_METADATA=${SKIP_GEAR_METADATA:-false}
FORCE=${FORCE:-true}

if [ ! -f "$LIBERO_HDF5_PATH" ]; then
    echo "ERROR: LIBERO HDF5 file not found: $LIBERO_HDF5_PATH"
    exit 1
fi

CMD=(
    python
    "$REPO_ROOT/scripts/data/convert_libero.py"
    --hdf5-path "$LIBERO_HDF5_PATH"
    --output-path "$OUTPUT_PATH"
    --fps "$FPS"
)

if [ -n "$LIMIT_DEMOS" ]; then
    CMD+=(--limit-demos "$LIMIT_DEMOS")
fi

if [ "$SKIP_GEAR_METADATA" = "true" ]; then
    CMD+=(--skip-gear-metadata)
fi

if [ "$FORCE" = "true" ]; then
    CMD+=(--force)
fi

echo "Converting single LIBERO task:"
echo "  HDF5:   $LIBERO_HDF5_PATH"
echo "  Output: $OUTPUT_PATH"
"${CMD[@]}"

