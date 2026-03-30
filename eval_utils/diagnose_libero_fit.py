#!/usr/bin/env python3
"""Diagnose LIBERO train-fit vs rollout behavior for a DreamZero checkpoint.

This script provides two complementary diagnostics:

1. Offline action-space fit on training demos:
   Feed ground-truth LIBERO image histories + state into the policy and compare
   predicted actions against the demo actions in action space.

2. Closed-loop rollout comparison in sim:
   Reconstruct a demo scene in OffScreenRenderEnv, then save a side-by-side
   video of dataset-action replay vs model rollout from the same initial state.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on active env
    torch = None

if torch is not None:
    import torch.distributed as dist
else:  # pragma: no cover - depends on active env
    dist = None

try:
    import robosuite.utils.transform_utils as T
except ModuleNotFoundError:  # pragma: no cover - depends on active env
    T = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LIBERO_ROOT = Path(__file__).resolve().parents[2] / "LIBERO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="DreamZero checkpoint path.")
    parser.add_argument("--demo-file", type=Path, required=True, help="LIBERO HDF5 demo file.")
    parser.add_argument(
        "--metadata-dataset-path",
        type=Path,
        required=True,
        help="Converted DreamZero/LIBERO dataset root used to build metadata.",
    )
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="Path to local LIBERO repo.")
    parser.add_argument("--benchmark-name", type=str, default="libero_goal", help="LIBERO benchmark name.")
    parser.add_argument("--task-order-index", type=int, default=0, help="Benchmark task-order index.")
    parser.add_argument("--task-id", type=int, default=None, help="Optional explicit task id.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only run offline action-space diagnostics. Skip simulator rollout comparison and avoid LIBERO/robosuite imports.",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        choices=("eval_history", "train_chunk", "teacher_forced_open_loop"),
        default="train_chunk",
        help=(
            "Offline alignment mode. "
            "'eval_history' uses past-frame history like deployment; "
            "'train_chunk' uses the converted dataset's training-time future chunk semantics; "
            "'teacher_forced_open_loop' replays an episode with ground-truth observations but "
            "keeps the policy's causal cache between chunk requests like deployment."
        ),
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=1,
        help=(
            "How many predicted actions to consume from each chunk before re-querying the model. "
            "Used by teacher_forced_open_loop mode."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=43200,
        help="Distributed timeout in seconds for torchrun mode.",
    )
    parser.add_argument("--history-frames", type=int, default=25, help="Video history length.")
    parser.add_argument(
        "--demo-ids",
        type=int,
        nargs="*",
        default=[0],
        help="Demo ids to analyze offline. Default: only demo 0.",
    )
    parser.add_argument(
        "--max-offline-steps",
        type=int,
        default=None,
        help="Optional cap on analyzed steps per demo for offline metrics.",
    )
    parser.add_argument(
        "--rollout-demo-id",
        type=int,
        default=0,
        help="Demo id to use for the side-by-side rollout comparison.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=300,
        help="Maximum closed-loop rollout steps for the model video.",
    )
    parser.add_argument("--camera-height", type=int, default=128, help="Render camera height.")
    parser.add_argument("--camera-width", type=int, default=128, help="Render camera width.")
    parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
    parser.add_argument(
        "--save-video-pred",
        action="store_true",
        help=(
            "Save decoded model video_pred clips for offline diagnostics. "
            "With teacher_forced_open_loop and open_loop_horizon=1, this gives one clip per step."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/libero_fit_diagnostics"),
        help="Directory for JSON summaries, CSV traces, and rollout video.",
    )
    return parser.parse_args()


def ensure_libero_imports(libero_root: Path) -> None:
    libero_root = libero_root.resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


def extract_language(demo_file: Path, data_group: h5py.Group) -> str:
    if "problem_info" in data_group.attrs:
        problem_info = json.loads(data_group.attrs["problem_info"])
        language = problem_info.get("language_instruction", "").strip()
        if language:
            return language

    if "env_args" in data_group.attrs:
        env_args = json.loads(data_group.attrs["env_args"])
        language = env_args.get("problem_info", {}).get("language_instruction", "").strip()
        if language:
            return language

    stem = demo_file.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ").strip()


def infer_task_id(benchmark, demo_file: Path) -> int:
    demo_name = demo_file.name
    matches: list[int] = []
    for task_id in range(benchmark.n_tasks):
        benchmark_demo = benchmark.get_task_demonstration(task_id)
        if Path(benchmark_demo).name == demo_name:
            matches.append(task_id)
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely infer task id for {demo_name}. Matches found: {matches}. "
            "Pass --task-id explicitly."
        )
    return matches[0]


def remap_demo_model_xml(xml_str: str, libero_root: Path) -> str:
    assets_root = (libero_root / "libero" / "libero" / "assets").resolve()
    libero_asset_roots = {
        "articulated_objects",
        "scenes",
        "stable_hope_objects",
        "stable_scanned_objects",
        "textures",
        "turbosquid_objects",
    }
    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    if asset is None:
        return xml_str

    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old_path = elem.get("file")
        if not old_path:
            continue
        parts = Path(old_path).parts
        if "assets" in parts:
            idx = parts.index("assets")
            relative_parts = parts[idx + 1 :]
            if relative_parts and relative_parts[0] in libero_asset_roots:
                new_path = assets_root.joinpath(*relative_parts)
                elem.set("file", str(new_path))
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def left_pad_history(frames: np.ndarray, end_index: int, history_frames: int) -> np.ndarray:
    start_index = max(0, end_index - history_frames + 1)
    window = [frame for frame in frames[start_index : end_index + 1]]
    if not window:
        raise RuntimeError("Cannot build history from an empty frame sequence.")
    padded = [window[0]] * max(0, history_frames - len(window)) + window
    return np.stack(padded[-history_frames:], axis=0)


def stack_recent_buffer(frames: list[np.ndarray], history_frames: int) -> np.ndarray:
    if not frames:
        raise RuntimeError("Frame buffer is empty.")
    padded = [frames[0]] * max(0, history_frames - len(frames)) + frames
    return np.stack(padded[-history_frames:], axis=0)


def make_rollout_frame(obs: dict) -> np.ndarray:
    agentview = np.flipud(np.asarray(obs["agentview_image"], dtype=np.uint8)).copy()
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)).copy()
    if wrist.shape[0] != agentview.shape[0]:
        row_idx = np.linspace(0, wrist.shape[0] - 1, agentview.shape[0]).astype(np.int64)
        target_width = max(1, int(round(wrist.shape[1] * agentview.shape[0] / wrist.shape[0])))
        col_idx = np.linspace(0, wrist.shape[1] - 1, target_width).astype(np.int64)
        wrist = wrist[row_idx][:, col_idx]
    return np.concatenate([agentview, wrist], axis=1)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with last dim 4, got shape {quat.shape}")
    if T is not None:
        return np.asarray(T.quat2axisangle(quat), dtype=np.float64)

    xyz = quat[:3]
    w = quat[3]
    xyz_norm = np.linalg.norm(xyz)
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(xyz_norm, w)
    axis = xyz / xyz_norm
    return axis * angle


def build_payload(
    agentview_video: np.ndarray,
    wrist_video: np.ndarray,
    eef_state: np.ndarray,
    gripper_position: np.ndarray,
    prompt: str,
    session_id: str | None = None,
    client_request_index: int = 0,
    client_env_step_index: int = 0,
    client_open_loop_horizon: int = 1,
    return_video_pred: bool = False,
) -> dict:
    return {
        "observation/exterior_image_0_left": np.asarray(agentview_video, dtype=np.uint8),
        "observation/wrist_image_left": np.asarray(wrist_video, dtype=np.uint8),
        "observation/ee_state": np.asarray(eef_state, dtype=np.float64),
        "observation/gripper_position": np.asarray(gripper_position, dtype=np.float64),
        "prompt": prompt,
        "session_id": str(uuid.uuid4()) if session_id is None else session_id,
        "client_request_index": client_request_index,
        "client_env_step_index": client_env_step_index,
        "client_open_loop_horizon": client_open_loop_horizon,
        "return_video_pred": return_video_pred,
    }


def format_action_dict(action_dict: dict, threshold_gripper: bool) -> np.ndarray:
    pose_delta = action_dict["action.pose_delta"]
    gripper = action_dict["action.gripper_position"]

    if isinstance(pose_delta, torch.Tensor):
        pose_delta = pose_delta.detach().cpu().numpy()
    if isinstance(gripper, torch.Tensor):
        gripper = gripper.detach().cpu().numpy()

    pose_delta = np.asarray(pose_delta, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)

    if pose_delta.ndim == 1:
        pose_delta = pose_delta[None, :]
    if gripper.ndim == 0:
        gripper = gripper.reshape(1, 1)
    elif gripper.ndim == 1:
        gripper = gripper[:, None]

    if threshold_gripper:
        gripper = 2.0 * gripper - 1.0
        gripper = np.sign(gripper)
        gripper = -gripper.astype(np.float32)

    return np.concatenate([pose_delta, gripper], axis=-1)


def summarize_offline_trace(trace: list[dict[str, float | int]]) -> dict[str, object]:
    first_pose_l2 = [float(row["first_pose_l2"]) for row in trace]
    first_action_l2 = [float(row["first_action_l2"]) for row in trace]
    first_pose_mae = [float(row["first_pose_mae"]) for row in trace]
    first_gripper_sign = [float(row["first_gripper_sign_match"]) for row in trace]
    chunk_pose_mae = [float(row["chunk_pose_mae"]) for row in trace]
    chunk_action_l2 = [float(row["chunk_action_l2"]) for row in trace]
    chunk_gripper_sign = [float(row["chunk_gripper_sign_accuracy"]) for row in trace]
    return {
        "num_steps_evaluated": len(trace),
        "mean_first_pose_l2": float(np.mean(first_pose_l2)),
        "mean_first_pose_mae": float(np.mean(first_pose_mae)),
        "mean_first_action_l2": float(np.mean(first_action_l2)),
        "first_gripper_sign_accuracy": float(np.mean(first_gripper_sign)),
        "mean_chunk_pose_mae": float(np.mean(chunk_pose_mae)),
        "mean_chunk_action_l2": float(np.mean(chunk_action_l2)),
        "chunk_gripper_sign_accuracy": float(np.mean(chunk_gripper_sign)),
        "trace": trace,
    }


def summarize_teacher_forced_open_loop_trace(trace: list[dict[str, float | int | bool]]) -> dict[str, object]:
    pose_l2 = [float(row["pose_l2"]) for row in trace]
    pose_mae = [float(row["pose_mae"]) for row in trace]
    action_l2 = [float(row["action_l2"]) for row in trace]
    gripper_sign = [float(row["gripper_sign_match"]) for row in trace]
    pred_abs_max = [float(row["pred_action_abs_max"]) for row in trace]
    query_steps = [int(bool(row["query_step"])) for row in trace]
    chunk_steps = [int(row["chunk_step_limit"]) for row in trace]
    nan_rows = [bool(row["pred_action_has_nan"]) for row in trace]
    inf_rows = [bool(row["pred_action_has_inf"]) for row in trace]
    return {
        "num_steps_evaluated": len(trace),
        "num_model_queries": int(np.sum(query_steps)),
        "mean_pose_l2": float(np.mean(pose_l2)),
        "mean_pose_mae": float(np.mean(pose_mae)),
        "mean_action_l2": float(np.mean(action_l2)),
        "gripper_sign_accuracy": float(np.mean(gripper_sign)),
        "mean_pred_action_abs_max": float(np.mean(pred_abs_max)),
        "max_pred_action_abs_max": float(np.max(pred_abs_max)),
        "mean_chunk_step_limit": float(np.mean(chunk_steps)),
        "num_nan_actions": int(np.sum(nan_rows)),
        "num_inf_actions": int(np.sum(inf_rows)),
        "trace": trace,
    }


def compute_trace_row(step_index: int, pred_raw: np.ndarray, pred_eval: np.ndarray, gt_chunk: np.ndarray) -> dict[str, float | int]:
    pose_err = pred_eval[:, :6] - gt_chunk[:, :6]
    first_pose_l2_value = float(np.linalg.norm(pose_err[0]))
    first_pose_mae_value = float(np.mean(np.abs(pose_err[0])))
    first_action_l2_value = float(np.linalg.norm(pred_eval[0] - gt_chunk[0]))
    first_gripper_match_value = float(np.sign(pred_eval[0, 6]) == np.sign(gt_chunk[0, 6]))
    chunk_pose_mae_value = float(np.mean(np.abs(pose_err)))
    chunk_action_l2_value = float(np.mean(np.linalg.norm(pred_eval - gt_chunk, axis=1)))
    chunk_gripper_sign_value = float(np.mean(np.sign(pred_eval[:, 6]) == np.sign(gt_chunk[:, 6])))

    return {
        "step": step_index,
        "gt_pose_x": float(gt_chunk[0, 0]),
        "gt_pose_y": float(gt_chunk[0, 1]),
        "gt_pose_z": float(gt_chunk[0, 2]),
        "pred_pose_x": float(pred_raw[0, 0]),
        "pred_pose_y": float(pred_raw[0, 1]),
        "pred_pose_z": float(pred_raw[0, 2]),
        "gt_gripper": float(gt_chunk[0, 6]),
        "pred_gripper_raw": float(pred_raw[0, 6]),
        "pred_gripper_eval": float(pred_eval[0, 6]),
        "first_pose_l2": first_pose_l2_value,
        "first_pose_mae": first_pose_mae_value,
        "first_action_l2": first_action_l2_value,
        "first_gripper_sign_match": first_gripper_match_value,
        "chunk_pose_mae": chunk_pose_mae_value,
        "chunk_action_l2": chunk_action_l2_value,
        "chunk_gripper_sign_accuracy": chunk_gripper_sign_value,
    }


def compute_offline_metrics_eval_history(
    policy_wrapper,
    demo_group: h5py.Group,
    prompt: str,
    history_frames: int,
    max_steps: int | None,
) -> dict:
    actions = np.asarray(demo_group["actions"][()], dtype=np.float32)
    eef_states = np.asarray(demo_group["obs"]["ee_states"][()], dtype=np.float64)
    gripper_states = np.asarray(demo_group["obs"]["gripper_states"][()], dtype=np.float64)
    agentview = np.asarray(demo_group["obs"]["agentview_rgb"][()], dtype=np.uint8)
    wrist = np.asarray(demo_group["obs"]["eye_in_hand_rgb"][()], dtype=np.uint8)

    num_steps = len(actions) if max_steps is None else min(len(actions), max_steps)
    trace: list[dict[str, float | int]] = []

    for step_index in range(num_steps):
        # Offline teacher-forcing should evaluate each timestep independently
        # under the ground-truth history/state, without accumulating causal KV
        # caches across the whole demo. Reusing the rollout cache here both
        # changes the meaning of the metric and can balloon memory usage.
        policy_wrapper.reset({"session_id": f"offline-{uuid.uuid4()}"})
        payload = build_payload(
            agentview_video=left_pad_history(agentview, step_index, history_frames),
            wrist_video=left_pad_history(wrist, step_index, history_frames),
            eef_state=eef_states[step_index],
            gripper_position=gripper_states[step_index],
            prompt=prompt,
        )
        result_batch = policy_wrapper._forward(payload)
        pred_raw = format_action_dict(result_batch.act, threshold_gripper=False)
        pred_eval = format_action_dict(result_batch.act, threshold_gripper=True)

        gt_chunk = actions[step_index : step_index + pred_eval.shape[0]]
        valid_len = min(len(gt_chunk), len(pred_eval))
        pred_chunk = pred_eval[:valid_len]
        pred_chunk_raw = pred_raw[:valid_len]
        gt_chunk = gt_chunk[:valid_len]
        trace.append(compute_trace_row(step_index, pred_chunk_raw, pred_chunk, gt_chunk))

    return summarize_offline_trace(trace)


def compute_offline_metrics_train_chunk(
    policy_wrapper,
    dataset,
    demo_id: int,
    prompt: str,
    max_steps: int | None,
) -> dict:
    trajectory_id = int(demo_id)
    trajectory_length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(trajectory_id)])
    usable_steps = trajectory_length - dataset.max_delta_index
    num_steps = usable_steps if max_steps is None else min(usable_steps, max_steps)
    trace: list[dict[str, float | int]] = []

    for step_index in range(num_steps):
        policy_wrapper.reset({"session_id": f"offline-{uuid.uuid4()}"})
        indices = {
            key: delta_indices + step_index for key, delta_indices in dataset.delta_indices.items()
        }
        step_data = dataset.get_step_data(trajectory_id, indices)

        payload = build_payload(
            agentview_video=np.asarray(step_data["video.agentview_rgb"], dtype=np.uint8),
            wrist_video=np.asarray(step_data["video.eye_in_hand_rgb"], dtype=np.uint8),
            eef_state=np.asarray(step_data["state.eef_state"], dtype=np.float64)[0],
            gripper_position=np.asarray(step_data["state.gripper_state"], dtype=np.float64)[0],
            prompt=prompt,
        )
        result_batch = policy_wrapper._forward(payload)
        pred_raw = format_action_dict(result_batch.act, threshold_gripper=False)
        pred_eval = format_action_dict(result_batch.act, threshold_gripper=True)

        gt_pose = np.asarray(step_data["action.pose_delta"], dtype=np.float32)
        gt_gripper = np.asarray(step_data["action.gripper_position"], dtype=np.float32)
        if gt_gripper.ndim == 1:
            gt_gripper = gt_gripper[:, None]
        gt_chunk = np.concatenate([gt_pose, gt_gripper], axis=-1)

        valid_len = min(len(gt_chunk), len(pred_eval))
        trace.append(
            compute_trace_row(
                step_index,
                pred_raw[:valid_len],
                pred_eval[:valid_len],
                gt_chunk[:valid_len],
            )
        )

    return summarize_offline_trace(trace)


def compute_teacher_forced_open_loop_metrics(
    policy_wrapper,
    demo_group: h5py.Group,
    prompt: str,
    history_frames: int,
    open_loop_horizon: int,
    max_steps: int | None,
    save_video_pred: bool,
    video_pred_dir: Path | None,
    fps: int,
) -> dict:
    actions = np.asarray(demo_group["actions"][()], dtype=np.float32)
    eef_states = np.asarray(demo_group["obs"]["ee_states"][()], dtype=np.float64)
    gripper_states = np.asarray(demo_group["obs"]["gripper_states"][()], dtype=np.float64)
    agentview = np.asarray(demo_group["obs"]["agentview_rgb"][()], dtype=np.uint8)
    wrist = np.asarray(demo_group["obs"]["eye_in_hand_rgb"][()], dtype=np.uint8)

    num_steps = len(actions) if max_steps is None else min(len(actions), max_steps)
    trace: list[dict[str, float | int | bool]] = []
    session_id = f"teacher-forced-open-loop-{uuid.uuid4()}"
    policy_wrapper.reset({"session_id": session_id})

    pred_raw_chunk: np.ndarray | None = None
    pred_eval_chunk: np.ndarray | None = None
    actions_from_chunk_completed = 0
    request_index = 0

    for step_index in range(num_steps):
        needs_new_chunk = (
            actions_from_chunk_completed == 0
            or pred_eval_chunk is None
            or actions_from_chunk_completed >= min(open_loop_horizon, len(pred_eval_chunk))
        )
        query_step = False

        if needs_new_chunk:
            actions_from_chunk_completed = 0
            request_index += 1
            # Re-query each chunk from the current ground-truth observation/history
            # under a fresh policy cache. This keeps the diagnostic aligned with
            # "teacher-forced open-loop over the dataset" instead of carrying the
            # model's internal latent-video cache across chunk requests.
            policy_wrapper.reset({"session_id": session_id, "request_index": request_index})
            payload = build_payload(
                agentview_video=left_pad_history(agentview, step_index, history_frames),
                wrist_video=left_pad_history(wrist, step_index, history_frames),
                eef_state=eef_states[step_index],
                gripper_position=gripper_states[step_index],
                prompt=prompt,
                session_id=session_id,
                client_request_index=request_index,
                client_env_step_index=step_index,
                client_open_loop_horizon=open_loop_horizon,
                return_video_pred=save_video_pred,
            )
            if save_video_pred:
                result_batch, video_pred = policy_wrapper._forward_with_video(payload)
                pred_video_path = None
                if video_pred_dir is not None:
                    decoded_video_pred = policy_wrapper._decode_video_pred(video_pred)
                    pred_video_path = video_pred_dir / f"request_{request_index:04d}_step_{step_index:04d}.mp4"
                    write_video_clip(decoded_video_pred, pred_video_path, fps=fps)
            else:
                result_batch = policy_wrapper._forward(payload)
                pred_video_path = None
            pred_raw_chunk = format_action_dict(result_batch.act, threshold_gripper=False)
            pred_eval_chunk = format_action_dict(result_batch.act, threshold_gripper=True)
            query_step = True
        else:
            pred_video_path = None

        assert pred_raw_chunk is not None and pred_eval_chunk is not None
        chunk_step_limit = min(open_loop_horizon, len(pred_eval_chunk))
        action_index_in_chunk = actions_from_chunk_completed
        pred_raw = pred_raw_chunk[action_index_in_chunk]
        pred_eval = pred_eval_chunk[action_index_in_chunk]
        gt_action = actions[step_index]

        trace.append(
            {
                "step": step_index,
                "query_step": query_step,
                "request_index": request_index,
                "action_index_in_chunk": action_index_in_chunk,
                "chunk_len": int(len(pred_eval_chunk)),
                "chunk_step_limit": int(chunk_step_limit),
                "gt_pose_x": float(gt_action[0]),
                "gt_pose_y": float(gt_action[1]),
                "gt_pose_z": float(gt_action[2]),
                "pred_pose_x": float(pred_raw[0]),
                "pred_pose_y": float(pred_raw[1]),
                "pred_pose_z": float(pred_raw[2]),
                "gt_gripper": float(gt_action[6]),
                "pred_gripper_raw": float(pred_raw[6]),
                "pred_gripper_eval": float(pred_eval[6]),
                "pose_l2": float(np.linalg.norm(pred_eval[:6] - gt_action[:6])),
                "pose_mae": float(np.mean(np.abs(pred_eval[:6] - gt_action[:6]))),
                "action_l2": float(np.linalg.norm(pred_eval - gt_action)),
                "gripper_sign_match": float(np.sign(pred_eval[6]) == np.sign(gt_action[6])),
                "pred_action_abs_max": float(np.max(np.abs(pred_eval))),
                "pred_action_has_nan": bool(np.isnan(pred_eval).any()),
                "pred_action_has_inf": bool(np.isinf(pred_eval).any()),
                "video_pred_path": str(pred_video_path.resolve()) if pred_video_path is not None else None,
            }
        )
        actions_from_chunk_completed += 1

    return summarize_teacher_forced_open_loop_trace(trace)


def run_rollout_comparison(
    policy_wrapper,
    task,
    prompt: str,
    demo_group: h5py.Group,
    libero_root: Path,
    history_frames: int,
    rollout_steps: int,
    camera_height: int,
    camera_width: int,
) -> dict:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import utils as libero_utils

    actions = np.asarray(demo_group["actions"][()], dtype=np.float32)
    states = np.asarray(demo_group["states"][()], dtype=np.float64)
    model_xml = remap_demo_model_xml(demo_group.attrs["model_file"], libero_root)
    model_xml = libero_utils.postprocess_model_xml(model_xml, {})

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_kwargs = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": camera_height,
        "camera_widths": camera_width,
    }
    env_replay = OffScreenRenderEnv(**env_kwargs)
    env_model = OffScreenRenderEnv(**env_kwargs)

    def reset_env(env):
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        return env.set_init_state(states[0])

    obs_replay = reset_env(env_replay)
    obs_model = reset_env(env_model)

    replay_frames = [make_rollout_frame(obs_replay)]
    model_frames = [make_rollout_frame(obs_model)]
    history_buffers = {
        "agentview": [np.asarray(obs_model["agentview_image"], dtype=np.uint8)],
        "wrist": [np.asarray(obs_model["robot0_eye_in_hand_image"], dtype=np.uint8)],
    }
    policy_wrapper.reset({"session_id": f"rollout-{uuid.uuid4()}"})

    rollout_trace: list[dict[str, float | int]] = []
    state_errors = []

    max_steps = min(len(actions), rollout_steps)
    success = False

    for step_index in range(max_steps):
        payload = build_payload(
            agentview_video=stack_recent_buffer(history_buffers["agentview"], history_frames),
            wrist_video=stack_recent_buffer(history_buffers["wrist"], history_frames),
            eef_state=np.concatenate(
                [
                    np.asarray(obs_model["robot0_eef_pos"], dtype=np.float64),
                    quat_to_axis_angle(np.asarray(obs_model["robot0_eef_quat"], dtype=np.float64)),
                ]
            ),
            gripper_position=np.asarray(obs_model["robot0_gripper_qpos"], dtype=np.float64),
            prompt=prompt,
        )
        pred_action = policy_wrapper.infer(payload)["actions"][0]
        gt_action = actions[step_index]

        obs_model, reward_model, done_model, _ = env_model.step(pred_action)
        obs_replay, _, _, _ = env_replay.step(gt_action)

        history_buffers["agentview"].append(np.asarray(obs_model["agentview_image"], dtype=np.uint8))
        history_buffers["wrist"].append(np.asarray(obs_model["robot0_eye_in_hand_image"], dtype=np.uint8))
        if len(history_buffers["agentview"]) > history_frames:
            del history_buffers["agentview"][:-history_frames]
            del history_buffers["wrist"][:-history_frames]

        replay_frames.append(make_rollout_frame(obs_replay))
        model_frames.append(make_rollout_frame(obs_model))

        model_state = env_model.sim.get_state().flatten()
        gt_state = states[min(step_index + 1, len(states) - 1)]
        state_error = float(np.linalg.norm(model_state - gt_state))
        state_errors.append(state_error)
        rollout_trace.append(
            {
                "step": step_index,
                "gt_action_0": float(gt_action[0]),
                "pred_action_0": float(pred_action[0]),
                "gt_gripper": float(gt_action[6]),
                "pred_gripper": float(pred_action[6]),
                "state_error_to_demo": state_error,
                "reward_model": float(reward_model),
            }
        )
        if reward_model > 0:
            success = True
        if done_model:
            break

    env_replay.close()
    env_model.close()

    side_by_side = []
    total_frames = max(len(replay_frames), len(model_frames))
    for index in range(total_frames):
        left = replay_frames[min(index, len(replay_frames) - 1)]
        right = model_frames[min(index, len(model_frames) - 1)]
        side_by_side.append(np.concatenate([left, right], axis=1))

    return {
        "frames": side_by_side,
        "trace": rollout_trace,
        "success": success,
        "mean_state_error_to_demo": float(np.mean(state_errors)) if state_errors else 0.0,
        "max_state_error_to_demo": float(np.max(state_errors)) if state_errors else 0.0,
        "num_steps_executed": len(rollout_trace),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_video_clip(frames: np.ndarray, output_path: Path, fps: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames), fps=fps, codec="libx264")


def init_runtime(device_arg: str, timeout_seconds: int) -> tuple[str, int, int, bool]:
    """Initialize distributed runtime for single-process or torchrun execution."""
    if dist is None or not dist.is_available():
        return device_arg, 0, 1, False

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size > 1 and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}", rank, world_size, False
        return device_arg, rank, world_size, False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        initialized_here = True
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        return device, rank, world_size, initialized_here

    initialized_here = True
    init_method = f"file:///tmp/dreamzero_diagnose_{uuid.uuid4().hex}"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=0,
        world_size=1,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    return device_arg, 0, 1, initialized_here


def shard_demo_ids(demo_ids: list[int], rank: int, world_size: int) -> list[int]:
    return [demo_id for index, demo_id in enumerate(demo_ids) if index % world_size == rank]


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError(
            "torch is not available in the current Python environment. "
            "Run this script from the DreamZero conda environment."
        )
    device, rank, world_size, initialized_dist_here = init_runtime(args.device, args.timeout_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_libero_imports(args.libero_root)

    torch._dynamo.config.recompile_limit = 800
    torch._dynamo.config.cache_size_limit = 1000

    from groot.vla.data.dataset.lerobot import LeRobotSingleDataset
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    from eval_utils.run_libero_server import LiberoDreamZeroPolicy

    benchmark = None
    task = None
    task_id = args.task_id
    if not args.offline_only or task_id is None:
        from libero.libero.benchmark import get_benchmark

        benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
        task_id = task_id if task_id is not None else infer_task_id(benchmark, args.demo_file)
        task = benchmark.get_task(task_id)
    assert task_id is not None

    default_language = None
    default_task_name = args.demo_file.stem.replace("_demo", "")
    if task is not None:
        default_language = task.language
        default_task_name = task.name

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=str(args.model_path),
        device=device,
        model_config_overrides=[],
        metadata_dataset_path=str(args.metadata_dataset_path),
    )
    policy_wrapper = LiberoDreamZeroPolicy(policy)
    if args.alignment == "teacher_forced_open_loop":
        model_num_frames = None
        try:
            model_num_frames = int(policy.trained_model.config.action_head_cfg.num_frames)
        except Exception:
            model_num_frames = int(policy.train_cfg.get("num_frames", args.history_frames))
        if args.history_frames != model_num_frames and rank == 0:
            print(
                f"[teacher_forced_open_loop] overriding history_frames from {args.history_frames} "
                f"to checkpoint num_frames={model_num_frames} for causal inference compatibility"
            )
        args.history_frames = model_num_frames
    offline_dataset = None
    if args.alignment == "train_chunk":
        fps = None
        if "fps" in policy.train_cfg and EmbodimentTag.LIBERO_SIM.value in policy.train_cfg.fps:
            fps = policy.train_cfg.fps[EmbodimentTag.LIBERO_SIM.value]
        offline_dataset = LeRobotSingleDataset(
            dataset_path=str(args.metadata_dataset_path),
            modality_configs=policy.modality_configs,
            embodiment_tag=EmbodimentTag.LIBERO_SIM,
            use_global_metadata=False,
            transforms=None,
            fps=fps,
            max_chunk_size=policy.train_cfg.get("max_chunk_size", None),
            relative_action=policy.train_cfg.get("relative_action", False),
            relative_action_keys=policy.train_cfg.get("relative_action_keys", None),
            relative_action_per_horizon=policy.train_cfg.get("relative_action_per_horizon", False),
        )

    summary: dict[str, object] = {
        "model_path": str(args.model_path.resolve()),
        "demo_file": str(args.demo_file.resolve()),
        "metadata_dataset_path": str(args.metadata_dataset_path.resolve()),
        "benchmark_name": args.benchmark_name,
        "task_order_index": args.task_order_index,
        "task_id": task_id,
        "task_name": default_task_name,
        "language": default_language,
        "history_frames": args.history_frames,
        "open_loop_horizon": args.open_loop_horizon,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "offline_only": args.offline_only,
        "alignment": args.alignment,
        "save_video_pred": args.save_video_pred,
        "offline_demos": [],
    }

    assigned_demo_ids = shard_demo_ids(list(args.demo_ids), rank, world_size)

    with h5py.File(args.demo_file, "r") as demo_h5:
        progress_desc = f"Offline fit diagnostics rank{rank}"
        for demo_id in tqdm(
            assigned_demo_ids,
            desc=progress_desc,
            unit="demo",
            disable=(rank != 0 and len(assigned_demo_ids) <= 1),
        ):
            demo_name = f"demo_{demo_id}"
            if demo_name not in demo_h5["data"]:
                raise KeyError(f"{demo_name} not found in {args.demo_file}")
            demo_group = demo_h5["data"][demo_name]
            prompt = extract_language(args.demo_file, demo_group)
            if summary["language"] is None:
                summary["language"] = prompt
            if args.alignment == "train_chunk":
                assert offline_dataset is not None
                metrics = compute_offline_metrics_train_chunk(
                    policy_wrapper=policy_wrapper,
                    dataset=offline_dataset,
                    demo_id=demo_id,
                    prompt=prompt,
                    max_steps=args.max_offline_steps,
                )
            elif args.alignment == "teacher_forced_open_loop":
                metrics = compute_teacher_forced_open_loop_metrics(
                    policy_wrapper=policy_wrapper,
                    demo_group=demo_group,
                    prompt=prompt,
                    history_frames=args.history_frames,
                    open_loop_horizon=args.open_loop_horizon,
                    max_steps=args.max_offline_steps,
                    save_video_pred=args.save_video_pred,
                    video_pred_dir=(args.output_dir / "video_pred" / demo_name) if args.save_video_pred else None,
                    fps=args.fps,
                )
            else:
                metrics = compute_offline_metrics_eval_history(
                    policy_wrapper=policy_wrapper,
                    demo_group=demo_group,
                    prompt=prompt,
                    history_frames=args.history_frames,
                    max_steps=args.max_offline_steps,
                )
            write_jsonl(args.output_dir / "offline_traces" / f"{demo_name}.jsonl", metrics.pop("trace"))
            summary["offline_demos"].append({"demo_name": demo_name, **metrics})

    if world_size > 1:
        gathered_summaries = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_summaries, summary["offline_demos"])
        merged_offline = []
        for shard in gathered_summaries:
            if shard:
                merged_offline.extend(shard)
        merged_offline.sort(key=lambda item: item["demo_name"])
        summary["offline_demos"] = merged_offline

    if rank == 0 and not args.offline_only:
        assert task is not None
        with h5py.File(args.demo_file, "r") as demo_h5:
            rollout_name = f"demo_{args.rollout_demo_id}"
            if rollout_name not in demo_h5["data"]:
                raise KeyError(f"{rollout_name} not found in {args.demo_file}")
            rollout_demo = demo_h5["data"][rollout_name]
            rollout_result = run_rollout_comparison(
                policy_wrapper=policy_wrapper,
                task=task,
                prompt=task.language,
                demo_group=rollout_demo,
                libero_root=args.libero_root,
                history_frames=args.history_frames,
                rollout_steps=args.rollout_steps,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
            )

        video_path = args.output_dir / "videos" / f"{rollout_name}_replay_vs_model.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, rollout_result.pop("frames"), fps=args.fps, codec="libx264")
        write_jsonl(args.output_dir / "rollout_trace.jsonl", rollout_result.pop("trace"))
        summary["rollout_comparison"] = {
            "demo_name": rollout_name,
            "video_path": str(video_path.resolve()),
            **rollout_result,
        }

        write_json(args.output_dir / "summary.json", summary)
        print(f"Saved diagnostics to {args.output_dir.resolve()}")
    elif rank == 0:
        write_json(args.output_dir / "summary.json", summary)
        print(f"Saved offline diagnostics to {args.output_dir.resolve()}")

    if initialized_dist_here and dist is not None and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
