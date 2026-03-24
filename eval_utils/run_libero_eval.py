#!/usr/bin/env python3
"""Run official LIBERO benchmark rollouts against a remote DreamZero policy server."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import uuid
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import websockets.sync.client
from tqdm.auto import tqdm


DEFAULT_LIBERO_ROOT = Path(__file__).resolve().parents[2] / "LIBERO"


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    try:
        import robosuite.utils.transform_utils as T
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on eval env
        raise ModuleNotFoundError(
            "run_libero_eval.py requires robosuite in the LIBERO evaluation environment "
            "to convert robot0_eef_quat into observation/ee_state."
        ) from exc

    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim == 1:
        return T.quat2axisangle(quat)
    return np.stack([T.quat2axisangle(q) for q in quat], axis=0)


class PickleWebsocketClient:
    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=60,
            ping_timeout=600,
        )
        self._metadata = pickle.loads(self._ws.recv())

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        obs["endpoint"] = "infer"
        self._ws.send(pickle.dumps(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return pickle.loads(response)

    def reset(self, reset_info: dict | None = None) -> None:
        payload = {} if reset_info is None else dict(reset_info)
        payload["endpoint"] = "reset"
        self._ws.send(pickle.dumps(payload))
        self._ws.recv()


class DreamZeroLiberoClient:
    def __init__(
        self,
        host: str,
        port: int,
        open_loop_horizon: int = 8,
        history_frames: int = 25,
        debug_open_loop: bool = False,
        return_video_pred: bool = False,
    ) -> None:
        if history_frames <= 0:
            raise ValueError(f"history_frames must be positive, got {history_frames}")
        self.client = PickleWebsocketClient(host=host, port=port)
        self.open_loop_horizon = open_loop_horizon
        self.history_frames = history_frames
        self.debug_open_loop = debug_open_loop
        self.return_video_pred = return_video_pred
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.pred_video_chunks: list[np.ndarray] = []
        self.session_id = str(uuid.uuid4())
        self.request_index = 0
        self.env_step_index = 0
        self._is_first_request = True
        self._frame_buffers = {
            "agentview": [],
            "wrist": [],
        }

    def _reset_history(self) -> None:
        for frames in self._frame_buffers.values():
            frames.clear()

    def _append_history(self, obs: dict) -> None:
        frame_sources = {
            "agentview": np.asarray(obs["agentview_image"], dtype=np.uint8),
            "wrist": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
        }
        for key, frame in frame_sources.items():
            frames = self._frame_buffers[key]
            frames.append(frame)
            if len(frames) > self.history_frames:
                del frames[:-self.history_frames]

    def _stack_recent_frames(self, key: str, num_frames: int) -> np.ndarray:
        frames = self._frame_buffers[key]
        if not frames:
            raise RuntimeError(f"No buffered frames available for {key}")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        padded_frames = [frames[0]] * max(0, num_frames - len(frames)) + frames
        return np.stack(padded_frames[-num_frames:], axis=0)

    def reset(self) -> None:
        if self.debug_open_loop:
            tqdm.write(
                f"[client][reset] session={self.session_id} "
                f"completed_env_steps={self.env_step_index} total_requests={self.request_index}"
            )
        self.client.reset({"session_id": self.session_id})
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.pred_video_chunks = []
        self.session_id = str(uuid.uuid4())
        self.request_index = 0
        self.env_step_index = 0
        self._is_first_request = True
        self._reset_history()

    def infer(self, obs: dict, instruction: str) -> np.ndarray:
        self._append_history(obs)
        needs_new_chunk = (
            self.actions_from_chunk_completed == 0
            or self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= min(self.open_loop_horizon, len(self.pred_action_chunk))
        )
        if needs_new_chunk:
            self.actions_from_chunk_completed = 0
            self.request_index += 1
            # Use the full history length for every request, including the first.
            # _stack_recent_frames() already left-pads with the earliest frame when
            # the buffer is still short, which keeps inference aligned with the
            # train-time 25-frame context and avoids VAE failures on 1-frame inputs.
            request_frames = self.history_frames
            agentview_video = self._stack_recent_frames("agentview", request_frames)
            wrist_video = self._stack_recent_frames("wrist", request_frames)
            eef_state = np.concatenate(
                [
                    np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                    quat_to_axis_angle(obs["robot0_eef_quat"]),
                ],
                axis=-1,
            )
            request_data = {
                "observation/exterior_image_0_left": agentview_video,
                "observation/wrist_image_left": wrist_video,
                "observation/ee_state": eef_state,
                "observation/gripper_position": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
                "client_request_index": self.request_index,
                "client_env_step_index": self.env_step_index,
                "client_open_loop_horizon": self.open_loop_horizon,
                "return_video_pred": self.return_video_pred,
            }
            if self.debug_open_loop:
                tqdm.write(
                    f"[client][request] session={self.session_id} request={self.request_index} "
                    f"env_step={self.env_step_index} open_loop_horizon={self.open_loop_horizon} "
                    f"request_frames={request_frames} history_shape={agentview_video.shape}"
                )
            result = self.client.infer(request_data)
            self._is_first_request = False
            actions = result["actions"] if isinstance(result, dict) else result
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"Expected action chunk of shape (N, 7), got {actions.shape}")
            self.pred_action_chunk = actions
            if isinstance(result, dict) and "video_pred" in result:
                video_pred = np.asarray(result["video_pred"], dtype=np.uint8)
                if video_pred.ndim != 4:
                    raise ValueError(f"Expected video_pred clip with shape [T, H, W, C], got {video_pred.shape}")
                self.pred_video_chunks.append(video_pred)
            if self.debug_open_loop:
                tqdm.write(
                    f"[client][response] session={self.session_id} request={self.request_index} "
                    f"received_chunk_shape={actions.shape}"
                )
        elif self.debug_open_loop:
            chunk_len = len(self.pred_action_chunk)
            tqdm.write(
                f"[client][reuse] session={self.session_id} request={self.request_index} "
                f"env_step={self.env_step_index} action_index={self.actions_from_chunk_completed} "
                f"chunk_len={chunk_len} open_loop_limit={min(self.open_loop_horizon, chunk_len)}"
            )

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1
        self.env_step_index += 1
        return action


def ensure_libero_imports(libero_root: Path) -> None:
    libero_root = libero_root.resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def load_init_states(init_states_path: Path):
    """Load LIBERO init states compatibly across PyTorch versions."""
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        return torch.load(init_states_path)


def make_rollout_frame(obs: dict) -> np.ndarray:
    # LIBERO render frames are vertically inverted in our current setup.
    # Keep policy inputs unchanged and only correct orientation for saved videos.
    agentview = np.flipud(np.asarray(obs["agentview_image"], dtype=np.uint8)).copy()
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)).copy()
    if wrist.shape[0] != agentview.shape[0]:
        # Simple nearest-neighbor resize without extra deps.
        row_idx = np.linspace(0, wrist.shape[0] - 1, agentview.shape[0]).astype(np.int64)
        target_width = max(1, int(round(wrist.shape[1] * agentview.shape[0] / wrist.shape[0])))
        col_idx = np.linspace(0, wrist.shape[1] - 1, target_width).astype(np.int64)
        wrist = wrist[row_idx][:, col_idx]
    return np.concatenate([agentview, wrist], axis=1)


def write_rollout_video(frames: list[np.ndarray], output_path: Path, fps: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")


def write_video_clip(frames: np.ndarray, output_path: Path, fps: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames), fps=fps, codec="libx264")


def write_results(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["task_id", "task_name", "language", "success_rate"])
        writer.writeheader()
        for task in results["tasks"]:
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "language": task["language"],
                    "success_rate": task["success_rate"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="localhost", help="DreamZero policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="DreamZero policy server port.")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="Path to the local LIBERO repo.")
    parser.add_argument("--benchmark-name", type=str, default="libero_spatial", help="LIBERO benchmark name.")
    parser.add_argument("--task-order-index", type=int, default=0, help="Task order index for 10-task suites.")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None, help="Optional task ids to evaluate.")
    parser.add_argument("--n-eval", type=int, default=20, help="Episodes per task.")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum rollout length.")
    parser.add_argument("--camera-height", type=int, default=128, help="Render camera height.")
    parser.add_argument("--camera-width", type=int, default=128, help="Render camera width.")
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=8,
        help="How many predicted actions to execute per server call. Default matches DreamZero sim-eval.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=25,
        help="How many recent frames to send per policy request. Default matches LIBERO train-time frame count.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/libero_eval"), help="Directory for JSON/CSV results.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional checkpoint path recorded in results.json.")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos for the first few episodes of each task.")
    parser.add_argument(
        "--save-video-pred",
        action="store_true",
        help="Save DreamZero internal decoded video_pred clips for policy requests in the first few episodes of each task.",
    )
    parser.add_argument(
        "--debug-open-loop",
        action="store_true",
        help="Print detailed client-side request/reuse logs. Disabled by default to keep tqdm readable.",
    )
    parser.add_argument(
        "--video-episodes-per-task",
        type=int,
        default=1,
        help="How many episodes per task to save when --save-video is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_libero_imports(args.libero_root)

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
    task_ids = args.task_ids or list(range(benchmark.n_tasks))
    client = DreamZeroLiberoClient(
        args.host,
        args.port,
        open_loop_horizon=args.open_loop_horizon,
        history_frames=args.history_frames,
        debug_open_loop=args.debug_open_loop,
        return_video_pred=args.save_video_pred,
    )
    tqdm.write(
        f"[eval][start] benchmark={args.benchmark_name} task_ids={task_ids} "
        f"n_eval={args.n_eval} max_steps={args.max_steps} open_loop_horizon={args.open_loop_horizon} "
        f"history_frames={args.history_frames}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark_name": args.benchmark_name,
        "task_order_index": args.task_order_index,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "open_loop_horizon": args.open_loop_horizon,
        "history_frames": args.history_frames,
        "checkpoint_path": str(args.checkpoint_path.resolve()) if args.checkpoint_path is not None else None,
        "server_metadata": client.client.metadata,
        "tasks": [],
    }

    task_progress = tqdm(task_ids, desc="Tasks", unit="task")
    for task_id in task_progress:
        task = benchmark.get_task(task_id)
        task_progress.set_postfix(task_id=task_id, task_name=task.name[:40], refresh=False)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=args.camera_height,
            camera_widths=args.camera_width,
        )
        init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = load_init_states(init_states_path)

        successes = 0
        task_result = {
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "success_rate": 0.0,
            "episodes": [],
        }
        results["tasks"].append(task_result)
        episode_progress = tqdm(
            range(args.n_eval),
            desc=f"Task {task_id} Episodes",
            unit="ep",
            leave=False,
        )
        for episode_idx in episode_progress:
            episode_progress.set_postfix(successes=successes, refresh=False)
            client.reset()
            env.reset()
            init_state = init_states[episode_idx % len(init_states)]
            if torch.is_tensor(init_state):
                init_state = init_state.cpu().numpy()
            obs = env.set_init_state(init_state)

            dummy_action = np.zeros(7, dtype=np.float32)
            for _ in range(5):
                obs, _, _, _ = env.step(dummy_action)

            success = False
            steps = 0
            video_frames = []
            save_video = args.save_video and episode_idx < args.video_episodes_per_task
            save_video_pred = args.save_video_pred and episode_idx < args.video_episodes_per_task
            if save_video:
                video_frames.append(make_rollout_frame(obs))
            rollout_progress = tqdm(
                total=args.max_steps,
                desc=f"Task {task_id} Ep {episode_idx}",
                unit="step",
                leave=False,
            )
            while steps < args.max_steps:
                action = client.infer(obs, task.language)
                obs, _, _, _ = env.step(action)
                steps += 1
                rollout_progress.update(1)
                if save_video:
                    video_frames.append(make_rollout_frame(obs))
                if env.check_success():
                    success = True
                    break
            rollout_progress.set_postfix(success=success, steps=steps, refresh=False)
            rollout_progress.close()

            successes += int(success)
            video_path = None
            pred_video_paths = []
            if save_video and video_frames:
                video_path = (
                    args.output_dir
                    / "videos"
                    / f"task_{task_id:02d}_{task.name}"
                    / f"episode_{episode_idx:03d}.mp4"
                )
                write_rollout_video(video_frames, video_path)
            if save_video_pred and client.pred_video_chunks:
                pred_dir = (
                    args.output_dir
                    / "video_pred"
                    / f"task_{task_id:02d}_{task.name}"
                    / f"episode_{episode_idx:03d}"
                )
                for chunk_idx, chunk in enumerate(client.pred_video_chunks):
                    pred_path = pred_dir / f"request_{chunk_idx:03d}.mp4"
                    write_video_clip(chunk, pred_path)
                    pred_video_paths.append(str(pred_path))
            task_result["episodes"].append(
                {
                    "episode_index": episode_idx,
                    "success": success,
                    "steps": steps,
                    "video_path": str(video_path) if video_path is not None else None,
                    "video_pred_paths": pred_video_paths,
                }
            )
            task_result["success_rate"] = successes / float(episode_idx + 1)
            results["mean_success_rate"] = (
                float(np.mean([task["success_rate"] for task in results["tasks"]])) if results["tasks"] else 0.0
            )
            write_results(args.output_dir, results)
            episode_progress.set_postfix(
                successes=successes,
                last_success=success,
                last_steps=steps,
                refresh=False,
            )
        episode_progress.close()

        env.close()
        success_rate = successes / float(args.n_eval)
        task_result["success_rate"] = success_rate
        task_progress.set_postfix(
            task_id=task_id,
            success_rate=f"{success_rate:.3f}",
            task_name=task.name[:24],
            refresh=False,
        )
        results["mean_success_rate"] = (
            float(np.mean([task["success_rate"] for task in results["tasks"]])) if results["tasks"] else 0.0
        )
        write_results(args.output_dir, results)
    task_progress.close()

    mean_success = float(np.mean([task["success_rate"] for task in results["tasks"]])) if results["tasks"] else 0.0
    results["mean_success_rate"] = mean_success

    write_results(args.output_dir, results)

    tqdm.write(f"Mean success rate: {mean_success:.4f}")


if __name__ == "__main__":
    main()
