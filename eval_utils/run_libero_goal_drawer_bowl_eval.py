#!/usr/bin/env python3
"""Evaluate DreamZero on the single LIBERO-Goal task: open the top drawer and put the bowl inside."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dreamzero.eval_utils.run_libero_eval import (
    DEFAULT_LIBERO_ROOT,
    DreamZeroLiberoClient,
    ensure_libero_imports,
    load_init_states,
    make_rollout_frame,
    write_results,
    write_rollout_video,
    write_video_clip,
)


TARGET_BENCHMARK_NAME = "libero_goal"
TARGET_TASK_ID = 3
TARGET_TASK_NAME = "open_the_top_drawer_and_put_the_bowl_inside"
TARGET_TASK_LANGUAGE = "open the top drawer and put the bowl inside"
TARGET_DEMO_FILENAME = "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="localhost", help="DreamZero policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="DreamZero policy server port.")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="Path to the local LIBERO repo.")
    parser.add_argument("--task-order-index", type=int, default=0, help="Task order index for LIBERO benchmark.")
    parser.add_argument("--n-eval", type=int, default=20, help="Episodes to evaluate.")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum rollout length.")
    parser.add_argument("--camera-height", type=int, default=128, help="Render camera height.")
    parser.add_argument("--camera-width", type=int, default=128, help="Render camera width.")
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=8,
        help="How many predicted actions to execute per server call.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=25,
        help="How many recent frames to send per policy request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/libero_goal_drawer_bowl_eval"),
        help="Directory for JSON/CSV/video results.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional checkpoint path recorded in results.json.")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos.")
    parser.add_argument(
        "--save-video-pred",
        action="store_true",
        help="Save decoded DreamZero internal video_pred clips for the first few episodes.",
    )
    parser.add_argument(
        "--debug-open-loop",
        action="store_true",
        help="Print detailed client-side request/reuse logs.",
    )
    parser.add_argument(
        "--video-episodes-per-task",
        type=int,
        default=1,
        help="How many episodes to save when --save-video is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_libero_imports(args.libero_root)

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(TARGET_BENCHMARK_NAME)(args.task_order_index)
    task = benchmark.get_task(TARGET_TASK_ID)
    demo_path = benchmark.get_task_demonstration(TARGET_TASK_ID)

    if task.name != TARGET_TASK_NAME:
        raise RuntimeError(
            f"Expected task_id={TARGET_TASK_ID} to be {TARGET_TASK_NAME}, got {task.name}."
        )
    if task.language != TARGET_TASK_LANGUAGE:
        raise RuntimeError(
            f"Expected task language '{TARGET_TASK_LANGUAGE}', got '{task.language}'."
        )
    if not demo_path.endswith(TARGET_DEMO_FILENAME):
        raise RuntimeError(
            f"Expected demo file ending with {TARGET_DEMO_FILENAME}, got {demo_path}."
        )

    client = DreamZeroLiberoClient(
        args.host,
        args.port,
        open_loop_horizon=args.open_loop_horizon,
        history_frames=args.history_frames,
        debug_open_loop=args.debug_open_loop,
        return_video_pred=args.save_video_pred,
    )

    tqdm.write(
        f"[eval][start] benchmark={TARGET_BENCHMARK_NAME} task_id={TARGET_TASK_ID} "
        f"task_name={task.name} n_eval={args.n_eval} max_steps={args.max_steps} "
        f"open_loop_horizon={args.open_loop_horizon} history_frames={args.history_frames}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark_name": TARGET_BENCHMARK_NAME,
        "task_order_index": args.task_order_index,
        "task_id": TARGET_TASK_ID,
        "task_name": task.name,
        "language": task.language,
        "demo_file": demo_path,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "open_loop_horizon": args.open_loop_horizon,
        "history_frames": args.history_frames,
        "checkpoint_path": str(args.checkpoint_path.resolve()) if args.checkpoint_path is not None else None,
        "server_metadata": client.client.metadata,
        "episodes": [],
        "success_rate": 0.0,
    }

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = load_init_states(init_states_path)

    successes = 0
    episode_progress = tqdm(
        range(args.n_eval),
        desc=f"Task {TARGET_TASK_ID} Episodes",
        unit="ep",
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
            desc=f"Task {TARGET_TASK_ID} Ep {episode_idx}",
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
                / f"task_{TARGET_TASK_ID:02d}_{task.name}"
                / f"episode_{episode_idx:03d}.mp4"
            )
            write_rollout_video(video_frames, video_path)
        if save_video_pred and client.pred_video_chunks:
            pred_dir = (
                args.output_dir
                / "video_pred"
                / f"task_{TARGET_TASK_ID:02d}_{task.name}"
                / f"episode_{episode_idx:03d}"
            )
            for chunk_idx, chunk in enumerate(client.pred_video_chunks):
                pred_path = pred_dir / f"request_{chunk_idx:03d}.mp4"
                write_video_clip(chunk, pred_path)
                pred_video_paths.append(str(pred_path))

        results["episodes"].append(
            {
                "episode_index": episode_idx,
                "success": success,
                "steps": steps,
                "video_path": str(video_path) if video_path is not None else None,
                "video_pred_paths": pred_video_paths,
            }
        )
        results["success_rate"] = successes / float(episode_idx + 1)
        write_results(args.output_dir, {"tasks": [results], "mean_success_rate": results["success_rate"], **results})
        episode_progress.set_postfix(
            successes=successes,
            last_success=success,
            last_steps=steps,
            refresh=False,
        )

    episode_progress.close()
    env.close()

    results["success_rate"] = successes / float(args.n_eval)
    payload = {
        "benchmark_name": TARGET_BENCHMARK_NAME,
        "task_order_index": args.task_order_index,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "open_loop_horizon": args.open_loop_horizon,
        "history_frames": args.history_frames,
        "checkpoint_path": results["checkpoint_path"],
        "server_metadata": results["server_metadata"],
        "tasks": [
            {
                "task_id": TARGET_TASK_ID,
                "task_name": task.name,
                "language": task.language,
                "success_rate": results["success_rate"],
                "episodes": results["episodes"],
            }
        ],
        "mean_success_rate": results["success_rate"],
        "demo_file": demo_path,
    }
    write_results(args.output_dir, payload)
    with open(args.output_dir / "task_info.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "task_id": TARGET_TASK_ID,
                "task_name": task.name,
                "language": task.language,
                "demo_file": demo_path,
            },
            file,
            indent=2,
        )
    tqdm.write(f"Single-task success rate: {results['success_rate']:.4f}")


if __name__ == "__main__":
    main()
