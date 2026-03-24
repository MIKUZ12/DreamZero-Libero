#!/usr/bin/env python3
"""Replay actions from a LIBERO HDF5 demo file inside the simulator and save rollout videos."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LIBERO_ROOT = Path(__file__).resolve().parents[2] / "LIBERO"


def ensure_libero_imports(libero_root: Path) -> None:
    libero_root = libero_root.resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def make_rollout_frame(obs: dict) -> np.ndarray:
    agentview = np.flipud(np.asarray(obs["agentview_image"], dtype=np.uint8)).copy()
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)).copy()
    if wrist.shape[0] != agentview.shape[0]:
        row_idx = np.linspace(0, wrist.shape[0] - 1, agentview.shape[0]).astype(np.int64)
        target_width = max(1, int(round(wrist.shape[1] * agentview.shape[0] / wrist.shape[0])))
        col_idx = np.linspace(0, wrist.shape[1] - 1, target_width).astype(np.int64)
        wrist = wrist[row_idx][:, col_idx]
    return np.concatenate([agentview, wrist], axis=1)


def write_rollout_video(frames: list[np.ndarray], output_path: Path, fps: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-file", type=Path, required=True, help="Path to LIBERO HDF5 demo file.")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="Path to local LIBERO repo.")
    parser.add_argument("--benchmark-name", type=str, default="libero_goal", help="LIBERO benchmark name.")
    parser.add_argument("--task-order-index", type=int, default=0, help="Task order index used to build benchmark.")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Optional task id. If omitted, infer it by matching the demo filename inside the benchmark.",
    )
    parser.add_argument(
        "--demo-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional numeric demo ids to replay, e.g. --demo-ids 0 1 2. Default replays all demos.",
    )
    parser.add_argument("--camera-height", type=int, default=128, help="Render camera height.")
    parser.add_argument("--camera-width", type=int, default=128, help="Render camera width.")
    parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/libero_dataset_replay"),
        help="Directory for replay videos and summary JSON.",
    )
    return parser.parse_args()


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


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
    """Rewrite asset file paths embedded in demo XML to the local LIBERO checkout."""
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


def main() -> None:
    args = parse_args()
    ensure_libero_imports(args.libero_root)

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import utils as libero_utils

    benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
    task_id = args.task_id if args.task_id is not None else infer_task_id(benchmark, args.demo_file)
    task = benchmark.get_task(task_id)

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "demo_file": str(args.demo_file.resolve()),
        "benchmark_name": args.benchmark_name,
        "task_order_index": args.task_order_index,
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "fps": args.fps,
        "episodes": [],
    }

    with h5py.File(args.demo_file, "r") as demo_h5:
        demo_names = sorted(demo_h5["data"].keys(), key=_demo_sort_key)
        if args.demo_ids is not None:
            wanted = {f"demo_{demo_id}" for demo_id in args.demo_ids}
            demo_names = [name for name in demo_names if name in wanted]
        if not demo_names:
            raise RuntimeError("No demos selected for replay.")

        progress = tqdm(demo_names, desc="Replaying demos", unit="demo")
        for episode_index, demo_name in enumerate(progress):
            demo_group = demo_h5["data"][demo_name]
            actions = np.asarray(demo_group["actions"][()], dtype=np.float32)
            states = np.asarray(demo_group["states"][()], dtype=np.float64)
            model_xml = remap_demo_model_xml(demo_group.attrs["model_file"], args.libero_root)
            model_xml = libero_utils.postprocess_model_xml(model_xml, {})

            env.reset()
            env.reset_from_xml_string(model_xml)
            env.sim.reset()

            frames: list[np.ndarray] = []
            divergence_errors: list[float] = []

            # Record the reconstructed initial state for reference.
            obs = env.set_init_state(states[0])
            frames.append(make_rollout_frame(obs))

            for step_index, action in enumerate(actions):
                obs, _, _, _ = env.step(action)
                frames.append(make_rollout_frame(obs))

                if step_index < len(states) - 1:
                    replay_state = env.sim.get_state().flatten()
                    divergence_errors.append(float(np.linalg.norm(states[step_index + 1] - replay_state)))

            video_path = args.output_dir / "videos" / f"{demo_name}.mp4"
            write_rollout_video(frames, video_path, fps=args.fps)

            summary["episodes"].append(
                {
                    "demo_name": demo_name,
                    "num_actions": int(actions.shape[0]),
                    "video_path": str(video_path),
                    "mean_state_error": float(np.mean(divergence_errors)) if divergence_errors else 0.0,
                    "max_state_error": float(np.max(divergence_errors)) if divergence_errors else 0.0,
                }
            )

    env.close()

    with open(args.output_dir / "replay_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    tqdm.write(
        f"Saved {len(summary['episodes'])} replay video(s) to {(args.output_dir / 'videos').resolve()}"
    )


if __name__ == "__main__":
    main()
