#!/usr/bin/env python3
"""Convert a LIBERO benchmark suite from raw HDF5 demos to a LeRobot-style dataset.

By default this script also invokes `convert_lerobot_to_gear.py` so the output
is immediately consumable by DreamZero training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    import av
except ImportError:  # pragma: no cover - depends on local environment
    av = None
    import imageio.v2 as imageio


DEFAULT_FPS = 20
CHUNK_SIZE = 1000
ANNOTATION_KEY = "annotation.language.language_instruction"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--suite-path",
        type=Path,
        default=None,
        help="Path to a LIBERO suite directory containing *_demo.hdf5 files.",
    )
    input_group.add_argument(
        "--hdf5-path",
        type=Path,
        default=None,
        help="Path to a single LIBERO *_demo.hdf5 file to convert.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output path for the converted LeRobot dataset.",
    )
    parser.add_argument(
        "--suite-name",
        type=str,
        default=None,
        help="Suite name used in metadata. Defaults to the suite-path basename.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Video FPS to write into LeRobot metadata.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Only convert the first N HDF5 files. Useful for smoke tests.",
    )
    parser.add_argument(
        "--limit-demos",
        type=int,
        default=None,
        help="Only convert the first N demos from each HDF5 file. Useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-gear-metadata",
        action="store_true",
        help="Only write the LeRobot dataset and skip DreamZero GEAR metadata generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    if av is None:
        height = int(frames.shape[1])
        width = int(frames.shape[2])
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(ffmpeg_cmd, input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True)
        return

    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"

    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")

    for frame in frames:
        frame_array[:] = frame
        packet = stream.encode(video_frame)
        if packet is not None:
            container.mux(packet)

    packet = stream.encode(None)
    if packet is not None:
        container.mux(packet)
    container.close()


def get_demo_groups(hdf5_file: h5py.File) -> list[str]:
    data_group = hdf5_file["data"]
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def extract_language(hdf5_path: Path, data_group: h5py.Group) -> str:
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

    stem = hdf5_path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ").strip()


def to_task_index(task_to_index: OrderedDict[str, int], task: str) -> int:
    if task not in task_to_index:
        task_to_index[task] = len(task_to_index)
    return task_to_index[task]


def encode_gripper_action_openvla(raw_gripper_action: np.ndarray) -> np.ndarray:
    """Match OpenVLA LIBERO action encoding: 0 = close, 1 = open.

    Raw LIBERO demos store the gripper command in the last action dimension with
    approximately -1 = open and +1 = close. OpenVLA clips this value into [0, 1]
    and inverts it so the training target becomes 1 = open, 0 = close.
    """
    return 1.0 - np.clip(raw_gripper_action, 0.0, 1.0)


def write_episode(
    demo_group: h5py.Group,
    task_index: int,
    episode_index: int,
    output_path: Path,
    fps: int,
) -> dict:
    actions = np.asarray(demo_group["actions"], dtype=np.float64)
    ee_states = np.asarray(demo_group["obs"]["ee_states"], dtype=np.float64)
    gripper_states = np.asarray(demo_group["obs"]["gripper_states"], dtype=np.float64)
    rewards = np.asarray(demo_group["rewards"], dtype=np.float64)
    dones = np.asarray(demo_group["dones"], dtype=bool)
    agentview = np.asarray(demo_group["obs"]["agentview_rgb"], dtype=np.uint8)
    eye_in_hand = np.asarray(demo_group["obs"]["eye_in_hand_rgb"], dtype=np.uint8)

    length = int(actions.shape[0])
    chunk_index = episode_index // CHUNK_SIZE
    chunk_dir = output_path / "data" / f"chunk-{chunk_index:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for video_key in VIDEO_KEYS:
        (output_path / "videos" / f"chunk-{chunk_index:03d}" / f"observation.images.{video_key}").mkdir(
            parents=True, exist_ok=True
        )

    state = np.concatenate([ee_states, gripper_states], axis=-1)
    pose_delta = actions[:, :6]
    gripper_action = encode_gripper_action_openvla(actions[:, 6:7])
    packed_action = np.concatenate([pose_delta, gripper_action], axis=-1)

    episode_df = pd.DataFrame(
        {
            STATE_KEY: state.tolist(),
            ACTION_KEY: packed_action.tolist(),
            "next.reward": rewards,
            "next.done": dones,
            "is_terminal": dones,
            "is_first": np.arange(length) == 0,
            "discount": np.where(dones, 0.0, 1.0),
            "timestamp": np.arange(length, dtype=np.float64) / float(fps),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
            ANNOTATION_KEY: np.full(length, task_index, dtype=np.int64),
        }
    )
    parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
    episode_df.to_parquet(parquet_path, engine="pyarrow")

    encode_video(
        agentview,
        output_path
        / "videos"
        / f"chunk-{chunk_index:03d}"
        / "observation.images.agentview_rgb"
        / f"episode_{episode_index:06d}.mp4",
        fps,
    )
    encode_video(
        eye_in_hand,
        output_path
        / "videos"
        / f"chunk-{chunk_index:03d}"
        / "observation.images.eye_in_hand_rgb"
        / f"episode_{episode_index:06d}.mp4",
        fps,
    )

    return {
        "episode_index": episode_index,
        "tasks": [task_index],
        "length": length,
        "success": bool(np.any(rewards > 0)),
    }


def write_metadata(
    output_path: Path,
    suite_name: str,
    fps: int,
    total_frames: int,
    episodes: list[dict],
    tasks: OrderedDict[str, int],
) -> None:
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as file:
        for task, index in tasks.items():
            file.write(json.dumps({"task_index": index, "task": task}) + "\n")

    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w", encoding="utf-8") as file:
        for episode in episodes:
            task_strings = [
                task for task, index in tasks.items() if index in episode["tasks"]
            ]
            payload = dict(episode)
            payload["tasks"] = task_strings
            file.write(json.dumps(payload) + "\n")

    num_chunks = (len(episodes) + CHUNK_SIZE - 1) // CHUNK_SIZE
    info = {
        "codebase_version": "v2.0",
        "robot_type": "libero",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(VIDEO_KEYS),
        "total_chunks": num_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.agentview_rgb": {
                "dtype": "video",
                "shape": [128, 128, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.eye_in_hand_rgb": {
                "dtype": "video",
                "shape": [128, 128, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            STATE_KEY: {
                "dtype": "float64",
                "shape": [8],
                "names": ["eef_state", "gripper_state"],
            },
            ACTION_KEY: {
                "dtype": "float64",
                "shape": [7],
                "names": ["pose_delta", "gripper_position"],
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "is_terminal": {"dtype": "bool", "shape": [1]},
            "is_first": {"dtype": "bool", "shape": [1]},
            "discount": {"dtype": "float64", "shape": [1]},
            ANNOTATION_KEY: {"dtype": "int64", "shape": [1]},
        },
        "libero_suite": suite_name,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)


def run_gear_conversion(output_path: Path) -> None:
    script_path = Path(__file__).with_name("convert_lerobot_to_gear.py")
    command = [
        sys.executable,
        str(script_path),
        "--dataset-path",
        str(output_path),
        "--embodiment-tag",
        "libero_sim",
        "--state-keys",
        '{"eef_state": [0, 6], "gripper_state": [6, 8]}',
        "--action-keys",
        '{"pose_delta": [0, 6], "gripper_position": [6, 7]}',
        "--task-key",
        ANNOTATION_KEY,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_path = args.output_path.resolve()
    hdf5_paths: list[Path]

    if args.hdf5_path is not None:
        hdf5_path = args.hdf5_path.resolve()
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 path does not exist: {hdf5_path}")
        hdf5_paths = [hdf5_path]
        inferred_suite_name = hdf5_path.stem
        if inferred_suite_name.endswith("_demo"):
            inferred_suite_name = inferred_suite_name[: -len("_demo")]
    else:
        suite_path = args.suite_path.resolve()
        if not suite_path.exists():
            raise FileNotFoundError(f"Suite path does not exist: {suite_path}")
        hdf5_paths = sorted(suite_path.glob("*_demo.hdf5"))
        inferred_suite_name = suite_path.name

    suite_name = args.suite_name or inferred_suite_name

    if output_path.exists():
        if not args.force:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)

    if args.limit_files is not None:
        hdf5_paths = hdf5_paths[: args.limit_files]
    if not hdf5_paths:
        if args.suite_path is not None:
            raise FileNotFoundError(f"No *_demo.hdf5 files found under {args.suite_path.resolve()}")
        raise FileNotFoundError("No input HDF5 files found to convert.")

    task_to_index: OrderedDict[str, int] = OrderedDict()
    episodes: list[dict] = []
    total_frames = 0
    episode_index = 0

    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as hdf5_file:
            data_group = hdf5_file["data"]
            task = extract_language(hdf5_path, data_group)
            task_index = to_task_index(task_to_index, task)

            demo_groups = get_demo_groups(hdf5_file)
            if args.limit_demos is not None:
                demo_groups = demo_groups[: args.limit_demos]

            for demo_name in demo_groups:
                demo_group = data_group[demo_name]
                episode = write_episode(
                    demo_group=demo_group,
                    task_index=task_index,
                    episode_index=episode_index,
                    output_path=output_path,
                    fps=args.fps,
                )
                total_frames += episode["length"]
                episodes.append(episode)
                episode_index += 1

    write_metadata(
        output_path=output_path,
        suite_name=suite_name,
        fps=args.fps,
        total_frames=total_frames,
        episodes=episodes,
        tasks=task_to_index,
    )

    if not args.skip_gear_metadata:
        run_gear_conversion(output_path)

    print(f"Converted {len(episodes)} episodes from {suite_name} into {output_path}")


if __name__ == "__main__":
    main()
