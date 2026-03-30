#!/usr/bin/env python3
"""Render a teacher-forced LIBERO trace as an annotated MP4 over the demo video."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", type=Path, required=True, help="Path to demo_*.jsonl trace file.")
    parser.add_argument("--demo-file", type=Path, required=True, help="Path to the LIBERO HDF5 demo file.")
    parser.add_argument("--output-path", type=Path, default=None, help="Output MP4 path.")
    parser.add_argument("--demo-id", type=int, default=None, help="Optional explicit demo id. Defaults to parsing from trace filename.")
    parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
    parser.add_argument("--panel-height", type=int, default=180, help="Bottom annotation panel height.")
    return parser.parse_args()


def infer_demo_id(trace_file: Path) -> int:
    match = re.search(r"demo_(\d+)\.jsonl$", trace_file.name)
    if not match:
        raise ValueError(f"Could not infer demo id from trace filename: {trace_file}")
    return int(match.group(1))


def load_trace(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_gripper_label(value: float) -> str:
    if value > 0:
        return "close (+1)"
    if value < 0:
        return "open (-1)"
    return "neutral (0)"


def normalize_bar_value(value: float, clamp: float = 1.0) -> float:
    if clamp <= 0:
        return 0.0
    return float(np.clip(value / clamp, -1.0, 1.0))


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    gt_value: float,
    pred_value: float,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    bg = (35, 38, 46)
    fg = (120, 128, 140)
    gt_color = (90, 200, 140)
    pred_color = (255, 170, 70)

    draw.rectangle((x, y, x + w, y + h), outline=fg, fill=bg)
    center_x = x + w // 2
    draw.line((center_x, y, center_x, y + h), fill=fg, width=1)

    def _draw_value(value: float, color: tuple[int, int, int], band_top: int, band_bottom: int) -> None:
        bar_value = normalize_bar_value(value)
        end_x = center_x + int(bar_value * (w // 2 - 4))
        left = min(center_x, end_x)
        right = max(center_x, end_x)
        draw.rectangle((left, band_top, right, band_bottom), fill=color)

    _draw_value(gt_value, gt_color, y + 6, y + h // 2 - 2)
    _draw_value(pred_value, pred_color, y + h // 2 + 2, y + h - 6)
    draw.text((x, y - 16), label, fill=(230, 230, 230), font=font)
    draw.text((x + 4, y + 4), "GT", fill=gt_color, font=font)
    draw.text((x + 4, y + h // 2 + 2), "Pred", fill=pred_color, font=font)


def make_dataset_frame(agentview: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    agentview = np.flipud(np.asarray(agentview, dtype=np.uint8)).copy()
    wrist = np.flipud(np.asarray(wrist, dtype=np.uint8)).copy()
    if wrist.shape[0] != agentview.shape[0]:
        row_idx = np.linspace(0, wrist.shape[0] - 1, agentview.shape[0]).astype(np.int64)
        target_width = max(1, int(round(wrist.shape[1] * agentview.shape[0] / wrist.shape[0])))
        col_idx = np.linspace(0, wrist.shape[1] - 1, target_width).astype(np.int64)
        wrist = wrist[row_idx][:, col_idx]
    return np.concatenate([agentview, wrist], axis=1)


def render_frame(
    frame: np.ndarray,
    row: dict,
    step_index: int,
    total_steps: int,
    panel_height: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    h, w, _ = frame.shape
    canvas = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    canvas[:h] = frame
    canvas[h:] = np.array([18, 20, 26], dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    draw.text(
        (12, h + 10),
        f"step {step_index:03d}/{total_steps - 1:03d} | request {row['request_index']:03d} | "
        f"query={row['query_step']} | chunk_idx={row['action_index_in_chunk']} | "
        f"pose_l2={row['pose_l2']:.3f} | action_l2={row['action_l2']:.3f}",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text(
        (12, h + 34),
        f"GT gripper:   {format_gripper_label(float(row['gt_gripper']))}",
        fill=(90, 200, 140),
        font=font,
    )
    draw.text(
        (12, h + 52),
        f"Pred gripper: {format_gripper_label(float(row['pred_gripper_eval']))} | "
        f"raw={float(row['pred_gripper_raw']):+.4f} | sign_match={int(row['gripper_sign_match'])}",
        fill=(255, 170, 70),
        font=font,
    )

    bar_y = h + 84
    bar_w = max(140, w // 4 - 12)
    bar_gap = 12
    left_margin = 12

    draw_bar(draw, left_margin + 0 * (bar_w + bar_gap), bar_y, bar_w, 54, float(row["gt_pose_x"]), float(row["pred_pose_x"]), "pose_x", font)
    draw_bar(draw, left_margin + 1 * (bar_w + bar_gap), bar_y, bar_w, 54, float(row["gt_pose_y"]), float(row["pred_pose_y"]), "pose_y", font)
    draw_bar(draw, left_margin + 2 * (bar_w + bar_gap), bar_y, bar_w, 54, float(row["gt_pose_z"]), float(row["pred_pose_z"]), "pose_z", font)

    draw.text(
        (12, h + 148),
        f"pred_abs_max={float(row['pred_action_abs_max']):.3f} | "
        f"nan={bool(row['pred_action_has_nan'])} | inf={bool(row['pred_action_has_inf'])}",
        fill=(220, 220, 220),
        font=font,
    )
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    trace_rows = load_trace(args.trace_file)
    if not trace_rows:
        raise ValueError(f"Trace file is empty: {args.trace_file}")

    demo_id = args.demo_id if args.demo_id is not None else infer_demo_id(args.trace_file)
    demo_name = f"demo_{demo_id}"
    output_path = (
        args.output_path
        if args.output_path is not None
        else args.trace_file.with_suffix(".mp4")
    )

    with h5py.File(args.demo_file, "r") as h5_file:
        if demo_name not in h5_file["data"]:
            raise KeyError(f"{demo_name} not found in {args.demo_file}")
        demo_group = h5_file["data"][demo_name]
        agentview = np.asarray(demo_group["obs"]["agentview_rgb"][()], dtype=np.uint8)
        wrist = np.asarray(demo_group["obs"]["eye_in_hand_rgb"][()], dtype=np.uint8)

    total_steps = min(len(trace_rows), len(agentview), len(wrist))
    font = ImageFont.load_default()
    frames = []
    for step_index in range(total_steps):
        video_frame = make_dataset_frame(agentview[step_index], wrist[step_index])
        rendered = render_frame(
            frame=video_frame,
            row=trace_rows[step_index],
            step_index=step_index,
            total_steps=total_steps,
            panel_height=args.panel_height,
            font=font,
        )
        frames.append(rendered)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=args.fps, codec="libx264")
    print(f"Saved annotated trace video to {output_path}")


if __name__ == "__main__":
    main()
