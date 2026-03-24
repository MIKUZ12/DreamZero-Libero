#!/usr/bin/env python3
"""Serve DreamZero as a websocket policy for LIBERO evaluation in a separate environment."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import logging
import os
import pickle
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
import websockets
import websockets.asyncio.server
import websockets.frames
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

try:
    import robosuite.utils.transform_utils as T
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    T = None


@dataclasses.dataclass
class ServerMetadata:
    embodiment: str = "libero_sim"
    action_space: str = "osc_pose"
    expected_views: int = 2


INFER_SIGNAL = 0
SHUTDOWN_SIGNAL = 1
RESET_SIGNAL = 2


class LiberoDreamZeroPolicy:
    def __init__(self, policy: GrootSimPolicy) -> None:
        self._policy = policy
        self._debug_open_loop = os.environ.get("DREAMZERO_DEBUG_OPEN_LOOP", "0") == "1"

    def reset(self, payload: dict) -> None:
        action_head = self._policy.trained_model.action_head
        action_head.current_start_frame = 0
        action_head.language = None
        action_head.clip_feas = None
        action_head.ys = None
        action_head.kv_cache1 = None
        action_head.kv_cache_neg = None
        action_head.crossattn_cache = None
        action_head.crossattn_cache_neg = None
        if self._debug_open_loop and getattr(action_head, "ip_rank", 0) == 0:
            print(
                f"[server][reset] session={payload.get('session_id', 'unknown')} "
                f"reason=client_reset"
            )

    def _as_video(self, value) -> np.ndarray:
        video = np.asarray(value, dtype=np.uint8)
        if video.ndim == 3:
            return video[None, ...]
        if video.ndim == 4:
            return video
        raise ValueError(f"Expected video input with 3 or 4 dims, got shape {video.shape}")

    def _as_state(self, value) -> np.ndarray:
        state = np.asarray(value, dtype=np.float64)
        if state.ndim == 1:
            return state[None, ...]
        if state.ndim == 2:
            return state
        raise ValueError(f"Expected state input with 1 or 2 dims, got shape {state.shape}")

    def _quat_to_axis_angle(self, quat) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        if quat.ndim == 1:
            quat = quat[None, ...]
        if quat.shape[-1] != 4:
            raise ValueError(f"Expected quaternion with last dim 4, got shape {quat.shape}")
        if T is not None:
            return np.stack([T.quat2axisangle(q) for q in quat], axis=0)

        xyz = quat[..., :3]
        w = quat[..., 3:4]
        xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
        safe_xyz_norm = np.where(xyz_norm < 1e-12, 1.0, xyz_norm)
        angle = 2.0 * np.arctan2(xyz_norm, w)
        axis = xyz / safe_xyz_norm
        axis_angle = axis * angle
        axis_angle = np.where(xyz_norm < 1e-12, 0.0, axis_angle)
        return axis_angle

    def _convert_observation(self, obs: dict) -> dict:
        if "observation/ee_state" in obs:
            eef_state = self._as_state(obs["observation/ee_state"])
        else:
            eef_pos = self._as_state(obs["observation/eef_pos"])
            eef_quat = self._as_state(obs["observation/eef_quat"])
            eef_axis_angle = self._quat_to_axis_angle(eef_quat)
            eef_state = np.concatenate([eef_pos, eef_axis_angle], axis=-1)
        return {
            "video.agentview_rgb": self._as_video(obs["observation/exterior_image_0_left"]),
            "video.eye_in_hand_rgb": self._as_video(obs["observation/wrist_image_left"]),
            "state.eef_state": eef_state,
            "state.gripper_state": self._as_state(obs["observation/gripper_position"]),
            "annotation.language.language_instruction": obs.get("prompt", ""),
        }

    def _forward(self, obs: dict):
        converted = self._convert_observation(obs)
        with torch.no_grad():
            result_batch, _ = self._policy.lazy_joint_forward_causal(Batch(obs=converted))
            return result_batch

    def _forward_with_video(self, obs: dict):
        converted = self._convert_observation(obs)
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(Batch(obs=converted))
            return result_batch, video_pred

    def _format_actions(self, action_dict) -> dict:
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
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        elif gripper.ndim == 0:
            gripper = gripper.reshape(1, 1)

        # Match OpenVLA LIBERO action decoding:
        # training target uses 0 = close, 1 = open, while the simulator expects
        # +1 = close and -1 = open.
        gripper = 2.0 * gripper - 1.0
        gripper = np.sign(gripper)
        gripper = -gripper.astype(np.float32)
        actions = np.concatenate([pose_delta, gripper], axis=-1)
        return {"actions": actions}

    def _decode_video_pred(self, video_pred: torch.Tensor) -> np.ndarray:
        action_head = self._policy.trained_model.action_head
        vae = action_head.vae
        with torch.no_grad():
            decoded = vae.decode(video_pred.to(device=action_head._device, dtype=torch.bfloat16))
        decoded = decoded.detach().float().cpu()
        if decoded.ndim != 5:
            raise ValueError(f"Expected decoded video to have 5 dims [B, C, T, H, W], got {decoded.shape}")
        decoded = decoded[0].permute(1, 2, 3, 0).numpy()
        decoded = ((decoded + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        return decoded

    def infer(self, obs: dict) -> dict:
        action_head = self._policy.trained_model.action_head
        if self._debug_open_loop and getattr(action_head, "ip_rank", 0) == 0:
            print(
                f"[server][infer] session={obs.get('session_id', 'unknown')} "
                f"request={obs.get('client_request_index', 'unknown')} "
                f"env_step={obs.get('client_env_step_index', 'unknown')} "
                f"open_loop_horizon={obs.get('client_open_loop_horizon', 'unknown')}"
            )
        request_video_pred = bool(obs.get("return_video_pred", False))
        if request_video_pred:
            result_batch, video_pred = self._forward_with_video(obs)
        else:
            result_batch = self._forward(obs)
            video_pred = None
        formatted = self._format_actions(result_batch.act)
        if request_video_pred and video_pred is not None:
            formatted["video_pred"] = self._decode_video_pred(video_pred)
        if self._debug_open_loop and getattr(action_head, "ip_rank", 0) == 0:
            print(
                f"[server][response] session={obs.get('session_id', 'unknown')} "
                f"request={obs.get('client_request_index', 'unknown')} "
                f"action_chunk_shape={formatted['actions'].shape}"
            )
        return formatted

    def participate(self, obs: dict) -> None:
        self._forward(obs)


class PicklePolicyServer:
    def __init__(
        self,
        policy: LiberoDreamZeroPolicy,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        signal_group: dist.ProcessGroup | None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._rank = rank
        self._world_size = world_size
        self._signal_group = signal_group
        if torch.cuda.is_available():
            self._broadcast_device = torch.device("cuda", torch.cuda.current_device())
        else:
            self._broadcast_device = torch.device("cpu")

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        if self._rank == 0:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
            ) as server:
                await server.serve_forever()
        else:
            await self._worker_loop()

    def _broadcast_signal(self, signal: int) -> None:
        if self._world_size == 1:
            return
        signal_tensor = torch.tensor([signal], dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

    def _broadcast_payload(self, payload: dict) -> None:
        if self._world_size == 1:
            return

        serialized = pickle.dumps(payload)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device=self._broadcast_device)
        dist.broadcast(size_tensor, src=0)

        data_tensor = torch.tensor(list(serialized), dtype=torch.uint8, device=self._broadcast_device)
        dist.broadcast(data_tensor, src=0)

    def _receive_payload(self) -> dict:
        size_tensor = torch.zeros(1, dtype=torch.int64, device=self._broadcast_device)
        dist.broadcast(size_tensor, src=0)
        data_size = int(size_tensor.item())

        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device=self._broadcast_device)
        dist.broadcast(data_tensor, src=0)
        return pickle.loads(data_tensor.cpu().numpy().tobytes())

    def _distributed_reset(self, payload: dict) -> None:
        if self._world_size > 1:
            self._broadcast_signal(RESET_SIGNAL)
            self._broadcast_payload(payload)
        self._policy.reset(payload)

    def _distributed_infer(self, payload: dict) -> dict:
        if self._world_size == 1:
            return self._policy.infer(payload)

        self._broadcast_signal(INFER_SIGNAL)
        self._broadcast_payload(payload)
        dist.barrier()
        result = self._policy.infer(payload)
        dist.barrier()
        return result

    async def _worker_loop(self) -> None:
        logging.info("Rank %d entering distributed worker loop", self._rank)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")

        while True:
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            signal = int(signal_tensor.item())
            if signal == SHUTDOWN_SIGNAL:
                logging.info("Rank %d received shutdown signal", self._rank)
                break

            payload = self._receive_payload()
            if signal == RESET_SIGNAL:
                self._policy.reset(payload)
                continue

            if signal != INFER_SIGNAL:
                raise RuntimeError(f"Unsupported distributed signal: {signal}")

            dist.barrier()
            self._policy.participate(payload)
            dist.barrier()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        await websocket.send(pickle.dumps(dataclasses.asdict(ServerMetadata())))
        while True:
            try:
                payload = pickle.loads(await websocket.recv())
                endpoint = payload.pop("endpoint")
                if endpoint == "reset":
                    self._distributed_reset(payload)
                    await websocket.send(pickle.dumps({"status": "reset successful"}))
                elif endpoint == "infer":
                    action = self._distributed_infer(payload)
                    await websocket.send(pickle.dumps(action))
                else:
                    raise ValueError(f"Unsupported endpoint: {endpoint}")
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True, help="DreamZero checkpoint path.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--max-chunk-size", type=int, default=None, help="Optional diffusion local-attention chunk override.")
    parser.add_argument(
        "--num-frame-per-block",
        type=int,
        default=None,
        help="Optional inference override for temporal block size. This changes memory/latency trade-offs.",
    )
    parser.add_argument(
        "--metadata-dataset-path",
        type=str,
        default=None,
        help="Optional converted dataset root used to build metadata for embodiments missing from checkpoint metadata.",
    )
    parser.add_argument(
        "--debug-open-loop",
        action="store_true",
        help="Enable verbose server-side request/response logging for debugging open-loop behavior.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=43200, help="Distributed control-plane timeout.")
    return parser.parse_args()


def init_runtime(device_arg: str, timeout_seconds: int) -> tuple[str, int, int, DeviceMesh | None, dist.ProcessGroup | None]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if dist.is_initialized():
        rank = dist.get_rank()
    elif world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        rank = 0

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-GPU mode requires CUDA.")
        torch.cuda.set_device(local_rank)
        device = "cuda"
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("ip",),
        )
        signal_group = dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        return device, rank, world_size, device_mesh, signal_group

    return device_arg, rank, 1, None, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    os.environ["DREAMZERO_DEBUG_OPEN_LOOP"] = "1" if args.debug_open_loop else "0"
    os.environ["DREAMZERO_DEBUG_INFER"] = "1" if args.debug_open_loop else "0"
    if "," in args.device:
        raise ValueError(
            "--device only accepts a single device. For multi-GPU use torchrun and omit the comma list."
        )

    # Match DreamZero's original autoregressive sim-eval path: lazy causal inference
    # produces several valid tensor shapes during cache warmup / sampling, so the
    # default torch._dynamo recompile limit is too small for this workload.
    torch._dynamo.config.recompile_limit = 800
    torch._dynamo.config.cache_size_limit = 1000

    device, rank, world_size, device_mesh, signal_group = init_runtime(
        device_arg=args.device,
        timeout_seconds=args.timeout_seconds,
    )
    logging.info("Initialized rank %d/%d on device %s", rank, world_size, device)

    model_config_overrides: list[str] = []
    if args.max_chunk_size is not None:
        # max_chunk_size lives under diffusion_model_cfg in checkpoint config.
        model_config_overrides.append(
            f"action_head_cfg.config.diffusion_model_cfg.max_chunk_size={args.max_chunk_size}"
        )
    if args.num_frame_per_block is not None:
        # Keep action_head and diffusion_model_cfg in sync.
        model_config_overrides.append(
            f"action_head_cfg.config.num_frame_per_block={args.num_frame_per_block}"
        )
        model_config_overrides.append(
            f"action_head_cfg.config.diffusion_model_cfg.num_frame_per_block={args.num_frame_per_block}"
        )

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device=device,
        model_config_overrides=model_config_overrides,
        device_mesh=device_mesh,
        metadata_dataset_path=args.metadata_dataset_path,
    )
    server = PicklePolicyServer(
        LiberoDreamZeroPolicy(policy),
        host=args.host,
        port=args.port,
        rank=rank,
        world_size=world_size,
        signal_group=signal_group,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
