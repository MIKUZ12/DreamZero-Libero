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

    def _convert_observation(self, obs: dict) -> dict:
        return {
            "video.agentview_rgb": np.asarray(obs["observation/exterior_image_0_left"], dtype=np.uint8)[None, ...],
            "video.eye_in_hand_rgb": np.asarray(obs["observation/wrist_image_left"], dtype=np.uint8)[None, ...],
            "state.joint_position": np.asarray(obs["observation/joint_position"], dtype=np.float64)[None, ...],
            "state.gripper_position": np.asarray(obs["observation/gripper_position"], dtype=np.float64)[None, ...],
            "annotation.language.language_instruction": obs.get("prompt", ""),
        }

    def _forward(self, obs: dict):
        converted = self._convert_observation(obs)
        with torch.no_grad():
            result_batch, _ = self._policy.lazy_joint_forward_causal(Batch(obs=converted))
            return result_batch

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

        gripper = np.where(gripper >= 0, 1.0, -1.0).astype(np.float32)
        actions = np.concatenate([pose_delta, gripper], axis=-1)
        return {"actions": actions}

    def infer(self, obs: dict) -> dict:
        action_head = self._policy.trained_model.action_head
        if self._debug_open_loop and getattr(action_head, "ip_rank", 0) == 0:
            print(
                f"[server][infer] session={obs.get('session_id', 'unknown')} "
                f"request={obs.get('client_request_index', 'unknown')} "
                f"env_step={obs.get('client_env_step_index', 'unknown')} "
                f"open_loop_horizon={obs.get('client_open_loop_horizon', 'unknown')}"
            )
        result_batch = self._forward(obs)
        formatted = self._format_actions(result_batch.act)
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
    parser.add_argument("--max-chunk-size", type=int, default=None, help="Optional inference override.")
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

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device=device,
        model_config_overrides=(
            [f"max_chunk_size={args.max_chunk_size}"] if args.max_chunk_size is not None else []
        ),
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
