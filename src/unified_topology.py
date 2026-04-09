from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from pytorch_lightning.strategies import FSDPStrategy


@dataclass
class TopologyConfig:
    strategy: str = "local"
    mesh_axis_name: str = "data"


class TopologyManager:
    """Single source of truth for JAX/PyTorch device and strategy initialization."""

    def __init__(self, config: TopologyConfig | None = None):
        self.config = config or TopologyConfig()
        self.n_devices = jax.device_count()

        self.mesh = Mesh(np.array(jax.devices()), (self.config.mesh_axis_name,))
        self.sharding = NamedSharding(self.mesh, PartitionSpec(self.config.mesh_axis_name))

        self.trainer_strategy = "auto"
        if self.config.strategy == "zero3" and self.n_devices > 1:
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            self.trainer_strategy = FSDPStrategy(auto_wrap_policy=None, cpu_offload=True)

    def shard_array(self, arr: jnp.ndarray) -> jnp.ndarray:
        return jax.device_put(arr, self.sharding)

    def get_fsdp_strategy(self):
        return self.trainer_strategy

    def get_torch_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
