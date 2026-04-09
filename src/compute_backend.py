from __future__ import annotations

import jax

from src.unified_topology import TopologyConfig, TopologyManager


def create_topology_manager() -> TopologyManager:
    """Notebook-friendly backend auto-selection with local-first fallback."""

    has_accelerator = any(device.platform in {"gpu", "tpu"} for device in jax.devices())
    strategy = "zero3" if has_accelerator and jax.device_count() > 1 else "local"
    return TopologyManager(TopologyConfig(strategy=strategy))
