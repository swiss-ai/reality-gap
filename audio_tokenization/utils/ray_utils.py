"""Ray initialization helpers."""

import os
from typing import Any, Dict, Optional

import ray


def init_ray(ray_config: Optional[Dict[str, Any]], num_gpus: int) -> None:
    """Initialize Ray with optional multi-node config.

    Args:
        ray_config: Dict with keys: address, redis_password, namespace.
        num_gpus: Number of GPU workers to allocate for local init.
    """
    if ray.is_initialized():
        return

    ray_config = ray_config or {}
    address = ray_config.get("address") or os.environ.get("RAY_ADDRESS")
    redis_password = (
        ray_config.get("redis_password")
        or os.environ.get("RAY_REDIS_PASSWORD")
        or os.environ.get("REDIS_PASSWORD")
    )
    namespace = ray_config.get("namespace") or os.environ.get("RAY_NAMESPACE")

    if address:
        ray.init(address=address, _redis_password=redis_password, namespace=namespace)
    else:
        num_cpus = ray_config.get("num_cpus") or os.cpu_count() or (num_gpus + 2)
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, namespace=namespace)
