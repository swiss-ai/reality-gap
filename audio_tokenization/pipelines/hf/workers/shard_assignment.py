"""Shard assignment utilities for HF workers."""

from typing import Any, Dict, List, Optional

import ray


@ray.remote
class ShardQueue:
    """Ray actor for work queue distribution (shared assignment)."""

    def __init__(self, total_shards: int, initial_shards: Optional[List[int]] = None):
        self.total_shards = total_shards
        if initial_shards is not None:
            self.pending = list(initial_shards)
        else:
            self.pending = list(range(total_shards))
        self.completed: List[int] = []
        self.failed: List[int] = []

    def get_next_shard(self) -> Optional[int]:
        if self.pending:
            return self.pending.pop(0)
        return None

    def mark_completed(self, shard_id: int) -> None:
        self.completed.append(shard_id)

    def mark_failed(self, shard_id: int) -> None:
        self.failed.append(shard_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "total": self.total_shards,
            "pending": len(self.pending),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }
