"""Shared shard-assignment utilities for distributed pipelines."""

from typing import Optional, Sequence

import ray


@ray.remote
class ShardQueue:
    """Ray actor that hands out shard IDs to workers."""

    def __init__(self, num_shards: int, initial_shards: Optional[Sequence[int]] = None):
        if initial_shards is None:
            self._pending = list(range(int(num_shards)))
        else:
            self._pending = [int(s) for s in initial_shards]
        self._next_index = 0
        self._completed = set()

    def get_next_shard(self) -> Optional[int]:
        if self._next_index >= len(self._pending):
            return None
        shard_id = self._pending[self._next_index]
        self._next_index += 1
        return shard_id

    def mark_completed(self, shard_id: int) -> None:
        self._completed.add(int(shard_id))

