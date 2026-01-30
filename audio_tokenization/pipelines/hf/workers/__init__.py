"""HF worker implementations and shard assignment utilities.

Execution modes:
- shared: dynamic shard queue (best load balance, per-shard DataLoader).
- static: pre-assigned shards with a single DataLoader per worker (best throughput
  for uniform/bucketed shards).
"""

from .base import Worker
from .shard_assignment import ShardQueue

__all__ = ["Worker", "ShardQueue"]
