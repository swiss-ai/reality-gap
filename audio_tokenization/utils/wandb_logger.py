"""Ray actor for live W&B logging with aggregated throughput."""

import time
from typing import Any, Dict, Optional

import ray


@ray.remote
class WandbLoggerActor:
    """Dedicated W&B logger running in a single Ray process."""

    def __init__(
        self,
        project: str,
        entity: Optional[str],
        name: str,
        tags: Optional[list],
        config: Optional[Dict[str, Any]],
        log_interval_seconds: int = 10,
    ):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            config=config or {},
            resume="allow",
        )

        self.log_interval_seconds = max(1, int(log_interval_seconds))
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.step = 0

        # Totals
        self.total_samples = 0
        self.total_tokens = 0
        self.total_errors = 0
        self.total_skipped = 0
        self.total_duration_skipped = 0

        # Window accumulators
        self.window_samples = 0
        self.window_tokens = 0
        self.window_errors = 0
        self.window_skipped = 0
        self.window_duration_skipped = 0

    def update(
        self,
        samples: int = 0,
        tokens: int = 0,
        errors: int = 0,
        skipped: int = 0,
        duration_skipped: int = 0,
    ) -> None:
        """Accumulate counts and log at the configured interval."""
        self.total_samples += samples
        self.total_tokens += tokens
        self.total_errors += errors
        self.total_skipped += skipped
        self.total_duration_skipped += duration_skipped

        self.window_samples += samples
        self.window_tokens += tokens
        self.window_errors += errors
        self.window_skipped += skipped
        self.window_duration_skipped += duration_skipped

        self._maybe_log()

    def _maybe_log(self, force: bool = False) -> None:
        now = time.time()
        elapsed = now - self.last_log_time
        if not force and elapsed < self.log_interval_seconds:
            return
        if elapsed <= 0:
            return

        total_elapsed = now - self.start_time
        window_samples_per_sec = self.window_samples / elapsed if elapsed > 0 else 0
        window_tokens_per_sec = self.window_tokens / elapsed if elapsed > 0 else 0
        avg_samples_per_sec = self.total_samples / total_elapsed if total_elapsed > 0 else 0
        avg_tokens_per_sec = self.total_tokens / total_elapsed if total_elapsed > 0 else 0

        self._wandb.log(
            {
                "live/samples_per_second": window_samples_per_sec,
                "live/tokens_per_second": window_tokens_per_sec,
                "progress/samples_processed": self.total_samples,
                "progress/tokens_generated": self.total_tokens,
                "progress/errors": self.total_errors,
                "progress/samples_skipped": self.total_skipped,
                "progress/duration_skipped": self.total_duration_skipped,
                "progress/elapsed_seconds": total_elapsed,
                "progress/avg_samples_per_second": avg_samples_per_sec,
                "progress/avg_tokens_per_second": avg_tokens_per_sec,
            },
            step=self.step,
        )
        self.step += 1
        self.last_log_time = now

        # Reset window
        self.window_samples = 0
        self.window_tokens = 0
        self.window_errors = 0
        self.window_skipped = 0
        self.window_duration_skipped = 0

    def log_final(self, metrics: Dict[str, Any]) -> None:
        """Log final summary metrics."""
        self._maybe_log(force=True)
        if metrics:
            self._wandb.log({f"final/{k}": v for k, v in metrics.items()})

    def finish(self) -> None:
        """Flush any pending logs and finish the run."""
        self._maybe_log(force=True)
        self._wandb.finish()
