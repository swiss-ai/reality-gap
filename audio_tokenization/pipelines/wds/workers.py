"""WebDataset (tar shard) workers for audio tokenization."""

import csv
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import ray
import torch
import webdataset as wds

from audio_tokenization.pipelines.base import WorkerStats
from audio_tokenization.pipelines.shard_io import open_shard_writer, finalize_shard_writer


# =============================================================================
# Type aliases
# =============================================================================
SampleMeta = Tuple[Optional[int], Optional[float], Optional[int]]  # (sample_rate, duration, num_samples)
ShardMetadata = Dict[str, SampleMeta]  # sample_key -> meta


# =============================================================================
# Metadata loading
# =============================================================================
def _parse_metadata_row(row: Dict[str, str]) -> Optional[Tuple[str, SampleMeta]]:
    """Parse a single metadata TSV row. Returns (sample_key, meta) or None if invalid."""
    if row.get("error"):
        return None
    try:
        sample_key = row["sample_key"].lower()
        meta = (
            int(row["sample_rate"]) if row.get("sample_rate") else None,
            float(row["duration"]) if row.get("duration") else None,
            int(row["num_samples"]) if row.get("num_samples") else None,
        )
        return sample_key, meta
    except (ValueError, KeyError):
        return None


def load_shard_metadata(metadata_dir: Path, shard: str) -> ShardMetadata:
    """Load metadata for a single shard from per-shard TSV file."""
    shard_path = metadata_dir / f"{shard.lower().replace('/', '_')}.tsv"
    if not shard_path.exists():
        return {}

    metadata = {}
    with open(shard_path, "r", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            parsed = _parse_metadata_row(row)
            if parsed:
                metadata[parsed[0]] = parsed[1]
    return metadata


class MetadataStore:
    """Handles metadata loading - either per-shard (lazy) or full TSV (indexed)."""

    def __init__(self, path: Optional[Path], logger: logging.Logger):
        self._path = path
        self._logger = logger
        self._is_dir = False
        self._shard_index: Optional[Dict[str, ShardMetadata]] = None

        if path is None:
            return
        if not path.exists():
            raise FileNotFoundError(f"Metadata path not found: {path}")

        if path.is_dir():
            self._is_dir = True
            logger.info(f"Using per-shard metadata from {path}")
        else:
            self._build_index_from_tsv(path)

    def _build_index_from_tsv(self, path: Path) -> None:
        """Load full TSV and build shard-indexed lookup."""
        self._shard_index = {}
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames or "shard" not in reader.fieldnames:
                raise ValueError(f"Invalid TSV: missing 'shard' column in {path}")

            for row in reader:
                shard = row.get("shard", "").lower()
                parsed = _parse_metadata_row(row)
                if parsed and shard:
                    self._shard_index.setdefault(shard, {})[parsed[0]] = parsed[1]

        total = sum(len(v) for v in self._shard_index.values())
        self._logger.info(f"Loaded {total} entries across {len(self._shard_index)} shards")

    def get(self, shard_path: Path) -> Optional[ShardMetadata]:
        """Get metadata for a shard."""
        if self._path is None:
            return None
        shard = f"{shard_path.parent.name}/{shard_path.stem}".lower()
        if self._is_dir:
            return load_shard_metadata(self._path, shard)
        return self._shard_index.get(shard) if self._shard_index else None

    @property
    def available(self) -> bool:
        return self._path is not None


# =============================================================================
# Sample rate detection
# =============================================================================
try:
    from mutagen.mp3 import MP3
    import io as _io
    _HAS_MUTAGEN = True
except ImportError:
    _HAS_MUTAGEN = False


class SampleRateDetector:
    """Detects audio sample rate using multiple methods."""

    def __init__(self, ffprobe_path: Optional[Path] = None):
        self._ffprobe_path = ffprobe_path

    def detect(self, data: bytes, ext: str) -> Optional[int]:
        """Detect sample rate using available methods."""
        if ext == "mp3":
            sr = self._detect_mp3_mutagen(data) or self._detect_mp3_header(data)
            if sr:
                return sr
        return self._detect_ffprobe(data) if self._ffprobe_path else None

    def _detect_mp3_mutagen(self, data: bytes) -> Optional[int]:
        if not _HAS_MUTAGEN:
            return None
        try:
            return MP3(_io.BytesIO(data)).info.sample_rate
        except Exception:
            return None

    def _detect_mp3_header(self, data: bytes, max_scan: int = 262144) -> Optional[int]:
        """Best-effort MP3 header scan."""
        if not data or len(data) < 4:
            return None

        offset = 0
        if data.startswith(b"ID3") and len(data) >= 10:
            size_bytes = data[6:10]
            offset = min(len(data), 10 + (
                (size_bytes[0] & 0x7F) << 21 | (size_bytes[1] & 0x7F) << 14 |
                (size_bytes[2] & 0x7F) << 7 | (size_bytes[3] & 0x7F)
            ))

        rates_map = {0x03: (44100, 48000, 32000), 0x02: (22050, 24000, 16000)}
        for i in range(offset, min(len(data) - 4, offset + max_scan)):
            if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                b1, b2 = data[i + 1], data[i + 2]
                version_id, layer, sr_index = (b1 >> 3) & 0x03, (b1 >> 1) & 0x03, (b2 >> 2) & 0x03
                if version_id == 0x01 or layer == 0x00 or sr_index == 0x03:
                    continue
                return rates_map.get(version_id, (11025, 12000, 8000))[sr_index]
        return None

    def _detect_ffprobe(self, data: bytes) -> Optional[int]:
        if not self._ffprobe_path or not self._ffprobe_path.exists():
            return None
        try:
            result = subprocess.run(
                [str(self._ffprobe_path), "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", "-i", "pipe:0"],
                input=data, capture_output=True, check=True
            )
            return int(float(result.stdout.decode().strip().split()[0]))
        except Exception:
            return None


# =============================================================================
# Audio decoding
# =============================================================================
_SOXR_SUPPORT: Dict[str, bool] = {}


def _ffmpeg_supports_soxr(ffmpeg_path: Path) -> bool:
    key = str(ffmpeg_path)
    if key not in _SOXR_SUPPORT:
        try:
            result = subprocess.run(
                [str(ffmpeg_path), "-hide_banner", "-h", "filter=aresample"],
                capture_output=True, text=True, check=True
            )
            _SOXR_SUPPORT[key] = "soxr" in result.stdout
        except Exception:
            _SOXR_SUPPORT[key] = False
    return _SOXR_SUPPORT[key]


def _decode_audio_bytes(
    data: bytes, target_sr: int, ffmpeg_path: Path, resampler: Optional[str] = None
) -> Optional[np.ndarray]:
    """Decode audio bytes to mono float32 at target_sr using ffmpeg."""
    cmd = [str(ffmpeg_path), "-v", "error", "-i", "pipe:0", "-ac", "1"]
    if resampler == "soxr":
        cmd += ["-af", "aresample=resampler=soxr:precision=20"]
    elif resampler:
        cmd += ["-af", f"aresample=resampler={resampler}"]
    cmd += ["-ar", str(target_sr), "-f", "f32le", "pipe:1"]

    try:
        raw = subprocess.check_output(cmd, input=data, stderr=subprocess.DEVNULL)
        if raw:
            audio = np.frombuffer(raw, dtype=np.float32)
            return np.array(audio, copy=True) if audio.size > 0 else None
    except subprocess.CalledProcessError:
        pass
    return None


# =============================================================================
# WDS sample handling
# =============================================================================
@dataclass
class DecodedSample:
    """Result of decoding a WDS sample."""
    url: Optional[str]
    audio: Optional[np.ndarray]
    sample_rate: Optional[int]
    duration: Optional[float]
    skip_reason: Optional[str]


def _find_audio_in_sample(
    sample: Dict[str, Any], extensions: Sequence[str]
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Find audio data in WDS sample. Returns (data, ext, ext_key)."""
    for ext in extensions:
        if ext in sample:
            return sample[ext], ext, ext
        for key in sample:
            if key.endswith(f".{ext}"):
                return sample[key], ext, key
    return None, None, None


def _get_sample_key(sample: Dict[str, Any], ext_key: str, ext: str) -> str:
    """Reconstruct full sample key from WDS sample, lowercased for matching."""
    wds_key = sample.get("__key__", "")
    if ext_key == ext:
        return wds_key.lower()
    return (wds_key + "." + ext_key[:-len(ext) - 1]).lower()


def _select_audio_sample(sample: Dict[str, Any], audio_extensions: Sequence[str]) -> bool:
    """Check if sample contains audio."""
    return _find_audio_in_sample(sample, audio_extensions)[0] is not None


def _decode_sample(
    sample: Dict[str, Any],
    audio_extensions: Sequence[str],
    target_sr: int,
    ffmpeg_path: Path,
    resampler: Optional[str] = None,
    min_sample_rate: Optional[int] = None,
    sr_detector: Optional[SampleRateDetector] = None,
    shard_metadata: Optional[ShardMetadata] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> DecodedSample:
    """Decode a WDS sample with optional metadata-based filtering."""
    url = sample.get("__url__", "").replace("file:", "") or None
    data, ext, ext_key = _find_audio_in_sample(sample, audio_extensions)
    if data is None:
        return DecodedSample(url, None, None, None, "missing_audio")

    # Get metadata
    sample_rate, duration = None, None
    if shard_metadata:
        meta = shard_metadata.get(_get_sample_key(sample, ext_key, ext))
        if meta:
            sample_rate, duration, _ = meta

    # Sample rate filter
    if min_sample_rate is not None:
        if sample_rate is None and sr_detector:
            sample_rate = sr_detector.detect(data, ext)
        if sample_rate is not None and sample_rate < min_sample_rate:
            return DecodedSample(url, None, sample_rate, duration, "frequency")
        if sample_rate is None:
            return DecodedSample(url, None, None, duration, "frequency_unknown")

    # Duration filter (skip decode if filtered)
    if duration is not None:
        if min_duration and duration < min_duration:
            return DecodedSample(url, None, sample_rate, duration, "duration_short")
        if max_duration and duration > max_duration:
            return DecodedSample(url, None, sample_rate, duration, "duration_long")

    # Decode
    audio = _decode_audio_bytes(data, target_sr, ffmpeg_path, resampler)
    return DecodedSample(url, audio, sample_rate, duration, "decode_error" if audio is None else None)


# =============================================================================
# WDS Worker
# =============================================================================
@ray.remote(num_gpus=1)
class WDSWorker:
    """Ray worker for WebDataset (tar shard) tokenization."""

    VALID_MODES = ["audio_only"]

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        worker_id: int,
        mode: str,
        audio_extensions: Sequence[str],
        batch_size: int = 1,
        buffer_size: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
        min_sample_rate: Optional[int] = None,
        target_bucket: Optional[int] = None,
        silence_unique_threshold: Optional[int] = None,
        torch_compile: bool = True,
        trim_last_tokens: int = 5,
        decode_workers_per_gpu: int = 0,
        dataloader_prefetch_factor: int = 2,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        wandb_logger=None,
        wandb_log_interval_seconds: Optional[int] = None,
    ):
        # Validate
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}")
        if not audio_extensions:
            raise ValueError("audio_extensions required")
        if target_sample_rate is None:
            raise ValueError("target_sample_rate required")

        self.logger = logging.getLogger(f"WDSWorker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Config
        self.tokenizer_path = tokenizer_path
        self.output_dir = Path(output_dir)
        self.worker_id = worker_id
        self.mode = mode
        self.audio_extensions = list(audio_extensions)
        self.batch_size = max(1, batch_size)
        self.buffer_size = max(self.batch_size, buffer_size or batch_size)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        self.min_sample_rate = min_sample_rate
        self.target_bucket = target_bucket
        self.silence_unique_threshold = silence_unique_threshold
        self.torch_compile = torch_compile
        self.trim_last_tokens = max(0, int(trim_last_tokens))
        self.decode_workers_per_gpu = decode_workers_per_gpu
        self.dataloader_prefetch_factor = dataloader_prefetch_factor

        # FFmpeg setup
        self.ffmpeg_path = Path(ffmpeg_path or os.environ.get("FFMPEG_PATH", ""))
        if not self.ffmpeg_path.exists():
            raise ValueError("ffmpeg_path required")
        self.ffprobe_path = None
        for p in [ffprobe_path, os.environ.get("FFPROBE_PATH"), self.ffmpeg_path.parent / "ffprobe"]:
            if p and Path(p).exists():
                self.ffprobe_path = Path(p)
                break
        self._resampler = "soxr" if _ffmpeg_supports_soxr(self.ffmpeg_path) else None

        # Metadata & SR detection
        self._metadata = MetadataStore(Path(metadata_path) if metadata_path else None, self.logger)
        self._sr_detector = SampleRateDetector(self.ffprobe_path) if min_sample_rate else None
        if min_sample_rate and not self._metadata.available and not self.ffprobe_path:
            raise ValueError("ffprobe required for min_sample_rate without metadata")

        # Lazy
        self._tokenizer = None
        self._vocab_size = None

        # W&B
        self._wandb_logger = wandb_logger
        self._wandb_interval = max(1.0, float(wandb_log_interval_seconds or 1))
        self._wandb_last = time.time()
        self._wandb_pending = {"samples": 0, "tokens": 0, "errors": 0, "skipped": 0, "duration_skipped": 0, "frequency_skipped": 0}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from audio_tokenization.vokenizers import create_tokenizer
            self._tokenizer = create_tokenizer(
                omni_tokenizer_path=self.tokenizer_path,
                device="cuda",
                torch_compile=self.torch_compile,
                trim_last_tokens=self.trim_last_tokens,
            )
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            self._vocab_size = len(getattr(self.tokenizer, "omni_tokenizer", None) or
                                   __import__("transformers").AutoTokenizer.from_pretrained(self.tokenizer_path))
        return self._vocab_size

    def _wandb_log(self, force: bool = False, **updates) -> None:
        if self._wandb_logger is None:
            return
        for k, v in updates.items():
            self._wandb_pending[k] = self._wandb_pending.get(k, 0) + v
        now = time.time()
        if force or now - self._wandb_last >= self._wandb_interval:
            if any(self._wandb_pending.values()):
                self._wandb_logger.update.remote(**self._wandb_pending)
                self._wandb_pending = {k: 0 for k in self._wandb_pending}
            self._wandb_last = now

    def _iter_decoded(self, shard_path: Path, metadata: Optional[ShardMetadata]) -> Iterable[DecodedSample]:
        """Iterate decoded audio from a shard."""
        dataset = wds.DataPipeline(
            wds.SimpleShardList([str(shard_path)]),
            wds.tarfile_to_samples(handler=wds.handlers.warn_and_continue),
            wds.select(partial(_select_audio_sample, audio_extensions=self.audio_extensions)),
        )
        decode = partial(
            _decode_sample, audio_extensions=self.audio_extensions, target_sr=self.target_sample_rate,
            ffmpeg_path=self.ffmpeg_path, resampler=self._resampler, min_sample_rate=self.min_sample_rate,
            sr_detector=self._sr_detector, shard_metadata=metadata, min_duration=self.min_duration, max_duration=self.max_duration,
        )
        try:
            if self.decode_workers_per_gpu > 0:
                max_inflight = self.decode_workers_per_gpu * max(1, self.dataloader_prefetch_factor)
                with ThreadPoolExecutor(self.decode_workers_per_gpu) as pool:
                    futures, nxt = {}, 0
                    for idx, s in enumerate(dataset):
                        futures[idx] = pool.submit(decode, s)
                        while len(futures) >= max_inflight:
                            yield futures.pop(nxt).result()
                            nxt += 1
                    while futures:
                        yield futures.pop(nxt).result()
                        nxt += 1
            else:
                for s in dataset:
                    yield decode(s)
        finally:
            getattr(dataset, "close", lambda: None)()

    def _process_tokens(self, tokens, builder, stats: WorkerStats) -> None:
        if tokens is None:
            stats.errors += 1
            return
        if self.silence_unique_threshold and tokens.numel() >= 4:
            audio_tok = tokens[2:-2] - getattr(self.tokenizer, "audio_token_offset", 0)
            if audio_tok.numel() > 0 and torch.unique(audio_tok.cpu()).numel() <= self.silence_unique_threshold:
                stats.samples_skipped += 1
                return
        t = torch.as_tensor(tokens, dtype=torch.int64).detach().cpu()
        builder.add_item(t)
        builder.end_document()
        stats.samples_processed += 1
        stats.tokens_generated += len(t)

    def _process_batch(self, batch: List[np.ndarray], builder, stats: WorkerStats) -> None:
        if not batch:
            return
        prev = (stats.samples_processed, stats.tokens_generated, stats.errors, stats.samples_skipped, stats.duration_skipped, stats.frequency_skipped)
        lengths = [a.shape[-1] for a in batch]
        if self.target_bucket:
            lengths = [min(l, self.target_bucket) for l in lengths]
            batch = [a[:l] for a, l in zip(batch, lengths)]
        try:
            if self.batch_size > 1 and hasattr(self.tokenizer, "tokenize_batch"):
                pad = self.target_bucket or max(lengths)
                if pad is not None and any(length != pad for length in lengths):
                    dtype = batch[0].dtype if batch else np.float32
                    padded = np.zeros((len(batch), pad), dtype=dtype)
                    for i, (audio, length) in enumerate(zip(batch, lengths)):
                        if length >= pad:
                            padded[i, :pad] = audio[:pad]
                        else:
                            padded[i, :length] = audio
                    batch = padded
                with torch.inference_mode():
                    for tok in self.tokenizer.tokenize_batch(batch, self.target_sample_rate, orig_audio_samples=lengths, pad_audio_samples=pad):
                        self._process_tokens(tok, builder, stats)
            else:
                for a in batch:
                    with torch.inference_mode():
                        self._process_tokens(self.tokenizer.tokenize(a, self.target_sample_rate), builder, stats)
        except Exception as e:
            self.logger.error(f"Tokenize error: {e}", exc_info=True)
            stats.errors += len(batch)
        self._wandb_log(samples=stats.samples_processed-prev[0], tokens=stats.tokens_generated-prev[1], errors=stats.errors-prev[2],
                        skipped=stats.samples_skipped-prev[3], duration_skipped=stats.duration_skipped-prev[4], frequency_skipped=stats.frequency_skipped-prev[5])

    def process_shard(self, shard_id: int, shard_path: Path, total_shards: int) -> Dict[str, Any]:
        stats = WorkerStats()
        builder, tmp_bin, tmp_idx, bin_path, idx_path = open_shard_writer(self.output_dir, self.worker_id, shard_id, total_shards, self.vocab_size)
        buffer: List[np.ndarray] = []
        try:
            for r in self._iter_decoded(shard_path, self._metadata.get(shard_path)):
                if r.skip_reason in ("frequency", "frequency_unknown"):
                    stats.frequency_skipped += 1
                elif r.skip_reason in ("duration_short", "duration_long"):
                    stats.duration_skipped += 1
                elif r.audio is None:
                    stats.errors += 1
                else:
                    if r.duration is None:
                        dur = r.audio.shape[-1] / self.target_sample_rate
                        if (self.min_duration and dur < self.min_duration) or (self.max_duration and dur > self.max_duration):
                            stats.duration_skipped += 1
                            continue
                    buffer.append(r.audio)
                    if len(buffer) >= self.buffer_size:
                        while len(buffer) >= self.batch_size:
                            self._process_batch(buffer[:self.batch_size], builder, stats)
                            del buffer[:self.batch_size]
            while buffer:
                self._process_batch(buffer[:self.batch_size], builder, stats)
                del buffer[:self.batch_size]
        except Exception as e:
            self.logger.error(f"Shard error: {e}")
            stats.errors += 1
        finally:
            finalize_shard_writer(builder, tmp_bin, tmp_idx, bin_path, idx_path)
            self._wandb_log(force=True)
        return stats.finalize()

    def run_shards(self, shard_source, shard_paths: Sequence[str], total_shards: int, progress_actor=None) -> Dict[str, Any]:
        total, shard_list = WorkerStats(), []
        def do(sid):
            self.logger.info(f"Worker {self.worker_id}: shard {sid}/{total_shards}")
            s = self.process_shard(sid, Path(shard_paths[sid]), total_shards)
            s["shard_id"], s["worker_id"] = sid, self.worker_id
            shard_list.append(s)
            for k in ["samples_processed", "tokens_generated", "errors", "samples_skipped", "duration_skipped"]:
                setattr(total, k, getattr(total, k) + s.get(k, 0))
            total.frequency_skipped += s.get("frequency_skipped", 0)
            if progress_actor:
                progress_actor.update.remote(s["samples_processed"])
        if isinstance(shard_source, list):
            for sid in shard_source:
                do(sid)
        else:
            while (sid := ray.get(shard_source.get_next_shard.remote())) is not None:
                do(sid)
                ray.get(shard_source.mark_completed.remote(sid))
        self._wandb_log(force=True)
        r = total.finalize()
        r["worker_id"], r["shard_stats"] = self.worker_id, shard_list
        return r
