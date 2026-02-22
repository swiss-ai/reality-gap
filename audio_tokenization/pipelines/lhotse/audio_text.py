"""AudioText mode handler for the Lhotse tokenization pipeline.

Writes Parquet cache files for audio-text interleaved training data.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class AudioTextHandler:
    """Handler for audio-text tokenization mode.

    Uses WavTokenizer to encode audio into raw token sequences (without
    BOS/EOS), pairs them with text metadata from Lhotse cuts, and writes
    the results to Parquet cache files for downstream interleaving.
    """

    def __init__(self, cfg):
        from audio_tokenization.utils.clip_id_parsers import get_clip_id_parser
        self.clip_id_parser = get_clip_id_parser(cfg.get("clip_id_parser", "generic"))
        self.dataset_name = cfg.get("dataset_name", "")
        self.chunk_samples = 0

    def create_dataset(self):
        from lhotse.dataset import K2SpeechRecognitionDataset
        from lhotse.dataset.input_strategies import AudioSamples
        return K2SpeechRecognitionDataset(
            return_cuts=True,
            input_strategy=AudioSamples(),
        )

    def setup_writer(self, output_dir, rank, chunk_id, tokenizer):
        from audio_tokenization.pipelines.shard_io import ParquetChunkWriter
        parquet_dir = str(Path(output_dir) / "parquet_cache")
        # Clean up stale parquet temp files from killed runs
        pdir = Path(parquet_dir)
        if pdir.is_dir():
            for tmp in pdir.glob(f"rank_{rank:04d}_*.parquet.tmp"):
                logger.warning(f"[rank {rank}] Removing stale temp file: {tmp.name}")
                tmp.unlink()
        self._writer = ParquetChunkWriter(parquet_dir, rank, chunk_id)
        self.chunk_samples = 0

    def process_batch(self, batch, tokenizer, stats, target_sr, device):
        audios = batch["inputs"]             # (B, T) from AudioSamples strategy
        cuts = batch["supervisions"]["cut"]  # list of Cut objects (return_cuts=True)

        # K2SpeechRecognitionDataset discards audio lengths from AudioSamples,
        # so we recover them from the Cut objects directly.
        audio_lens = torch.tensor(
            [c.num_samples for c in cuts], dtype=torch.int64,
        )

        batch_audio_secs = audio_lens.sum().item() / target_sr
        audios_gpu = audios.to(device, non_blocking=True)

        with torch.inference_mode():
            raw_tokens = tokenizer.tokenize_batch_raw(
                audios_gpu,
                target_sr,
                orig_audio_samples=audio_lens.tolist(),
                pad_audio_samples=audios.shape[1],
            )

        rows = []
        batch_audio_tok = 0
        batch_text_tok = 0
        for tokens, cut in zip(raw_tokens, cuts):
            source_id, clip_num = self.clip_id_parser(cut.id)
            text = cut.supervisions[0].text if cut.supervisions else ""
            speaker = cut.supervisions[0].speaker if cut.supervisions else ""
            text_tokens = cut.custom.get("text_tokens", []) if cut.custom else []

            rows.append({
                "clip_id": cut.id,
                "source_id": source_id,
                "clip_num": clip_num,
                "speaker": speaker or "",
                "duration": cut.duration,
                "text": text or "",
                "text_tokens": text_tokens,
                "audio_tokens": tokens,
                "dataset": self.dataset_name,
            })
            batch_audio_tok += len(tokens)
            batch_text_tok += len(text_tokens)

        self._writer.add_rows(rows)
        self._writer.flush_if_needed()

        stats.samples_processed += len(rows)
        stats.tokens_generated += batch_audio_tok
        stats.text_tokens_generated += batch_text_tok
        self.chunk_samples += len(rows)

        return batch_audio_secs

    def checkpoint_writer(self) -> int:
        done_id = self._writer.finalize()
        self.chunk_samples = 0
        return done_id

    def finalize_writer(self):
        if self.chunk_samples > 0:
            self._writer.finalize()
