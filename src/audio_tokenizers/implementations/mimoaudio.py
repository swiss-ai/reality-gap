"""
MiMo-Audio tokenizer implementation.

MiMo-Audio-Tokenizer: 1.2B parameter Transformer-based audio tokenizer
- Tokenizer frame rate: 25 Hz
- LLM frame rate: 6.25 Hz (after patch encoder groups 4 timesteps)
- RVQ layers: 20 total, 8 used for LLM
- Codebook sizes: [1024, 1024, 128, 128, ...] (first 2 are 1024, rest are 128)
- Sample rate: 24 kHz
"""

import torch
from torchaudio.transforms import MelSpectrogram

from ..base import BaseAudioTokenizer


class MiMoAudioTokenizer(BaseAudioTokenizer):
    """MiMo-Audio tokenizer wrapper."""

    name = "mimoaudio"
    repo_path = "MiMo-Audio/src"
    default_checkpoint = "XiaomiMiMo/MiMo-Audio-Tokenizer"
    default_sample_rate = 24000

    def _load_model(self):
        """Load MiMo-Audio tokenizer model."""
        from mimo_audio_tokenizer import MiMoAudioTokenizer as MiMoTokenizer

        self.model = MiMoTokenizer.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

        config = self.model.config
        self._sampling_rate = config.sampling_rate
        self._output_sample_rate = config.sampling_rate
        self._num_quantizers = config.num_quantizers
        self._codebook_sizes = config.codebook_size or [1024] * self._num_quantizers
        self._codebook_size = self._codebook_sizes[0]
        self._downsample_rate = self.model.downsample_rate
        self._frame_rate = self._sampling_rate // self._downsample_rate

        self._mel_transform = MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.nfft,
            hop_length=config.hop_length,
            win_length=config.window_size,
            f_min=config.fmin,
            f_max=config.fmax,
            n_mels=config.n_mels,
            power=1.0,
            center=True,
        ).to(self.device)

    def encode_audio(self, audio: torch.Tensor, n_q: int = 8) -> torch.Tensor:
        """Encode audio to discrete tokens.

        Args:
            audio: Audio tensor [B, C, T] at 24kHz
            n_q: Number of RVQ layers (default: all 20, use 8 for LLM)

        Returns:
            tokens: [n_q, seq_len]
        """
        if audio.ndim == 3:
            audio = audio[:, 0, :] if audio.shape[1] > 1 else audio.squeeze(1)
        if audio.ndim == 2:
            audio = audio[0]

        # wav -> mel
        spec = self._mel_transform(audio[None, :])
        log_mel = torch.log(torch.clamp(spec, min=1e-7))
        mel_features = log_mel.squeeze(0).transpose(0, 1)  # [T_mel, n_mels]

        # Segment for encoding
        segment_size = 6000
        total_len = mel_features.shape[0]
        input_len_seg = [segment_size] * (total_len // segment_size)
        if total_len % segment_size > 0:
            input_len_seg.append(total_len % segment_size)
        input_len_seg = torch.tensor(input_len_seg, device=mel_features.device)

        codes, _ = self.model.encoder.encode(
            input_features=mel_features,
            input_lens=input_len_seg,
            return_codes_only=True,
            n_q=n_q,
        )
        return codes

    def decode_tokens(self, tokens: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Decode tokens to audio.

        Args:
            tokens: [num_quantizers, seq_len]

        Returns:
            audio: [1, 1, T] at 24kHz
        """
        tokens = tokens.long()

        # Decode in segments
        segment_len = 1500
        wav_list = []
        for start in range(0, tokens.shape[-1], segment_len):
            wav_segment = self.model.decode(tokens[:, start:start + segment_len]).float()
            wav_list.append(wav_segment)

        audio = torch.cat(wav_list, dim=-1)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        return audio

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def codebook_sizes(self) -> list:
        return self._codebook_sizes

    @property
    def downsample_rate(self) -> int:
        return self._downsample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    @property
    def frame_rate(self) -> int:
        return self._frame_rate

    @property
    def num_quantizers(self) -> int:
        return self._num_quantizers
