"""
Unified base class for audio tokenizers with automatic registry integration.
This combines the abstract interface with easy import management.
"""

import sys
import time
import torch
import numpy as np
import torchaudio
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List
import logging

# Import config for repo paths
from ..config import get_repo_path

logger = logging.getLogger(__name__)


class RegisteredTokenizerMeta(ABCMeta):
    """Metaclass that automatically registers tokenizers when they're defined."""

    _registry: Dict[str, type] = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        # Only skip registration for the BaseAudioTokenizer class itself
        # All subclasses should be registered
        if name == 'BaseAudioTokenizer':
            return cls

        # Auto-register with lowercase name
        tokenizer_name = attrs.get('name', name.lower().replace('tokenizer', '').replace('wrapper', ''))
        mcs._registry[tokenizer_name] = cls
        logger.info(f"Registered tokenizer: {tokenizer_name} -> {name}")

        return cls
    
    @classmethod
    def get_tokenizer(mcs, name: str) -> type:
        """Get a registered tokenizer class by name."""
        if name not in mcs._registry:
            raise ValueError(f"Tokenizer '{name}' not found. Available: {list(mcs._registry.keys())}")
        return mcs._registry[name]
    
    @classmethod
    def list_tokenizers(mcs) -> List[str]:
        """List all registered tokenizers."""
        return list(mcs._registry.keys())


class BaseAudioTokenizer(ABC, metaclass=RegisteredTokenizerMeta):
    """
    Base class for all audio tokenizers with automatic registration.
    Subclasses are automatically registered and can be accessed by name.
    """
    
    # Override these in subclasses
    name: str = None  # Tokenizer name for registry
    repo_path: str = None  # Path relative to repos/ directory
    default_checkpoint: str = None  # Default model checkpoint
    default_sample_rate: int = 16000
    
    def __init__(self, 
                 checkpoint: Optional[str] = None,
                 device: str = "cuda",
                 sample_rate: Optional[int] = None,
                 **kwargs):
        """
        Initialize tokenizer with common setup.
        
        Args:
            checkpoint: Model checkpoint path or HF model ID
            device: Device to run on
            sample_rate: Input sample rate
            **kwargs: Additional tokenizer-specific arguments
        """
        self.checkpoint = checkpoint or self.default_checkpoint
        self.device = device if device == "cpu" else (device if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate or self.default_sample_rate
        self.config = kwargs
        
        # Add repo to path if specified
        if self.repo_path:
            repo_full_path = get_repo_path(self.repo_path)
            if repo_full_path.exists() and str(repo_full_path) not in sys.path:
                sys.path.insert(0, str(repo_full_path))
                logger.debug(f"Added {repo_full_path} to sys.path")
            elif not repo_full_path.exists():
                logger.warning(f"Repo path does not exist: {repo_full_path}")
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Move model to device
        if self.model and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the specific tokenizer model."""
        pass
    
    @abstractmethod
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio tensor to discrete tokens.
        
        Args:
            audio: Audio tensor (B, C, T) at self.sample_rate
            
        Returns:
            Token codes
        """
        pass
    
    @abstractmethod
    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode tokens back to audio.

        Args:
            tokens: Token codes
            **kwargs: Additional arguments (e.g., text for text-aware decoding)

        Returns:
            Audio tensor (B, C, T)
        """
        pass
    
    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Size of the tokenizer's codebook."""
        pass
    
    @property
    @abstractmethod
    def downsample_rate(self) -> int:
        """Temporal downsampling factor."""
        pass

    def tokens_from_waveform_samples(self, num_waveform_samples: int) -> int:
        """Return token length from waveform sample count.

        Subclasses should override when token length can be derived from
        waveform length (e.g., fixed downsample_rate).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement tokens_from_waveform_samples"
        )
    
    @property  
    @abstractmethod
    def output_sample_rate(self) -> int:
        """Output sample rate after decoding."""
        pass
    
    def preprocess_audio(self, 
                        audio: Union[np.ndarray, torch.Tensor, str],
                        sr: Optional[int] = None) -> torch.Tensor:
        """
        Preprocess various audio inputs to correct tensor format.
        
        Args:
            audio: Audio array, tensor, or file path
            sr: Sample rate (if audio is array/tensor)
            
        Returns:
            Preprocessed audio tensor (B, C, T)
        """
        # Load from file if path is given
        if isinstance(audio, str):
            audio, sr = torchaudio.load(audio)
        
        # Convert numpy to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape: (B, C, T)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            if audio.shape[0] > 8:  # Likely (samples, channels) not (batch/channels, samples)
                audio = audio.T
            if audio.shape[0] == 1:  # Single channel
                audio = audio.unsqueeze(0)
        
        # Resample if needed (do this before moving to device for efficiency)
        if sr and sr != self.sample_rate:
            # Ensure audio is on CPU for resampling, then move to device after
            if audio.is_cuda:
                audio = audio.cpu()
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Move to device
        audio = audio.to(self.device)
        
        return audio
    
    def encode(self,
              audio: Union[np.ndarray, torch.Tensor, str],
              sr: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full encoding pipeline with preprocessing.
        
        Returns:
            tokens: Encoded tokens
            info: Encoding information
        """
        # Preprocess
        audio_tensor = self.preprocess_audio(audio, sr)
        
        # Encode
        start_time = time.time()
        with torch.no_grad():
            tokens = self.encode_audio(audio_tensor)
        encode_time = time.time() - start_time
        
        # Info
        info = {
            "encode_time": encode_time,
            "input_shape": list(audio_tensor.shape),
            "token_shape": list(tokens.shape),
            "num_tokens": tokens.numel(),
        }
        
        return tokens, info
    
    def decode(self, tokens: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full decoding pipeline.

        Args:
            tokens: Token tensor to decode
            **kwargs: Additional arguments passed to decode_tokens (e.g., text for TaDiCodec)

        Returns:
            audio: Decoded audio
            info: Decoding information
        """
        start_time = time.time()
        with torch.no_grad():
            audio = self.decode_tokens(tokens, **kwargs)
        decode_time = time.time() - start_time
        
        info = {
            "decode_time": decode_time,
            "output_shape": list(audio.shape),
            "output_sample_rate": self.output_sample_rate
        }
        
        return audio, info
    
    def encode_decode(self,
                     audio: Union[np.ndarray, torch.Tensor, str],
                     sr: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full encode-decode pipeline with metrics.
        
        Returns:
            reconstructed: Reconstructed audio
            metrics: Complete metrics dictionary
        """
        # Encode
        tokens, encode_info = self.encode(audio, sr)
        
        # Decode
        reconstructed, decode_info = self.decode(tokens)
        
        # Calculate comprehensive metrics
        input_duration = encode_info["input_shape"][-1] / self.sample_rate
        output_duration = reconstructed.shape[-1] / self.output_sample_rate
        
        metrics = {
            **encode_info,
            **decode_info,
            "codebook_size": self.codebook_size,
            "downsample_rate": self.downsample_rate,
            "compression_ratio": encode_info["input_shape"][-1] / tokens.numel(),
            "tokens_per_second": tokens.numel() / input_duration,
            "encode_rtf": encode_info["encode_time"] / input_duration,
            "decode_rtf": decode_info["decode_time"] / output_duration,
            "total_time": encode_info["encode_time"] + decode_info["decode_time"],
            "input_sample_rate": self.sample_rate,
            "output_sample_rate": self.output_sample_rate,
        }
        
        return reconstructed, metrics
    
    @classmethod
    def from_pretrained(cls, checkpoint: str = None, **kwargs):
        """Create tokenizer from pretrained checkpoint."""
        return cls(checkpoint=checkpoint, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available tokenizers."""
        return RegisteredTokenizerMeta.list_tokenizers()
    
    @classmethod
    def create(cls, name: str, **kwargs):
        """Create a tokenizer by name."""
        tokenizer_class = RegisteredTokenizerMeta.get_tokenizer(name)
        return tokenizer_class(**kwargs)
    
    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: Optional[int] = None):
        """Save audio tensor to file."""
        if audio.ndim == 3:
            audio = audio[0]  # Remove batch dimension
        
        if audio.is_cuda:
            audio = audio.cpu()
        
        sr = sample_rate or self.output_sample_rate
        torchaudio.save(path, audio, sr)
    
    def benchmark_file(self, audio_path: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark a single audio file.
        
        Args:
            audio_path: Input audio file path
            save_path: Optional path to save reconstruction
            
        Returns:
            Complete metrics dictionary
        """
        # Process
        reconstructed, metrics = self.encode_decode(audio_path)
        
        # Save if requested
        if save_path:
            self.save_audio(reconstructed, save_path)
            metrics["saved_to"] = save_path
        
        metrics["input_file"] = audio_path
        return metrics
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"checkpoint='{self.checkpoint}', "
                f"device='{self.device}', "
                f"sample_rate={self.sample_rate})")


# Convenience function
def get_tokenizer(name: str, **kwargs) -> BaseAudioTokenizer:
    """Get a tokenizer instance by name."""
    return BaseAudioTokenizer.create(name, **kwargs)


def list_tokenizers() -> List[str]:
    """List all available tokenizers."""
    return BaseAudioTokenizer.list_available()
