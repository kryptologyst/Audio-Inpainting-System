"""Audio data loading and preprocessing utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Dataset class for audio inpainting tasks."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        duration: Optional[float] = None,
        split: str = "train",
        augment: bool = True,
    ):
        """Initialize audio dataset.
        
        Args:
            data_dir: Directory containing audio files.
            metadata_file: Path to metadata CSV file.
            sample_rate: Target sample rate for audio.
            duration: Maximum duration in seconds (None for full length).
            split: Dataset split ('train', 'val', 'test').
            augment: Whether to apply data augmentation.
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.split = split
        self.augment = augment
        
        if metadata_file:
            self.metadata = pd.read_csv(metadata_file)
            self.metadata = self.metadata[self.metadata["split"] == split]
        else:
            # Create metadata from directory structure
            self.metadata = self._create_metadata_from_dir()
            
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _create_metadata_from_dir(self) -> pd.DataFrame:
        """Create metadata DataFrame from directory structure."""
        audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.flac"))
        
        metadata = []
        for i, audio_file in enumerate(audio_files):
            metadata.append({
                "id": f"sample_{i:06d}",
                "path": str(audio_file),
                "split": self.split,
                "duration": None,  # Will be computed on first load
            })
        
        return pd.DataFrame(metadata)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing audio tensors and metadata.
        """
        row = self.metadata.iloc[idx]
        audio_path = Path(row["path"])
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Truncate or pad to desired duration
        if self.duration:
            target_length = int(self.duration * self.sample_rate)
            if len(audio) > target_length:
                start = np.random.randint(0, len(audio) - target_length)
                audio = audio[start:start + target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        return {
            "audio": audio_tensor,
            "id": row["id"],
            "path": str(audio_path),
            "sample_rate": self.sample_rate,
        }


def load_audio(file_path: Union[str, Path], sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr


def save_audio(audio: np.ndarray, file_path: Union[str, Path], sample_rate: int = 16000) -> None:
    """Save audio array to file.
    
    Args:
        audio: Audio array to save.
        file_path: Output file path.
        sample_rate: Sample rate of audio.
    """
    sf.write(file_path, audio, sample_rate)


def create_missing_mask(
    audio_length: int,
    missing_percentage: float = 0.2,
    missing_length_range: Tuple[float, float] = (0.1, 0.5),
    sample_rate: int = 16000,
) -> np.ndarray:
    """Create a mask for missing audio segments.
    
    Args:
        audio_length: Length of audio in samples.
        missing_percentage: Percentage of audio to mask.
        missing_length_range: Range of missing segment lengths in seconds.
        sample_rate: Sample rate.
        
    Returns:
        Boolean mask where True indicates missing samples.
    """
    mask = np.zeros(audio_length, dtype=bool)
    
    # Calculate number of missing samples
    total_missing_samples = int(audio_length * missing_percentage)
    
    # Create missing segments
    remaining_samples = total_missing_samples
    while remaining_samples > 0:
        # Random segment length
        min_length = int(missing_length_range[0] * sample_rate)
        max_length = int(missing_length_range[1] * sample_rate)
        segment_length = np.random.randint(min_length, max_length + 1)
        segment_length = min(segment_length, remaining_samples)
        
        # Random start position
        start_pos = np.random.randint(0, audio_length - segment_length + 1)
        
        # Check if this segment overlaps with existing missing regions
        if not np.any(mask[start_pos:start_pos + segment_length]):
            mask[start_pos:start_pos + segment_length] = True
            remaining_samples -= segment_length
    
    return mask


def apply_missing_mask(audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply missing mask to audio (set missing samples to NaN).
    
    Args:
        audio: Input audio array.
        mask: Boolean mask indicating missing samples.
        
    Returns:
        Audio with missing samples set to NaN.
    """
    audio_with_missing = audio.copy()
    audio_with_missing[mask] = np.nan
    return audio_with_missing


def interpolate_missing(audio_with_missing: np.ndarray) -> np.ndarray:
    """Interpolate missing audio samples using linear interpolation.
    
    Args:
        audio_with_missing: Audio with NaN values for missing samples.
        
    Returns:
        Interpolated audio.
    """
    from scipy.interpolate import interp1d
    
    # Find valid (non-NaN) indices
    valid_indices = ~np.isnan(audio_with_missing)
    valid_values = audio_with_missing[valid_indices]
    
    if len(valid_values) < 2:
        # Not enough valid samples for interpolation
        return np.zeros_like(audio_with_missing)
    
    # Create interpolation function
    f = interp1d(
        np.where(valid_indices)[0],
        valid_values,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    
    # Interpolate all samples
    interpolated = f(np.arange(len(audio_with_missing)))
    
    return interpolated


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to audio at specified SNR.
    
    Args:
        audio: Input audio.
        snr_db: Signal-to-noise ratio in dB.
        
    Returns:
        Noisy audio.
    """
    signal_power = np.mean(audio**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio to target RMS level.
    
    Args:
        audio: Input audio.
        target_rms: Target RMS level.
        
    Returns:
        Normalized audio.
    """
    current_rms = np.sqrt(np.mean(audio**2))
    if current_rms > 0:
        return audio * (target_rms / current_rms)
    return audio


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 100,
    sample_rate: int = 16000,
    duration: float = 2.0,
) -> None:
    """Create a synthetic dataset for testing.
    
    Args:
        output_dir: Directory to save synthetic data.
        num_samples: Number of samples to generate.
        sample_rate: Sample rate.
        duration: Duration of each sample in seconds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    for i in range(num_samples):
        # Generate synthetic audio (mixture of sine waves)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Random frequencies and amplitudes
        freqs = np.random.uniform(200, 2000, np.random.randint(2, 5))
        amps = np.random.uniform(0.1, 0.5, len(freqs))
        
        audio = np.zeros_like(t)
        for freq, amp in zip(freqs, amps):
            audio += amp * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        audio += 0.05 * np.random.randn(len(audio))
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Save audio file
        filename = f"synthetic_{i:06d}.wav"
        filepath = output_dir / filename
        save_audio(audio, filepath, sample_rate)
        
        # Add to metadata
        metadata.append({
            "id": f"synthetic_{i:06d}",
            "path": str(filepath),
            "duration": duration,
            "sample_rate": sample_rate,
            "split": "train" if i < 0.7 * num_samples else "val" if i < 0.85 * num_samples else "test",
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    
    logger.info(f"Created synthetic dataset with {num_samples} samples in {output_dir}")
