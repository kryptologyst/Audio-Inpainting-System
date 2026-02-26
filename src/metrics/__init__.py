"""Evaluation metrics for audio inpainting."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi

logger = logging.getLogger(__name__)


def si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        
    Returns:
        SI-SDR value in dB.
    """
    # Remove DC component
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Compute optimal scaling
    alpha = np.dot(estimation, reference) / np.dot(reference, reference)
    
    # Compute distortion
    distortion = estimation - alpha * reference
    
    # Compute SI-SDR
    si_sdr_value = 10 * np.log10(
        np.dot(alpha * reference, alpha * reference) / np.dot(distortion, distortion)
    )
    
    return si_sdr_value


def sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Signal-to-Distortion Ratio.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        
    Returns:
        SDR value in dB.
    """
    # Remove DC component
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Compute SDR
    sdr_value = 10 * np.log10(
        np.dot(reference, reference) / np.dot(estimation - reference, estimation - reference)
    )
    
    return sdr_value


def pesq_score(reference: np.ndarray, estimation: np.ndarray, sample_rate: int = 16000) -> float:
    """PESQ (Perceptual Evaluation of Speech Quality) score.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        sample_rate: Sample rate.
        
    Returns:
        PESQ score.
    """
    try:
        # Ensure signals are float32
        reference = reference.astype(np.float32)
        estimation = estimation.astype(np.float32)
        
        # Compute PESQ
        pesq_value = pesq(sample_rate, reference, estimation, "wb")
        return pesq_value
    except Exception as e:
        logger.warning(f"PESQ computation failed: {e}")
        return 0.0


def stoi_score(reference: np.ndarray, estimation: np.ndarray, sample_rate: int = 16000) -> float:
    """STOI (Short-Time Objective Intelligibility) score.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        sample_rate: Sample rate.
        
    Returns:
        STOI score.
    """
    try:
        # Ensure signals are float32
        reference = reference.astype(np.float32)
        estimation = estimation.astype(np.float32)
        
        # Compute STOI
        stoi_value = stoi(reference, estimation, sample_rate, extended=False)
        return stoi_value
    except Exception as e:
        logger.warning(f"STOI computation failed: {e}")
        return 0.0


def mse(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Mean Squared Error.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        
    Returns:
        MSE value.
    """
    return np.mean((reference - estimation) ** 2)


def mae(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Mean Absolute Error.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        
    Returns:
        MAE value.
    """
    return np.mean(np.abs(reference - estimation))


def snr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Signal-to-Noise Ratio.
    
    Args:
        reference: Reference signal.
        estimation: Estimated signal.
        
    Returns:
        SNR value in dB.
    """
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - estimation) ** 2)
    
    if noise_power == 0:
        return float("inf")
    
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value


class AudioInpaintingMetrics:
    """Metrics calculator for audio inpainting tasks."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize metrics calculator.
        
        Args:
            sample_rate: Sample rate of audio signals.
        """
        self.sample_rate = sample_rate
        self.metrics = {}
    
    def compute_metrics(
        self,
        reference: Union[np.ndarray, torch.Tensor],
        estimation: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            reference: Reference audio signal.
            estimation: Estimated audio signal.
            mask: Optional mask indicating missing regions.
            
        Returns:
            Dictionary of metric values.
        """
        # Convert to numpy arrays
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
        if isinstance(estimation, torch.Tensor):
            estimation = estimation.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        # Ensure 1D arrays
        if reference.ndim > 1:
            reference = reference.flatten()
        if estimation.ndim > 1:
            estimation = estimation.flatten()
        
        # Ensure same length
        min_length = min(len(reference), len(estimation))
        reference = reference[:min_length]
        estimation = estimation[:min_length]
        
        metrics = {}
        
        # Basic metrics
        metrics["mse"] = mse(reference, estimation)
        metrics["mae"] = mae(reference, estimation)
        metrics["snr"] = snr(reference, estimation)
        metrics["sdr"] = sdr(reference, estimation)
        metrics["si_sdr"] = si_sdr(reference, estimation)
        
        # Perceptual metrics
        metrics["pesq"] = pesq_score(reference, estimation, self.sample_rate)
        metrics["stoi"] = stoi_score(reference, estimation, self.sample_rate)
        
        # Mask-specific metrics (if mask provided)
        if mask is not None:
            if mask.ndim > 1:
                mask = mask.flatten()
            mask = mask[:min_length].astype(bool)
            
            if np.any(mask):
                # Metrics on inpainted regions only
                metrics["mse_inpainted"] = mse(reference[mask], estimation[mask])
                metrics["mae_inpainted"] = mae(reference[mask], estimation[mask])
                metrics["snr_inpainted"] = snr(reference[mask], estimation[mask])
                metrics["si_sdr_inpainted"] = si_sdr(reference[mask], estimation[mask])
                
                # Metrics on non-missing regions (preservation)
                non_missing = ~mask
                if np.any(non_missing):
                    metrics["mse_preserved"] = mse(reference[non_missing], estimation[non_missing])
                    metrics["mae_preserved"] = mae(reference[non_missing], estimation[non_missing])
                    metrics["snr_preserved"] = snr(reference[non_missing], estimation[non_missing])
        
        return metrics
    
    def compute_batch_metrics(
        self,
        references: Union[np.ndarray, torch.Tensor],
        estimations: Union[np.ndarray, torch.Tensor],
        masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, List[float]]:
        """Compute metrics for a batch of samples.
        
        Args:
            references: Batch of reference signals.
            estimations: Batch of estimated signals.
            masks: Optional batch of masks.
            
        Returns:
            Dictionary of metric lists.
        """
        if isinstance(references, torch.Tensor):
            references = references.detach().cpu().numpy()
        if isinstance(estimations, torch.Tensor):
            estimations = estimations.detach().cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        
        batch_size = len(references)
        batch_metrics = {}
        
        for i in range(batch_size):
            ref = references[i]
            est = estimations[i]
            mask = masks[i] if masks is not None else None
            
            metrics = self.compute_metrics(ref, est, mask)
            
            for key, value in metrics.items():
                if key not in batch_metrics:
                    batch_metrics[key] = []
                batch_metrics[key].append(value)
        
        return batch_metrics
    
    def compute_average_metrics(
        self,
        references: Union[np.ndarray, torch.Tensor],
        estimations: Union[np.ndarray, torch.Tensor],
        masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute average metrics for a batch.
        
        Args:
            references: Batch of reference signals.
            estimations: Batch of estimated signals.
            masks: Optional batch of masks.
            
        Returns:
            Dictionary of average metric values.
        """
        batch_metrics = self.compute_batch_metrics(references, estimations, masks)
        
        average_metrics = {}
        for key, values in batch_metrics.items():
            average_metrics[key] = np.mean(values)
            average_metrics[f"{key}_std"] = np.std(values)
        
        return average_metrics


def create_metrics_calculator(sample_rate: int = 16000) -> AudioInpaintingMetrics:
    """Create metrics calculator.
    
    Args:
        sample_rate: Sample rate.
        
    Returns:
        Metrics calculator instance.
    """
    return AudioInpaintingMetrics(sample_rate)
