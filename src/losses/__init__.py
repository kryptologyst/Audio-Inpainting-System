"""Loss functions for audio inpainting."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def __init__(self, reduction: str = "mean"):
        """Initialize MSE loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.mse_loss(pred, target, reduction=self.reduction)


class L1Loss(nn.Module):
    """L1 loss."""
    
    def __init__(self, reduction: str = "mean"):
        """Initialize L1 loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.l1_loss(pred, target, reduction=self.reduction)


class SpectralLoss(nn.Module):
    """Spectral loss in frequency domain."""
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        window: str = "hann",
        reduction: str = "mean",
    ):
        """Initialize spectral loss.
        
        Args:
            n_fft: FFT size.
            hop_length: Hop length.
            win_length: Window length.
            window: Window type.
            reduction: Reduction method.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute STFT
        pred_stft = torch.stft(
            pred.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=pred.device),
            return_complex=True,
        )
        
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=target.device),
            return_complex=True,
        )
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        loss = F.mse_loss(pred_mag, target_mag, reduction=self.reduction)
        
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features."""
    
    def __init__(self, reduction: str = "mean"):
        """Initialize perceptual loss.
        
        Args:
            reduction: Reduction method.
        """
        super().__init__()
        self.reduction = reduction
        
        # Simple perceptual loss using learned features
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 15, 1, 7),
            nn.ReLU(),
            nn.Conv1d(32, 64, 15, 1, 7),
            nn.ReLU(),
            nn.Conv1d(64, 128, 15, 1, 7),
            nn.ReLU(),
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        loss = F.mse_loss(pred_features, target_features, reduction=self.reduction)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        spectral_weight: float = 0.1,
        reconstruction_type: str = "mse",
    ):
        """Initialize combined loss.
        
        Args:
            reconstruction_weight: Weight for reconstruction loss.
            perceptual_weight: Weight for perceptual loss.
            spectral_weight: Weight for spectral loss.
            reconstruction_type: Type of reconstruction loss ('mse', 'l1').
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.spectral_weight = spectral_weight
        
        # Reconstruction loss
        if reconstruction_type == "mse":
            self.reconstruction_loss = MSELoss()
        elif reconstruction_type == "l1":
            self.reconstruction_loss = L1Loss()
        else:
            raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")
        
        # Perceptual loss
        self.perceptual_loss = PerceptualLoss()
        
        # Spectral loss
        self.spectral_loss = SpectralLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            pred: Predicted audio.
            target: Target audio.
            mask: Optional mask for loss computation.
            
        Returns:
            Dictionary of loss components.
        """
        losses = {}
        
        # Reconstruction loss
        if mask is not None:
            # Only compute loss on non-missing regions
            pred_masked = pred * (1 - mask.float())
            target_masked = target * (1 - mask.float())
            losses["reconstruction"] = self.reconstruction_loss(pred_masked, target_masked)
        else:
            losses["reconstruction"] = self.reconstruction_loss(pred, target)
        
        # Perceptual loss
        losses["perceptual"] = self.perceptual_loss(pred, target)
        
        # Spectral loss
        losses["spectral"] = self.spectral_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.reconstruction_weight * losses["reconstruction"]
            + self.perceptual_weight * losses["perceptual"]
            + self.spectral_weight * losses["spectral"]
        )
        losses["total"] = total_loss
        
        return losses


def create_loss_function(loss_config: Dict) -> nn.Module:
    """Create loss function from configuration.
    
    Args:
        loss_config: Loss configuration dictionary.
        
    Returns:
        Loss function.
    """
    loss_type = loss_config.get("type", "combined")
    
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "l1":
        return L1Loss()
    elif loss_type == "spectral":
        return SpectralLoss(**loss_config.get("params", {}))
    elif loss_type == "perceptual":
        return PerceptualLoss()
    elif loss_type == "combined":
        return CombinedLoss(**loss_config.get("params", {}))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
