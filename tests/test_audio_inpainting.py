"""Unit tests for audio inpainting system."""

import numpy as np
import pytest
import torch

from src.data import (
    AudioDataset,
    create_missing_mask,
    apply_missing_mask,
    interpolate_missing,
    normalize_audio,
    add_noise,
)
from src.models import create_model, get_model_info
from src.losses import create_loss_function
from src.metrics import create_metrics_calculator, si_sdr, pesq_score, stoi_score
from src.utils import get_device, set_seed, count_parameters


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_create_missing_mask(self):
        """Test missing mask creation."""
        audio_length = 1000
        mask = create_missing_mask(audio_length, missing_percentage=0.2)
        
        assert len(mask) == audio_length
        assert mask.dtype == bool
        assert np.sum(mask) <= audio_length * 0.3  # Allow some tolerance
    
    def test_apply_missing_mask(self):
        """Test applying missing mask."""
        audio = np.random.randn(100)
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True  # Missing region
        
        corrupted = apply_missing_mask(audio, mask)
        
        assert len(corrupted) == len(audio)
        assert np.all(np.isnan(corrupted[mask]))
        assert np.allclose(corrupted[~mask], audio[~mask])
    
    def test_interpolate_missing(self):
        """Test interpolation of missing samples."""
        # Create signal with missing middle section
        audio = np.sin(np.linspace(0, 4*np.pi, 100))
        mask = np.zeros(100, dtype=bool)
        mask[30:70] = True  # Missing middle section
        
        corrupted = apply_missing_mask(audio, mask)
        interpolated = interpolate_missing(corrupted)
        
        assert len(interpolated) == len(audio)
        assert not np.any(np.isnan(interpolated))
        assert np.allclose(interpolated[~mask], audio[~mask])
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        audio = np.random.randn(1000)
        normalized = normalize_audio(audio, target_rms=0.1)
        
        actual_rms = np.sqrt(np.mean(normalized**2))
        assert abs(actual_rms - 0.1) < 0.01
    
    def test_add_noise(self):
        """Test noise addition."""
        audio = np.sin(np.linspace(0, 2*np.pi, 100))
        noisy = add_noise(audio, snr_db=10)
        
        assert len(noisy) == len(audio)
        assert not np.allclose(audio, noisy)  # Should be different


class TestModels:
    """Test model creation and functionality."""
    
    def test_create_model(self):
        """Test model creation."""
        # Test U-Net
        model = create_model("unet1d", in_channels=1, out_channels=1)
        assert isinstance(model, torch.nn.Module)
        
        # Test Conv-TasNet
        model = create_model("conv_tasnet", in_channels=1, out_channels=1)
        assert isinstance(model, torch.nn.Module)
        
        # Test simple interpolation
        model = create_model("simple_interpolation")
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = create_model("unet1d", in_channels=1, out_channels=1)
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 1, 1000)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == x.shape
    
    def test_model_info(self):
        """Test model info extraction."""
        model = create_model("unet1d", in_channels=1, out_channels=1)
        info = get_model_info(model)
        
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info
        assert info["total_parameters"] > 0


class TestLosses:
    """Test loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss."""
        loss_fn = create_loss_function({"type": "mse"})
        
        pred = torch.randn(10)
        target = torch.randn(10)
        
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_config = {
            "type": "combined",
            "params": {
                "reconstruction_weight": 1.0,
                "perceptual_weight": 0.1,
                "spectral_weight": 0.1,
            }
        }
        loss_fn = create_loss_function(loss_config)
        
        pred = torch.randn(1, 1000)
        target = torch.randn(1, 1000)
        
        losses = loss_fn(pred, target)
        
        assert "total" in losses
        assert "reconstruction" in losses
        assert "perceptual" in losses
        assert "spectral" in losses


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_si_sdr(self):
        """Test SI-SDR calculation."""
        # Perfect reconstruction should give high SI-SDR
        reference = np.sin(np.linspace(0, 2*np.pi, 100))
        estimation = reference.copy()
        
        si_sdr_value = si_sdr(reference, estimation)
        assert si_sdr_value > 20  # Should be very high for perfect reconstruction
    
    def test_metrics_calculator(self):
        """Test metrics calculator."""
        calculator = create_metrics_calculator(sample_rate=16000)
        
        reference = np.sin(np.linspace(0, 2*np.pi, 1000))
        estimation = reference + 0.1 * np.random.randn(1000)
        
        metrics = calculator.compute_metrics(reference, estimation)
        
        assert "pesq" in metrics
        assert "stoi" in metrics
        assert "si_sdr" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
    
    def test_batch_metrics(self):
        """Test batch metrics computation."""
        calculator = create_metrics_calculator(sample_rate=16000)
        
        references = np.random.randn(3, 1000)
        estimations = references + 0.1 * np.random.randn(3, 1000)
        
        batch_metrics = calculator.compute_batch_metrics(references, estimations)
        
        assert "pesq" in batch_metrics
        assert len(batch_metrics["pesq"]) == 3


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate two random numbers
        r1 = np.random.rand()
        set_seed(42)
        r2 = np.random.rand()
        
        assert r1 == r2  # Should be identical with same seed
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        count = count_parameters(model)
        
        assert count == 55  # 10*5 + 5 bias terms


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_inpainting(self):
        """Test complete inpainting pipeline."""
        # Create synthetic audio
        audio = np.sin(np.linspace(0, 4*np.pi, 1000))
        
        # Create missing mask
        mask = create_missing_mask(len(audio), missing_percentage=0.2)
        
        # Apply mask
        corrupted = apply_missing_mask(audio, mask)
        
        # Interpolate
        inpainted = interpolate_missing(corrupted)
        
        # Check results
        assert len(inpainted) == len(audio)
        assert not np.any(np.isnan(inpainted))
        
        # Compute metrics
        calculator = create_metrics_calculator(sample_rate=16000)
        metrics = calculator.compute_metrics(audio, inpainted, mask)
        
        assert "pesq" in metrics
        assert "stoi" in metrics
        assert metrics["pesq"] > 0
        assert metrics["stoi"] > 0
    
    def test_model_training_step(self):
        """Test a single training step."""
        # Create model
        model = create_model("unet1d", in_channels=1, out_channels=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = create_loss_function({"type": "mse"})
        
        # Create dummy data
        audio = torch.randn(1, 1, 1000)
        corrupted = torch.randn(1, 1, 1000)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        predicted = model(corrupted)
        loss = loss_fn(predicted, audio)
        
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])
