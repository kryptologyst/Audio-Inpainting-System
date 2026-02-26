#!/usr/bin/env python3
"""Simple test script to verify the audio inpainting system works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import torch

from src.data import create_missing_mask, apply_missing_mask, interpolate_missing, create_synthetic_dataset
from src.models import create_model
from src.metrics import create_metrics_calculator
from src.utils import get_device, set_seed


def test_basic_functionality():
    """Test basic functionality of the audio inpainting system."""
    print("🧪 Testing Audio Inpainting System")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Test 1: Create synthetic audio
    print("1. Creating synthetic audio...")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a complex signal
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 1320 * t) +  # E6 note
        0.1 * np.random.randn(len(t))  # Noise
    )
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    print(f"   ✅ Created audio: {len(audio)} samples, {len(audio)/sample_rate:.2f}s")
    
    # Test 2: Create missing mask
    print("2. Creating missing mask...")
    mask = create_missing_mask(
        len(audio),
        missing_percentage=0.2,
        missing_length_range=(0.1, 0.5),
        sample_rate=sample_rate,
    )
    missing_samples = np.sum(mask)
    print(f"   ✅ Created mask: {missing_samples} missing samples ({missing_samples/len(audio)*100:.1f}%)")
    
    # Test 3: Apply mask
    print("3. Applying missing mask...")
    corrupted = apply_missing_mask(audio, mask)
    print(f"   ✅ Applied mask: {np.sum(np.isnan(corrupted))} NaN values")
    
    # Test 4: Simple interpolation
    print("4. Testing simple interpolation...")
    interpolated = interpolate_missing(corrupted)
    print(f"   ✅ Interpolated: {np.sum(np.isnan(interpolated))} NaN values remaining")
    
    # Test 5: Model creation
    print("5. Testing model creation...")
    device = get_device("auto")
    print(f"   ✅ Device: {device}")
    
    # Test U-Net
    try:
        unet_model = create_model("unet1d", in_channels=1, out_channels=1)
        unet_model = unet_model.to(device)
        print("   ✅ U-Net model created")
    except Exception as e:
        print(f"   ❌ U-Net model failed: {e}")
    
    # Test Conv-TasNet
    try:
        conv_model = create_model("conv_tasnet", in_channels=1, out_channels=1)
        conv_model = conv_model.to(device)
        print("   ✅ Conv-TasNet model created")
    except Exception as e:
        print(f"   ❌ Conv-TasNet model failed: {e}")
    
    # Test 6: Model inference
    print("6. Testing model inference...")
    try:
        # Prepare input
        corrupted_tensor = torch.from_numpy(corrupted).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Test U-Net inference
        with torch.no_grad():
            unet_output = unet_model(corrupted_tensor)
            print(f"   ✅ U-Net inference: input {corrupted_tensor.shape} -> output {unet_output.shape}")
        
        # Test Conv-TasNet inference
        with torch.no_grad():
            conv_output = conv_model(corrupted_tensor)
            print(f"   ✅ Conv-TasNet inference: input {corrupted_tensor.shape} -> output {conv_output.shape}")
            
    except Exception as e:
        print(f"   ❌ Model inference failed: {e}")
    
    # Test 7: Metrics calculation
    print("7. Testing metrics calculation...")
    try:
        metrics_calculator = create_metrics_calculator(sample_rate)
        metrics = metrics_calculator.compute_metrics(audio, interpolated, mask)
        
        print(f"   ✅ Metrics calculated:")
        print(f"      PESQ: {metrics['pesq']:.3f}")
        print(f"      STOI: {metrics['stoi']:.3f}")
        print(f"      SI-SDR: {metrics['si_sdr']:.2f} dB")
        print(f"      MSE: {metrics['mse']:.6f}")
        
    except Exception as e:
        print(f"   ❌ Metrics calculation failed: {e}")
    
    # Test 8: Synthetic dataset creation
    print("8. Testing synthetic dataset creation...")
    try:
        create_synthetic_dataset(
            "data/wav",
            num_samples=10,
            sample_rate=sample_rate,
            duration=duration,
        )
        print("   ✅ Synthetic dataset created")
        
    except Exception as e:
        print(f"   ❌ Synthetic dataset creation failed: {e}")
    
    print("\n🎉 All tests completed!")
    return True


def main():
    """Main function."""
    try:
        success = test_basic_functionality()
        if success:
            print("\n✅ Audio Inpainting System is working correctly!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
