#!/usr/bin/env python3
"""Setup script for audio inpainting system."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🎵 Audio Inpainting System - Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    directories = [
        "data/wav",
        "data/meta", 
        "checkpoints",
        "logs",
        "assets",
        "tests",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Try pip install first
    if run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("✅ Dependencies installed successfully")
    else:
        print("❌ Failed to install dependencies")
        print("Please install manually: pip install -r requirements.txt")
        return 1
    
    # Create synthetic dataset
    print("\n🎵 Creating synthetic dataset...")
    
    try:
        from src.data import create_synthetic_dataset
        
        create_synthetic_dataset(
            "data/wav",
            num_samples=100,
            sample_rate=16000,
            duration=2.0,
        )
        print("✅ Synthetic dataset created")
        
    except Exception as e:
        print(f"❌ Failed to create synthetic dataset: {e}")
        return 1
    
    # Test installation
    print("\n🧪 Testing installation...")
    
    try:
        import torch
        import torchaudio
        import librosa
        import numpy as np
        import pandas as pd
        import soundfile as sf
        import scipy
        import matplotlib
        import streamlit
        
        print("✅ All required packages imported successfully")
        
        # Test device detection
        from src.utils import get_device
        device = get_device("auto")
        print(f"✅ Device detection working: {device}")
        
        # Test model creation
        from src.models import create_model
        model = create_model("simple_interpolation")
        print("✅ Model creation working")
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/streamlit_demo.py")
    print("2. Train a model: python -m src.train --config configs/config.yaml")
    print("3. Run tests: pytest tests/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
