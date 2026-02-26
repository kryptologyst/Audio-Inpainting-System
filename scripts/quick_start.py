#!/usr/bin/env python3
"""Quick start script for audio inpainting system."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and handle errors."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        "torch", "torchaudio", "librosa", "numpy", "pandas", 
        "soundfile", "scipy", "matplotlib", "streamlit"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages:")
        logger.info("pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True

def create_sample_data():
    """Create sample data for testing."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.data import create_synthetic_dataset
        
        data_dir = Path("data/wav")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating synthetic dataset...")
        create_synthetic_dataset(
            data_dir,
            num_samples=100,
            sample_rate=16000,
            duration=2.0,
        )
        
        logger.info("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")
        return False

def run_demo():
    """Run the Streamlit demo."""
    logger = logging.getLogger(__name__)
    
    demo_file = Path("demo/streamlit_demo.py")
    if not demo_file.exists():
        logger.error("Demo file not found")
        return False
    
    logger.info("Starting Streamlit demo...")
    logger.info("The demo will open in your browser at http://localhost:8501")
    logger.info("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_file)], check=True)
        return True
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Demo failed: {e}")
        return False

def run_training():
    """Run model training."""
    logger = logging.getLogger(__name__)
    
    config_file = Path("configs/config.yaml")
    if not config_file.exists():
        logger.error("Config file not found")
        return False
    
    logger.info("Starting model training...")
    logger.info("This may take several minutes depending on your hardware")
    
    cmd = f"{sys.executable} -m src.train --config {config_file}"
    return run_command(cmd, "Model training")

def run_evaluation():
    """Run model evaluation."""
    logger = logging.getLogger(__name__)
    
    checkpoint_file = Path("checkpoints/best.pth")
    if not checkpoint_file.exists():
        logger.warning("No trained checkpoint found, using untrained model")
    
    config_file = Path("configs/config.yaml")
    if not config_file.exists():
        logger.error("Config file not found")
        return False
    
    logger.info("Running model evaluation...")
    
    cmd = f"{sys.executable} -m src.eval --config {config_file} --checkpoint {checkpoint_file}"
    return run_command(cmd, "Model evaluation")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Audio Inpainting System - Quick Start")
    parser.add_argument(
        "action",
        choices=["demo", "train", "eval", "setup", "check"],
        help="Action to perform"
    )
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("🎵 Audio Inpainting System - Quick Start")
    logger.info("=" * 50)
    
    # Check dependencies unless skipped
    if not args.skip_deps and not check_dependencies():
        logger.error("Please install dependencies first: pip install -r requirements.txt")
        return 1
    
    # Perform requested action
    success = False
    
    if args.action == "check":
        success = check_dependencies()
    elif args.action == "setup":
        success = create_sample_data()
    elif args.action == "demo":
        success = run_demo()
    elif args.action == "train":
        success = run_training()
    elif args.action == "eval":
        success = run_evaluation()
    
    if success:
        logger.info("✅ Operation completed successfully")
        return 0
    else:
        logger.error("❌ Operation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
