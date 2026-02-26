"""Streamlit demo for audio inpainting system."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_audio, save_audio, create_missing_mask, apply_missing_mask, interpolate_missing
from src.metrics import create_metrics_calculator
from src.models import create_model
from src.utils import get_device, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Audio Inpainting System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Privacy & Ethics Disclaimer</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>This system is designed for audio restoration and enhancement research</li>
        <li>It is NOT intended for biometric identification or voice cloning</li>
        <li>Any misuse for deceptive purposes is strictly prohibited</li>
        <li>Please respect privacy and ethical guidelines when using this tool</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎵 Audio Inpainting System</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.2rem; color: #666;">
    Fill in missing parts of audio signals using advanced neural networks
</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Simple Interpolation", "1D U-Net", "Conv-TasNet"],
    help="Choose the inpainting model to use"
)

# Parameters
st.sidebar.subheader("Inpainting Parameters")
missing_percentage = st.sidebar.slider(
    "Missing Percentage",
    min_value=0.05,
    max_value=0.8,
    value=0.2,
    step=0.05,
    help="Percentage of audio to mask as missing"
)

missing_length_min = st.sidebar.slider(
    "Min Missing Length (seconds)",
    min_value=0.05,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Minimum length of missing segments"
)

missing_length_max = st.sidebar.slider(
    "Max Missing Length (seconds)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Maximum length of missing segments"
)

# Load model function
@st.cache_resource
def load_model(model_name: str):
    """Load the selected model."""
    try:
        config = load_config("configs/config.yaml")
        device = get_device("auto")
        
        if model_name == "Simple Interpolation":
            return None, device  # No model needed for interpolation
        elif model_name == "1D U-Net":
            model_config = config.model.unet1d
            model = create_model("unet1d", **model_config)
        elif model_name == "Conv-TasNet":
            model_config = config.model.conv_tasnet
            model = create_model("conv_tasnet", **model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        model.eval()
        
        # Try to load checkpoint if available
        checkpoint_path = "checkpoints/best.pth"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            st.sidebar.success(f"Loaded trained {model_name} model")
        else:
            st.sidebar.warning(f"No trained checkpoint found for {model_name}")
        
        return model, device
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, device = load_model(model_type)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Input Audio")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "m4a"],
        help="Upload an audio file to inpaint missing regions"
    )
    
    # Or use sample
    if st.button("🎵 Use Sample Audio"):
        # Generate synthetic sample
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a more complex synthetic signal
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.2 * np.sin(2 * np.pi * 1320 * t) +  # E6 note
            0.1 * np.random.randn(len(t))  # Noise
        )
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            save_audio(audio, tmp_file.name, sample_rate)
            uploaded_file = tmp_file.name

# Process audio
if uploaded_file is not None:
    try:
        # Load audio
        if isinstance(uploaded_file, str):
            # Sample audio
            audio, sample_rate = load_audio(uploaded_file)
        else:
            # Uploaded file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio, sample_rate = load_audio(tmp_file.name)
        
        st.success(f"✅ Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Create missing mask
        mask = create_missing_mask(
            len(audio),
            missing_percentage=missing_percentage,
            missing_length_range=(missing_length_min, missing_length_max),
            sample_rate=sample_rate,
        )
        
        # Apply mask
        corrupted_audio = apply_missing_mask(audio, mask)
        
        # Inpaint
        if model_type == "Simple Interpolation":
            inpainted_audio = interpolate_missing(corrupted_audio)
        else:
            if model is not None and device is not None:
                # Convert to tensor
                audio_tensor = torch.from_numpy(corrupted_audio).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward pass
                with torch.no_grad():
                    predicted_tensor = model(audio_tensor)
                    inpainted_audio = predicted_tensor.squeeze().cpu().numpy()
            else:
                st.error("Model not available, falling back to interpolation")
                inpainted_audio = interpolate_missing(corrupted_audio)
        
        # Compute metrics
        metrics_calculator = create_metrics_calculator(sample_rate)
        metrics = metrics_calculator.compute_metrics(audio, inpainted_audio, mask)
        
        with col2:
            st.subheader("📊 Results")
            
            # Metrics
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric("PESQ", f"{metrics['pesq']:.3f}")
                st.metric("STOI", f"{metrics['stoi']:.3f}")
            
            with col2b:
                st.metric("SI-SDR", f"{metrics['si_sdr']:.2f} dB")
                st.metric("MSE", f"{metrics['mse']:.6f}")
            
            # Audio players
            st.subheader("🎧 Audio Playback")
            
            # Original
            st.write("**Original Audio:**")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                save_audio(audio, tmp_file.name, sample_rate)
                st.audio(tmp_file.name)
            
            # Corrupted
            st.write("**Corrupted Audio (Missing Regions):**")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                save_audio(corrupted_audio, tmp_file.name, sample_rate)
                st.audio(tmp_file.name)
            
            # Inpainted
            st.write("**Inpainted Audio:**")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                save_audio(inpainted_audio, tmp_file.name, sample_rate)
                st.audio(tmp_file.name)
            
            # Download buttons
            st.subheader("💾 Download Results")
            
            col2c, col2d = st.columns(2)
            
            with col2c:
                if st.button("Download Inpainted Audio"):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        save_audio(inpainted_audio, tmp_file.name, sample_rate)
                        with open(tmp_file.name, "rb") as f:
                            st.download_button(
                                label="Download WAV",
                                data=f.read(),
                                file_name="inpainted_audio.wav",
                                mime="audio/wav"
                            )
            
            with col2d:
                if st.button("Download Metrics"):
                    metrics_text = f"""Audio Inpainting Results
Model: {model_type}
Missing Percentage: {missing_percentage:.1%}
Missing Length Range: {missing_length_min:.2f}-{missing_length_max:.2f}s

Metrics:
- PESQ: {metrics['pesq']:.3f}
- STOI: {metrics['stoi']:.3f}
- SI-SDR: {metrics['si_sdr']:.2f} dB
- MSE: {metrics['mse']:.6f}
- MAE: {metrics['mae']:.6f}
- SNR: {metrics['snr']:.2f} dB
"""
                    st.download_button(
                        label="Download TXT",
                        data=metrics_text,
                        file_name="inpainting_metrics.txt",
                        mime="text/plain"
                    )
        
        # Visualization
        st.subheader("📈 Waveform Visualization")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        time = np.arange(len(audio)) / sample_rate
        
        # Original
        axes[0].plot(time, audio, "b-", alpha=0.7)
        axes[0].set_title("Original Audio")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        
        # Corrupted
        axes[1].plot(time, corrupted_audio, "r-", alpha=0.7)
        axes[1].fill_between(time, corrupted_audio, alpha=0.3, color="red")
        axes[1].set_title("Corrupted Audio (Missing Regions)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)
        
        # Inpainted
        axes[2].plot(time, inpainted_audio, "g-", alpha=0.7)
        axes[2].set_title("Inpainted Audio")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True)
        
        # Comparison
        axes[3].plot(time, audio, "b-", alpha=0.5, label="Original")
        axes[3].plot(time, inpainted_audio, "g-", alpha=0.7, label="Inpainted")
        axes[3].set_title("Comparison")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Amplitude")
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed metrics
        st.subheader("📋 Detailed Metrics")
        
        metrics_df = {
            "Metric": ["PESQ", "STOI", "SI-SDR (dB)", "SDR (dB)", "MSE", "MAE", "SNR (dB)"],
            "Value": [
                f"{metrics['pesq']:.3f}",
                f"{metrics['stoi']:.3f}",
                f"{metrics['si_sdr']:.2f}",
                f"{metrics['sdr']:.2f}",
                f"{metrics['mse']:.6f}",
                f"{metrics['mae']:.6f}",
                f"{metrics['snr']:.2f}",
            ]
        }
        
        import pandas as pd
        st.dataframe(pd.DataFrame(metrics_df), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        logger.error(f"Error processing audio: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Audio Inpainting System - Research & Education Demo</p>
    <p>⚠️ This tool is for research purposes only. Please use responsibly.</p>
</div>
""", unsafe_allow_html=True)
