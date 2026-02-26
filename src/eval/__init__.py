"""Evaluation script for audio inpainting models."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import AudioDataset
from src.metrics import create_metrics_calculator
from src.models import create_model
from src.utils import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


class AudioInpaintingEvaluator:
    """Evaluator class for audio inpainting models."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """Initialize evaluator.
        
        Args:
            config_path: Path to configuration file.
            checkpoint_path: Path to model checkpoint.
        """
        self.config = load_config(config_path)
        
        # Setup
        set_seed(self.config.seed)
        setup_logging(self.config.logging.level, self.config.logging.log_dir)
        
        self.device = get_device(self.config.device)
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Setup metrics
        self.metrics_calculator = create_metrics_calculator(self.config.data.sample_rate)
        
        # Create output directory
        os.makedirs("assets", exist_ok=True)
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model."""
        # Create model
        model_config = self.config.model[self.config.model.name]
        self.model = create_model(self.config.model.name, **model_config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
    
    def evaluate_dataset(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        split: str = "test",
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            data_dir: Directory containing audio files.
            metadata_file: Path to metadata CSV file.
            split: Dataset split to evaluate.
            num_samples: Number of samples to evaluate (None for all).
            
        Returns:
            Dictionary of average metrics.
        """
        # Create dataset
        dataset = AudioDataset(
            data_dir,
            metadata_file=metadata_file,
            sample_rate=self.config.data.sample_rate,
            duration=2.0,
            split=split,
            augment=False,
        )
        
        # Limit samples if specified
        if num_samples:
            dataset.metadata = dataset.metadata.head(num_samples)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one at a time for detailed analysis
            shuffle=False,
            num_workers=0,
        )
        
        logger.info(f"Evaluating {len(dataset)} samples from {split} split")
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                audio = batch["audio"].to(self.device)
                audio_id = batch["id"][0]
                
                # Create missing mask
                audio_length = len(audio[0])
                mask = self._create_missing_mask(audio_length)
                corrupted_audio = self._apply_missing_mask(audio[0], mask)
                
                # Forward pass
                predicted = self.model(corrupted_audio.unsqueeze(0).unsqueeze(0))
                predicted = predicted.squeeze(0).squeeze(0)
                
                # Compute metrics
                metrics = self.metrics_calculator.compute_metrics(
                    audio[0].cpu().numpy(),
                    predicted.cpu().numpy(),
                    mask.cpu().numpy(),
                )
                
                all_metrics.append(metrics)
                
                # Save sample results
                if batch_idx < 5:  # Save first 5 samples
                    self._save_sample_results(
                        audio[0].cpu().numpy(),
                        corrupted_audio.cpu().numpy(),
                        predicted.cpu().numpy(),
                        mask.cpu().numpy(),
                        audio_id,
                        metrics,
                    )
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(dataset)} samples")
        
        # Compute average metrics
        average_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            average_metrics[key] = np.mean(values)
            average_metrics[f"{key}_std"] = np.std(values)
        
        return average_metrics
    
    def _create_missing_mask(self, audio_length: int) -> torch.Tensor:
        """Create missing mask for audio."""
        from src.data import create_missing_mask
        
        mask = create_missing_mask(
            audio_length,
            missing_percentage=self.config.augmentation.missing_prob,
            missing_length_range=self.config.augmentation.missing_length_range,
            sample_rate=self.config.data.sample_rate,
        )
        
        return torch.from_numpy(mask).bool()
    
    def _apply_missing_mask(self, audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply missing mask to audio."""
        corrupted = audio.clone()
        corrupted[mask] = 0.0  # Set missing samples to 0
        return corrupted
    
    def _save_sample_results(
        self,
        original: np.ndarray,
        corrupted: np.ndarray,
        predicted: np.ndarray,
        mask: np.ndarray,
        audio_id: str,
        metrics: Dict[str, float],
    ):
        """Save sample results for visualization."""
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        time = np.arange(len(original)) / self.config.data.sample_rate
        
        # Original audio
        axes[0].plot(time, original, "b-", alpha=0.7)
        axes[0].set_title("Original Audio")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        
        # Corrupted audio
        axes[1].plot(time, corrupted, "r-", alpha=0.7)
        axes[1].fill_between(time, corrupted, alpha=0.3, color="red")
        axes[1].set_title("Corrupted Audio (Missing Regions)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)
        
        # Predicted audio
        axes[2].plot(time, predicted, "g-", alpha=0.7)
        axes[2].set_title("Predicted Audio")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True)
        
        # Comparison
        axes[3].plot(time, original, "b-", alpha=0.5, label="Original")
        axes[3].plot(time, predicted, "g-", alpha=0.7, label="Predicted")
        axes[3].set_title("Comparison")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Amplitude")
        axes[3].legend()
        axes[3].grid(True)
        
        # Add metrics text
        metrics_text = f"PESQ: {metrics['pesq']:.3f}, STOI: {metrics['stoi']:.3f}, SI-SDR: {metrics['si_sdr']:.2f} dB"
        fig.suptitle(f"Audio Inpainting Results - {audio_id}\n{metrics_text}")
        
        plt.tight_layout()
        plt.savefig(f"assets/sample_{audio_id}.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Save audio files
        from src.data import save_audio
        
        save_audio(original, f"assets/original_{audio_id}.wav", self.config.data.sample_rate)
        save_audio(corrupted, f"assets/corrupted_{audio_id}.wav", self.config.data.sample_rate)
        save_audio(predicted, f"assets/predicted_{audio_id}.wav", self.config.data.sample_rate)
    
    def create_leaderboard(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a leaderboard from evaluation results.
        
        Args:
            results: Dictionary of results for different models/configurations.
            
        Returns:
            DataFrame with leaderboard.
        """
        # Prepare data for DataFrame
        leaderboard_data = []
        
        for model_name, metrics in results.items():
            row = {"Model": model_name}
            row.update(metrics)
            leaderboard_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by PESQ score (descending)
        df = df.sort_values("pesq", ascending=False)
        
        # Save to CSV
        df.to_csv("assets/leaderboard.csv", index=False)
        
        return df
    
    def run_ablation_study(self, data_dir: str) -> Dict[str, Dict[str, float]]:
        """Run ablation study with different configurations.
        
        Args:
            data_dir: Directory containing test data.
            
        Returns:
            Dictionary of results for different configurations.
        """
        logger.info("Running ablation study...")
        
        results = {}
        
        # Test different missing percentages
        missing_percentages = [0.1, 0.2, 0.3, 0.5]
        
        for missing_pct in missing_percentages:
            logger.info(f"Testing missing percentage: {missing_pct}")
            
            # Temporarily modify config
            original_missing_prob = self.config.augmentation.missing_prob
            self.config.augmentation.missing_prob = missing_pct
            
            # Evaluate
            metrics = self.evaluate_dataset(data_dir, split="test", num_samples=50)
            results[f"missing_{missing_pct}"] = metrics
            
            # Restore original config
            self.config.augmentation.missing_prob = original_missing_prob
        
        return results


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate audio inpainting model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--data_dir", type=str, default="data/wav", help="Data directory")
    parser.add_argument("--metadata", type=str, help="Metadata file path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    
    args = parser.parse_args()
    
    evaluator = AudioInpaintingEvaluator(args.config, args.checkpoint)
    
    if args.ablation:
        # Run ablation study
        results = evaluator.run_ablation_study(args.data_dir)
        
        # Create leaderboard
        leaderboard = evaluator.create_leaderboard(results)
        print("\nAblation Study Results:")
        print(leaderboard.to_string(index=False))
        
    else:
        # Standard evaluation
        metrics = evaluator.evaluate_dataset(
            args.data_dir,
            metadata_file=args.metadata,
            split=args.split,
            num_samples=args.num_samples,
        )
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
