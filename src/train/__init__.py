"""Training script for audio inpainting models."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import AudioDataset, create_synthetic_dataset
from src.losses import create_loss_function
from src.metrics import create_metrics_calculator
from src.models import create_model, get_model_info
from src.utils import EarlyStopping, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


class AudioInpaintingTrainer:
    """Trainer class for audio inpainting models."""
    
    def __init__(self, config_path: str):
        """Initialize trainer.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        
        # Setup
        set_seed(self.config.seed)
        setup_logging(self.config.logging.level, self.config.logging.log_dir)
        
        self.device = get_device(self.config.device)
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.config.logging.log_dir, exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_metrics()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Model info: {get_model_info(self.model)}")
    
    def _setup_data(self):
        """Setup data loaders."""
        # Create synthetic dataset if needed
        data_dir = Path("data/wav")
        if not data_dir.exists() or len(list(data_dir.glob("*.wav"))) == 0:
            logger.info("Creating synthetic dataset...")
            create_synthetic_dataset(
                data_dir,
                num_samples=1000,
                sample_rate=self.config.data.sample_rate,
                duration=2.0,
            )
        
        # Create datasets
        self.train_dataset = AudioDataset(
            data_dir,
            metadata_file=data_dir.parent / "metadata.csv",
            sample_rate=self.config.data.sample_rate,
            duration=2.0,
            split="train",
            augment=True,
        )
        
        self.val_dataset = AudioDataset(
            data_dir,
            metadata_file=data_dir.parent / "metadata.csv",
            sample_rate=self.config.data.sample_rate,
            duration=2.0,
            split="val",
            augment=False,
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def _setup_model(self):
        """Setup model."""
        model_config = self.config.model[self.config.model.name]
        self.model = create_model(self.config.model.name, **model_config)
        self.model = self.model.to(self.device)
        
        logger.info(f"Created model: {self.config.model.name}")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )
    
    def _setup_loss(self):
        """Setup loss function."""
        loss_config = {
            "type": "combined",
            "params": {
                "reconstruction_weight": self.config.loss.reconstruction_weight,
                "perceptual_weight": self.config.loss.perceptual_weight,
                "spectral_weight": 0.1,
                "reconstruction_type": "mse",
            },
        }
        self.loss_fn = create_loss_function(loss_config)
        self.loss_fn = self.loss_fn.to(self.device)
    
    def _setup_metrics(self):
        """Setup metrics calculator."""
        self.metrics_calculator = create_metrics_calculator(self.config.data.sample_rate)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "perceptual": 0.0, "spectral": 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch["audio"].to(self.device)
            
            # Create missing mask
            batch_size, audio_length = audio.shape
            masks = []
            corrupted_audio = []
            
            for i in range(batch_size):
                mask = self._create_missing_mask(audio_length)
                corrupted = self._apply_missing_mask(audio[i], mask)
                masks.append(mask)
                corrupted_audio.append(corrupted)
            
            corrupted_audio = torch.stack(corrupted_audio).to(self.device)
            masks = torch.stack(masks).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted = self.model(corrupted_audio.unsqueeze(1))
            
            # Compute loss
            losses = self.loss_fn(predicted.squeeze(1), audio, masks)
            
            # Backward pass
            losses["total"].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "recon": f"{losses['reconstruction'].item():.4f}",
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "perceptual": 0.0, "spectral": 0.0}
        epoch_metrics = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                audio = batch["audio"].to(self.device)
                
                # Create missing mask
                batch_size, audio_length = audio.shape
                masks = []
                corrupted_audio = []
                
                for i in range(batch_size):
                    mask = self._create_missing_mask(audio_length)
                    corrupted = self._apply_missing_mask(audio[i], mask)
                    masks.append(mask)
                    corrupted_audio.append(corrupted)
                
                corrupted_audio = torch.stack(corrupted_audio).to(self.device)
                masks = torch.stack(masks).to(self.device)
                
                # Forward pass
                predicted = self.model(corrupted_audio.unsqueeze(1))
                
                # Compute loss
                losses = self.loss_fn(predicted.squeeze(1), audio, masks)
                
                # Update losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                
                # Compute metrics (on first batch only to save time)
                if batch_idx == 0:
                    metrics = self.metrics_calculator.compute_average_metrics(
                        audio[:4].cpu().numpy(),
                        predicted.squeeze(1)[:4].cpu().numpy(),
                        masks[:4].cpu().numpy(),
                    )
                    epoch_metrics.update(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.4f}",
                    "pesq": f"{epoch_metrics.get('pesq', 0):.3f}",
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.val_loader)
        
        return epoch_losses, epoch_metrics
    
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
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, "checkpoints/latest.pth")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, "checkpoints/best.pth")
            logger.info(f"Saved best checkpoint at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """Train the model."""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            if epoch % self.config.training.validate_every == 0:
                val_losses, val_metrics = self.validate_epoch()
                self.val_losses.append(val_losses)
                
                # Update learning rate
                self.scheduler.step(val_losses["total"])
                
                # Check for best model
                is_best = val_losses["total"] < self.best_loss
                if is_best:
                    self.best_loss = val_losses["total"]
                
                # Save checkpoint
                if epoch % self.config.training.save_every == 0 or is_best:
                    self.save_checkpoint(is_best)
                
                # Early stopping
                if early_stopping(val_losses["total"], self.model):
                    logger.info("Early stopping triggered")
                    break
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_losses['total']:.4f}, "
                    f"Val Loss: {val_losses['total']:.4f}, "
                    f"PESQ: {val_metrics.get('pesq', 0):.3f}, "
                    f"STOI: {val_metrics.get('stoi', 0):.3f}"
                )
        
        logger.info("Training completed!")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train audio inpainting model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    trainer = AudioInpaintingTrainer(args.config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
