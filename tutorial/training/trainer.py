"""
Simplified Trainer for Self-Forcing Tutorial

This module provides a simplified training loop for educational purposes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import os
from tqdm import tqdm


class SimplifiedTrainer:
    """
    Simplified trainer for Self-Forcing algorithm.
    
    This trainer demonstrates the core training loop without the complexity
    of distributed training, FSDP, etc.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        loss_fn: nn.Module,
        device: str = "cuda",
        log_dir: str = "tutorial/logs",
        save_interval: int = 10,
        log_interval: int = 5
    ):
        """
        Args:
            generator: The video generation model
            optimizer: Optimizer for the generator
            scheduler: Noise scheduler
            loss_fn: Loss function
            device: Device to train on
            log_dir: Directory to save logs and checkpoints
            save_interval: Save checkpoint every N steps
            log_interval: Log metrics every N steps
        """
        self.generator = generator.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.log_dir = Path(log_dir)
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.metrics_history = {
            "loss": [],
            "step": []
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of data containing prompts and optionally videos
            conditional_dict: Conditional information (e.g., text embeddings)
        
        Returns:
            Dictionary of metrics for this step
        """
        self.generator.train()
        self.optimizer.zero_grad()
        
        # Get batch info
        batch_size = len(batch["prompts"])
        prompts = batch["prompts"]
        
        # Sample noise for video generation
        # Shape: (batch_size, num_frames, channels, height, width)
        num_frames = 9  # 3 blocks × 3 frames per block
        channels, height, width = 3, 64, 64
        
        noise = torch.randn(
            batch_size, num_frames, channels, height, width,
            device=self.device
        )
        
        # Simulate Self-Forcing inference during training
        # This is the key: we generate autoregressively, just like inference
        generated_video = self._simulate_self_forcing(
            noise, conditional_dict
        )
        
        # Compute loss
        # In simplified version, we use MSE loss
        # In full implementation, this would be distribution matching loss
        if "video" in batch:
            # If we have ground truth video (for supervised learning demo)
            target_video = batch["video"].to(self.device)
            loss = self.loss_fn(generated_video, target_video)
        else:
            # For Self-Forcing, we typically use distribution matching
            # For tutorial, we'll use a simple reconstruction loss
            # In practice, this would use a discriminator/score network
            loss = self._compute_self_forcing_loss(generated_video, prompts)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update metrics
        metrics = {
            "loss": loss.item(),
            "step": self.step
        }
        
        self.metrics_history["loss"].append(loss.item())
        self.metrics_history["step"].append(self.step)
        
        self.step += 1
        
        return metrics
    
    def _simulate_self_forcing(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Simulate Self-Forcing inference during training.
        
        This is a simplified version. In the full implementation, this would
        use the SelfForcingTrainingPipeline with proper KV caching.
        
        Args:
            noise: Initial noise tensor
            conditional_dict: Conditional information
        
        Returns:
            Generated video latents
        """
        # Simplified: just do standard denoising
        # In full implementation, this would be block-by-block with KV cache
        batch_size, num_frames = noise.shape[:2]
        
        # Denoising steps
        denoising_steps = [1000, 750, 500, 250]
        noisy_input = noise
        
        for step_idx, timestep in enumerate(denoising_steps):
            # Create timestep tensor
            timestep_tensor = torch.full(
                (batch_size, num_frames),
                timestep,
                device=self.device,
                dtype=torch.long
            )
            
            # Forward pass
            denoised, _ = self.generator(noisy_input, timestep_tensor, conditional_dict)
            
            # If not last step, add noise for next iteration
            if step_idx < len(denoising_steps) - 1:
                next_timestep = denoising_steps[step_idx + 1]
                # Simplified noise addition
                noise_to_add = torch.randn_like(denoised)
                alpha = 1.0 - (next_timestep / 1000.0)
                noisy_input = alpha * denoised + (1 - alpha) * noise_to_add
            else:
                return denoised
        
        return denoised
    
    def _compute_self_forcing_loss(
        self,
        generated_video: torch.Tensor,
        prompts: list
    ) -> torch.Tensor:
        """
        Compute Self-Forcing loss.
        
        In the full implementation, this would use distribution matching
        (DMD, SiD, CausVid). For tutorial, we use a simplified version.
        
        Args:
            generated_video: Generated video tensor
            prompts: List of text prompts
        
        Returns:
            Loss tensor
        """
        # Simplified: use a simple regularization loss
        # In practice, this would use a discriminator or score network
        
        # Temporal consistency loss (encourage smooth transitions)
        temporal_loss = 0.0
        for t in range(generated_video.shape[1] - 1):
            frame_diff = generated_video[:, t+1] - generated_video[:, t]
            temporal_loss += torch.mean(frame_diff ** 2)
        temporal_loss = temporal_loss / (generated_video.shape[1] - 1)
        
        # Regularization loss (encourage reasonable values)
        reg_loss = torch.mean(generated_video ** 2)
        
        # Combined loss
        total_loss = temporal_loss + 0.1 * reg_loss
        
        return total_loss
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 10,
        text_encoder: Optional[nn.Module] = None
    ):
        """
        Main training loop.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            text_encoder: Optional text encoder for encoding prompts
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Log directory: {self.log_dir}")
        
        # Simple text encoder if not provided
        if text_encoder is None:
            # Dummy encoder: just create random embeddings
            class DummyTextEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                
                def forward(self, text_prompts):
                    batch_size = len(text_prompts)
                    return {
                        "text_embeddings": torch.randn(
                            batch_size, 77, 768, device=self.device
                        )
                    }
            
            text_encoder = DummyTextEncoder().to(self.device)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Encode text prompts
                with torch.no_grad():
                    conditional_dict = text_encoder(batch["prompts"])
                
                # Training step
                metrics = self.train_step(batch, conditional_dict)
                epoch_losses.append(metrics["loss"])
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
                
                # Logging
                if self.step % self.log_interval == 0:
                    self._log_metrics(metrics)
                
                # Save checkpoint
                if self.step % self.save_interval == 0:
                    self._save_checkpoint()
            
            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        print("\nTraining completed!")
        self._save_checkpoint(final=True)
        self._save_metrics()
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics (can be extended to use wandb, tensorboard, etc.)."""
        if self.step % self.log_interval == 0:
            print(f"Step {self.step}: Loss = {metrics['loss']:.4f}")
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history
        }
        
        suffix = "final" if final else f"step_{self.step:06d}"
        checkpoint_path = self.log_dir / f"checkpoint_{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if final or self.step % self.save_interval == 0:
            print(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Save training metrics to file."""
        import json
        
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.metrics_history = checkpoint.get("metrics_history", {"loss": [], "step": []})
        print(f"Loaded checkpoint from {checkpoint_path}")
