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
            device: Device to train on
            log_dir: Directory to save logs and checkpoints
            save_interval: Save checkpoint every N steps
            log_interval: Log metrics every N steps
        """
        self.generator = generator.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        # For Self-Forcing, we generate 21 frames (7 blocks × 3 frames per block)
        # This matches the standard training setup where last 21 frames are used for loss
        num_frames = 21  # 7 blocks × 3 frames per block
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
        
        # Compute Self-Forcing loss
        # Self-Forcing is data-free: it does NOT use ground truth videos.
        # Instead, it uses distribution matching (DMD/SiD/CausVid) to match
        # the distribution of generated videos to real videos.
        # 
        # In the full implementation, this would use a discriminator/score network
        # for distribution matching. For tutorial, we use a simplified loss that
        # encourages temporal consistency and reasonable values.
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
        
        This implements block-by-block autoregressive generation with KV caching,
        matching the official Self-Forcing implementation.
        
        Key features:
        - Block-by-block generation (not all frames at once)
        - KV cache for efficient autoregressive generation
        - Gradient control: only last 21 frames compute gradients
        - Random timestep selection per block for efficiency
        
        Args:
            noise: Initial noise tensor of shape [B, F, C, H, W]
            conditional_dict: Conditional information
        
        Returns:
            Generated video latents of shape [B, F, C, H, W]
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # Configuration
        num_frame_per_block = 3  # Frames per block
        denoising_steps = [1000, 750, 500, 250]  # Denoising timesteps
        context_noise = 0  # Noise level for cache update
        
        # Calculate number of blocks
        assert num_frames % num_frame_per_block == 0, \
            f"num_frames ({num_frames}) must be divisible by num_frame_per_block ({num_frame_per_block})"
        num_blocks = num_frames // num_frame_per_block
        
        # Initialize output tensor
        output = torch.zeros_like(noise)
        
        # Initialize KV cache (simplified - in full impl this would be more complex)
        # For tutorial, we'll use a simple approach: track which frames have gradients
        kv_cache = None  # Simplified: actual KV cache would be model-specific
        
        # Gradient control: only compute gradients for last 21 frames
        num_output_frames = num_frames
        start_gradient_frame_index = max(0, num_output_frames - 21)
        
        # Random exit flags: which timestep to compute gradients on (for efficiency)
        # In full impl, this is synchronized across distributed processes
        num_denoising_steps = len(denoising_steps)
        exit_flags = [
            torch.randint(0, num_denoising_steps, (1,), device=self.device).item()
            for _ in range(num_blocks)
        ]
        
        # Block-by-block generation
        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks
        
        for block_index, current_num_frames in enumerate(all_num_frames):
            # Get noise for this block
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames
            ]
            
            # Spatial denoising loop (multiple timesteps)
            for index, current_timestep in enumerate(denoising_steps):
                exit_flag = (index == exit_flags[block_index])
                
                # Create timestep tensor for this block
                timestep = torch.full(
                    (batch_size, current_num_frames),
                    current_timestep,
                    device=self.device,
                    dtype=torch.long
                )
                
                if not exit_flag:
                    # Intermediate timesteps: no gradients (for efficiency)
                    with torch.no_grad():
                        denoised_pred, _ = self.generator(
                            noisy_input, timestep, conditional_dict
                        )
                    
                    # Add noise for next timestep
                    if index < len(denoising_steps) - 1:
                        next_timestep = denoising_steps[index + 1]
                        noise_to_add = torch.randn_like(denoised_pred)
                        alpha = 1.0 - (next_timestep / 1000.0)
                        noisy_input = alpha * denoised_pred + (1 - alpha) * noise_to_add
                else:
                    # Selected timestep: compute gradients only for last 21 frames
                    if current_start_frame < start_gradient_frame_index:
                        # Early blocks: no gradients
                        with torch.no_grad():
                            denoised_pred, _ = self.generator(
                                noisy_input, timestep, conditional_dict
                            )
                    else:
                        # Later blocks: gradients enabled
                        denoised_pred, _ = self.generator(
                            noisy_input, timestep, conditional_dict
                        )
                    break
            
            # Store output for this block
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            # Update KV cache (simplified - in full impl, this reruns with timestep=0)
            # For tutorial, we simulate cache update by running generator again with no_grad
            # This matches the official implementation where cache is updated after each block
            context_timestep = torch.full(
                (batch_size, current_num_frames),
                context_noise,
                device=self.device,
                dtype=torch.long
            )
            
            # Add context noise for cache update (if context_noise > 0)
            if context_noise > 0:
                context_noisy = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    context_noise * torch.ones(
                        batch_size * current_num_frames,
                        device=self.device,
                        dtype=torch.long
                    )
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                context_noisy = denoised_pred
            
            # Update cache (detached, no gradients)
            # In full implementation, this would update the actual KV cache structure
            # For tutorial, we simulate this by running generator (cache update happens internally)
            with torch.no_grad():
                _ = self.generator(
                    context_noisy, context_timestep, conditional_dict
                )
            
            # Move to next block
            current_start_frame += current_num_frames
        
        # Return only last 21 frames for loss computation (matching official impl)
        if output.shape[1] > 21:
            return output[:, -21:]
        return output
    
    def _compute_self_forcing_loss(
        self,
        generated_video: torch.Tensor,
        prompts: list
    ) -> torch.Tensor:
        """
        Compute Self-Forcing loss.
        
        Self-Forcing is data-free and does NOT require ground truth videos.
        In the full implementation, this would use distribution matching:
        - DMD (Distribution Matching Distillation)
        - SiD (Score Identity Distillation)  
        - CausVid (Causal Video model)
        
        These methods use a discriminator/score network to match distributions
        rather than matching individual pixels to ground truth.
        
        For tutorial, we use a simplified version that encourages:
        - Temporal consistency (smooth transitions between frames)
        - Regularization (reasonable values)
        
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
