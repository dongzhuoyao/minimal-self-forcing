"""
Simplified Trainer for Self-Forcing Tutorial

This module provides a simplified training loop for educational purposes.
Includes the trainer class, scheduler, text encoder, and main training script.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import os
import numpy as np

from toy_dataset import ToyDataset
from moving_mnist import MovingMNISTDataset
from visualization import TrainingPlotter, create_video_gif, save_video_grid
from tiny_causal_wan import TinyCausalWanModel
from dynamic_sf import SelfForcingEngine

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        save_interval: Optional[int] = None,
        log_interval: Optional[int] = None,
        viz_interval: Optional[int] = None,
        use_wandb: Optional[bool] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None
    ):
        """
        Args:
            generator: The video generation model
            optimizer: Optimizer for the generator
            scheduler: Noise scheduler
            config: Config dictionary with hyperparameters (used to fill in missing values)
            device: Device to train on (overrides config if provided)
            log_dir: Directory to save logs and checkpoints (overrides config if provided)
            save_interval: Save checkpoint every N steps (overrides config if provided)
            log_interval: Log metrics every N steps (overrides config if provided)
            viz_interval: Generate and save sample videos every N steps (overrides config if provided)
            use_wandb: Whether to use Weights & Biases for logging (overrides config if provided)
            wandb_project: W&B project name (overrides config if provided)
            wandb_entity: W&B entity/team name (overrides config if provided)
            wandb_name: W&B run name (overrides config if provided)
        """
        # Extract values from config if not provided
        if config is None:
            config = {}
        
        training_cfg = config.get('training', {})
        paths_cfg = config.get('paths', {})
        wandb_cfg = config.get('wandb', {})
        
        # Use provided values or fall back to config, then defaults
        self.device = device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = Path(log_dir or paths_cfg.get('log_dir', 'logs/training'))
        self.save_interval = save_interval or training_cfg.get('save_interval', 10)
        self.log_interval = log_interval or training_cfg.get('log_interval', 5)
        self.viz_interval = viz_interval or training_cfg.get('viz_interval', 100)
        
        # Wandb settings
        if use_wandb is None:
            use_wandb = wandb_cfg.get('enabled', False)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        wandb_project = wandb_project or wandb_cfg.get('project', 'self-forcing')
        wandb_entity = wandb_entity if wandb_entity is not None else wandb_cfg.get('entity', None)
        wandb_name = wandb_name if wandb_name is not None else wandb_cfg.get('name', None)
        
        self.generator = generator.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Load training config values
        self.num_frames_per_block = training_cfg.get('num_frames_per_block', 3)
        self.denoising_steps = training_cfg.get('denoising_steps', [1000, 750, 500, 250])
        self.context_noise = training_cfg.get('context_noise', 0)
        self.training_num_frames = training_cfg.get('num_frames', 21)
        self.video_height = training_cfg.get('video_height', 64)
        self.video_width = training_cfg.get('video_width', 64)
        self.gradient_clip_norm = training_cfg.get('gradient_clip_norm', 1.0)
        self.use_dmd_loss = training_cfg.get('use_dmd_loss', True)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples directory for visualizations
        self.samples_dir = self.log_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.step = 0
        self.metrics_history = {
            "loss": [],
            "step": []
        }
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name or f"self-forcing-{self.log_dir.name}",
                config=config,
                dir=str(self.log_dir)
            )
            # Log model architecture info
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": device
            })
            print(f"Initialized wandb: project={wandb_project}, name={wandb_name or self.log_dir.name}")
        elif use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")
    
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
        # For Self-Forcing, we generate frames (configurable)
        num_frames = getattr(self, 'training_num_frames', 21)
        channels = 3
        height = getattr(self, 'video_height', 64)
        width = getattr(self, 'video_width', 64)
        
        noise = torch.randn(
            batch_size, num_frames, channels, height, width,
            device=self.device
        )
        
        # Simulate Self-Forcing inference during training
        # This is the key: we generate autoregressively, just like inference
        generated_video = self.sf_engine.simulate_self_forcing(
            noise, conditional_dict
        )
        
        # Compute Self-Forcing loss
        # Self-Forcing is data-free: it does NOT use ground truth videos.
        # Uses DMD (Distribution Matching Distillation) for distribution matching.
        loss = self.sf_engine.compute_self_forcing_loss(
            generated_video, 
            conditional_dict=conditional_dict,
            use_dmd=self.use_dmd_loss
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional but recommended)
        gradient_clip_norm = getattr(self, 'gradient_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=gradient_clip_norm)
        
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
    def train(
        self,
        dataloader: DataLoader,
        num_steps: int = 1000,
        text_encoder: Optional[nn.Module] = None
    ):
        """
        Main training loop.
        
        Args:
            dataloader: DataLoader for training data
            num_steps: Number of training steps to perform
            text_encoder: Optional text encoder for encoding prompts
        """
        print(f"Starting training for {num_steps} steps...")
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
        
        # Create iterator that cycles through dataloader
        dataloader_iter = iter(dataloader)
        
        # Progress bar
        pbar = tqdm(range(num_steps), desc="Training")
        
        while self.step < num_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Restart iterator if dataloader is exhausted
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Encode text prompts
            with torch.no_grad():
                conditional_dict = text_encoder(batch["prompts"])
            
            # Training step
            metrics = self.train_step(batch, conditional_dict)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{metrics['loss']:.8f}", "step": self.step})
            pbar.update(1)
            
            # Logging
            if self.step % self.log_interval == 0:
                self._log_metrics(metrics)
            
            # Save checkpoint
            if self.step % self.save_interval == 0:
                self._save_checkpoint()
            
            # Check if we've reached the target number of steps
            if self.step >= num_steps:
                break
        
        pbar.close()
        print("\nTraining completed!")
        self._save_checkpoint(final=True)
        self._save_metrics()
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and optionally to wandb.
        
        Both console and wandb logging happen at log_interval.
        """
        if self.step % self.log_interval == 0:
            print(f"Step {self.step}: Loss = {metrics['loss']:.8f}")
            
            # Log to wandb at log_interval
            if self.use_wandb:
                wandb.log(metrics, step=self.step)
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
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
        self.metrics_history = checkpoint.get("metrics_history", {"loss": [], "step": []})
        print(f"Loaded checkpoint from {checkpoint_path} (step {self.step})")
    
    def generate_sample_videos(
        self,
        text_encoder: nn.Module,
        num_samples: int = 4,
        num_frames: int = 9,
        num_frames_per_block: int = 3,
        prompts: Optional[List[str]] = None,
        ground_truth_videos: Optional[torch.Tensor] = None,
        gif_fps: int = 2
    ):
        """
        Generate sample videos for visualization during training.
        
        Args:
            text_encoder: Text encoder for encoding prompts
            num_samples: Number of sample videos to generate
            num_frames: Number of frames per video
            num_frames_per_block: Frames per block for generation
            prompts: Optional list of prompts (uses defaults if None)
            ground_truth_videos: Optional ground truth videos [B, F, C, H, W] to log alongside generated videos
            gif_fps: Frames per second for GIF playback (default: 2, slower playback)
        """
        self.generator.eval()
        
        # Sample prompts from config or defaults
        if prompts is None:
            sample_prompts = [
                "A red circle moving horizontally",
                "A blue square rotating",
                "A green triangle moving diagonally",
                "A color gradient transitioning from red to blue"
            ][:num_samples]
        else:
            sample_prompts = prompts[:num_samples]
        
        with torch.no_grad():
            # Encode prompts
            conditional_dict = text_encoder(sample_prompts)
            
            # Create noise
            batch_size = len(sample_prompts)
            noise = torch.randn(
                batch_size, num_frames, 3, 64, 64,
                device=self.device
            )
            
            # Generate videos using simplified inference (full video, not just last 21 frames)
            generated_videos = self._generate_full_video(noise, conditional_dict, num_frames_per_block)
            
            # Convert from latents to pixel space (denormalize)
            # The model outputs in [-1, 1] range, convert to [0, 1]
            generated_videos = (generated_videos + 1.0) / 2.0
            generated_videos = generated_videos.clamp(0, 1)
            
            # Process ground truth videos if provided
            gt_videos_list = None
            if ground_truth_videos is not None:
                # Ensure ground truth is on correct device and has correct shape
                if ground_truth_videos.device != self.device:
                    ground_truth_videos = ground_truth_videos.to(self.device)
                
                # Ground truth videos are in [B, F, C, H, W] format
                # Convert from [-1, 1] to [0, 1] if needed
                if ground_truth_videos.min() < 0:
                    ground_truth_videos = (ground_truth_videos + 1.0) / 2.0
                ground_truth_videos = ground_truth_videos.clamp(0, 1)
                
                # Resize to match generated video dimensions if needed
                if ground_truth_videos.shape[2:] != generated_videos.shape[2:]:
                    # Resize spatial dimensions
                    gt_resized = []
                    for gt_vid in ground_truth_videos:
                        # Resize each frame
                        frames_resized = []
                        for frame in gt_vid:
                            frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                            from PIL import Image
                            img = Image.fromarray((frame_np * 255).astype(np.uint8))
                            target_size = (generated_videos.shape[-1], generated_videos.shape[-2])
                            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                            frame_resized = torch.from_numpy(np.array(img_resized)).float() / 255.0
                            frame_resized = frame_resized.permute(2, 0, 1)  # [C, H, W]
                            frames_resized.append(frame_resized)
                        gt_resized.append(torch.stack(frames_resized))
                    ground_truth_videos = torch.stack(gt_resized).to(self.device)
                
                # Take matching number of frames
                if ground_truth_videos.shape[1] != num_frames:
                    ground_truth_videos = ground_truth_videos[:, :num_frames]
                
                # Limit to num_samples
                ground_truth_videos = ground_truth_videos[:num_samples]
                
                gt_videos_list = []
                for i, gt_video in enumerate(ground_truth_videos):
                    gt_videos_list.append(gt_video)
            
            # Save individual GIFs for generated videos
            videos_list = []
            for i, video_tensor in enumerate(generated_videos):
                gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                create_video_gif(video_tensor, str(gif_path), fps=gif_fps)
                videos_list.append(video_tensor)
            
            # Save GIFs for ground truth videos if provided
            gt_gif_paths = []
            if gt_videos_list is not None:
                for i, gt_video in enumerate(gt_videos_list):
                    gt_gif_path = self.samples_dir / f"step_{self.step:06d}_gt_{i:02d}.gif"
                    create_video_gif(gt_video, str(gt_gif_path), fps=gif_fps)
                    gt_gif_paths.append(gt_gif_path)
            
            # Save grid of generated videos
            grid_path = self.samples_dir / f"step_{self.step:06d}_grid.png"
            save_video_grid(videos_list, str(grid_path), prompts=sample_prompts)
            
            # Save comparison grid if ground truth is available
            comparison_grid_path = None
            if gt_videos_list is not None:
                comparison_grid_path = self.samples_dir / f"step_{self.step:06d}_comparison_grid.png"
                # Create side-by-side comparison: generated | ground truth
                comparison_videos = []
                comparison_prompts = []
                for gen_vid, gt_vid, prompt in zip(videos_list, gt_videos_list, sample_prompts):
                    comparison_videos.append(gen_vid)
                    comparison_videos.append(gt_vid)
                    comparison_prompts.append(f"{prompt} (Generated)")
                    comparison_prompts.append(f"{prompt} (Ground Truth)")
                save_video_grid(comparison_videos, str(comparison_grid_path), prompts=comparison_prompts, ncols=2)
            
            print(f"  Saved sample videos to {self.samples_dir}")
            
            # Log to wandb
            if self.use_wandb:
                # Log grid image of generated videos
                wandb.log({
                    "samples/grid": wandb.Image(str(grid_path)),
                }, step=self.step)
                
                # Log comparison grid if ground truth is available
                if comparison_grid_path is not None:
                    wandb.log({
                        "samples/comparison_grid": wandb.Image(str(comparison_grid_path)),
                    }, step=self.step)
                
                # Create a summary table of prompts for better visibility
                prompts_table = wandb.Table(
                    columns=["Sample", "Prompt"],
                    data=[[i, prompt] for i, prompt in enumerate(sample_prompts)]
                )
                wandb.log({
                    "samples/prompts_table": prompts_table
                }, step=self.step)
                
                # Log individual generated videos as GIFs with captions
                for i, (video_tensor, prompt) in enumerate(zip(videos_list, sample_prompts)):
                    gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                    wandb.log({
                        f"samples/generated_video_{i}": wandb.Video(
                            str(gif_path), 
                            format="gif",
                            caption=f"Generated: {prompt}"
                        ),
                    }, step=self.step)
                
                # Log ground truth videos if available with captions
                if gt_gif_paths:
                    for i, (gt_gif_path, prompt) in enumerate(zip(gt_gif_paths, sample_prompts)):
                        wandb.log({
                            f"samples/ground_truth_video_{i}": wandb.Video(
                                str(gt_gif_path), 
                                format="gif",
                                caption=f"Ground Truth: {prompt}"
                            ),
                        }, step=self.step)
                
                # Also log prompts as text for easy reference (one log entry per prompt)
                for i, prompt in enumerate(sample_prompts):
                    wandb.log({
                        f"samples/prompt_{i}": prompt
                    }, step=self.step)
        
        self.generator.train()
    
    def _generate_full_video(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        num_frames_per_block: int = 3
    ) -> torch.Tensor:
        """
        Generate full video for visualization (simplified version without gradient control).
        
        Args:
            noise: Initial noise tensor of shape [B, F, C, H, W]
            conditional_dict: Conditional information
            num_frames_per_block: Frames per block
        
        Returns:
            Generated video of shape [B, F, C, H, W]
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        denoising_steps = getattr(self, 'denoising_steps', [1000, 750, 500, 250])
        
        # Calculate number of blocks
        assert num_frames % num_frames_per_block == 0
        num_blocks = num_frames // num_frames_per_block
        
        # Initialize output tensor
        output = torch.zeros_like(noise)
        current_start_frame = 0
        
        # Generate block by block
        for block_idx in range(num_blocks):
            # Get noise for this block
            block_noise = noise[:, current_start_frame:current_start_frame + num_frames_per_block]
            noisy_input = block_noise
            
            # Denoising loop
            for step_idx, timestep in enumerate(denoising_steps):
                timestep_tensor = torch.full(
                    (batch_size, num_frames_per_block),
                    timestep,
                    device=self.device,
                    dtype=torch.long
                )
                
                # Forward pass
                denoised, _ = self.generator(noisy_input, timestep_tensor, conditional_dict)
                
                # If not last step, add noise for next iteration
                if step_idx < len(denoising_steps) - 1:
                    next_timestep = denoising_steps[step_idx + 1]
                    noise_to_add = torch.randn_like(denoised)
                    alpha = 1.0 - (next_timestep / 1000.0)
                    noisy_input = alpha * denoised + (1 - alpha) * noise_to_add
                else:
                    # Last step: store result
                    output[:, current_start_frame:current_start_frame + num_frames_per_block] = denoised
            
            current_start_frame += num_frames_per_block
        
        return output


class SimpleScheduler:
    """Simple noise scheduler."""
    
    def __init__(self):
        self.timesteps = torch.linspace(1000, 0, 1000)
    
    def add_noise(self, x, noise, timestep):
        """Add noise to clean data."""
        alpha = 1.0 - (timestep.float() / 1000.0)
        alpha = alpha.clamp(0, 1)
        
        if len(x.shape) == 4:
            alpha = alpha.view(-1, 1, 1, 1)
        elif len(x.shape) == 5:
            alpha = alpha.view(-1, 1, 1, 1, 1)
        
        return alpha * x + (1 - alpha) * noise


class SimpleTextEncoder(nn.Module):
    """Simple deterministic text encoder for tutorial.
    
    RECOMMENDED APPROACH: Character-level tokenizer with learned embeddings
    
    Why this is good for tutorials:
    1. Educational: Shows how tokenization works step-by-step
    2. Deterministic: Same text always produces same embeddings (enables learning)
    3. Learnable: Embeddings can be trained to capture text-video relationships
    4. Simple: No external dependencies, works with any text
    5. Transparent: Easy to understand and debug
    
    Alternative approaches (see comments below):
    - Word-level tokenizer: More semantic but requires vocabulary building
    - Bag-of-words: Simplest but loses sequence information
    - Pre-trained models (CLIP/T5): Best quality but adds complexity
    """
    
    def __init__(self, device="cuda", text_dim=128, text_len=77, vocab_size=256):
        super().__init__()
        self.device = device
        self.text_dim = text_dim
        self.text_len = text_len
        self.vocab_size = vocab_size
        
        # Character embedding lookup table (learnable)
        # Maps character tokens (0-255 for ASCII) to embeddings
        self.char_embedding = nn.Embedding(vocab_size, text_dim)
        
        # Learned projection layer for better representations
        self.proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )
        
        # Move all modules to the specified device
        self.to(device)
    
    def _tokenize(self, text: str) -> list:
        """Character-level tokenization.
        
        Converts each character to its ASCII/UTF-8 byte value (0-255).
        This is deterministic - same text always produces same tokens.
        
        Example:
            "A red circle" -> [65, 32, 114, 101, 100, 32, 99, 105, 114, 99, 108, 101]
            "A" = 65, space = 32, "r" = 114, etc.
        """
        tokens = []
        for char in text:
            # Get character code (0-255 for ASCII, can handle UTF-8)
            byte_val = ord(char)
            # Clamp to vocab_size
            token = min(byte_val, self.vocab_size - 1)
            tokens.append(token)
        return tokens
    
    def forward(self, text_prompts):
        """Encode text prompts deterministically.
        
        Process:
        1. Tokenize: Convert text to character tokens
        2. Pad/Truncate: Ensure fixed length (text_len)
        3. Embed: Lookup embeddings for each token
        4. Project: Apply learned transformation
        
        Args:
            text_prompts: List of text strings
                Example: ["A red circle moving horizontally", "A blue square rotating"]
            
        Returns:
            Dictionary with 'prompt_embeds' of shape [B, text_len, text_dim]
        """
        batch_size = len(text_prompts)
        prompt_embeds_list = []
        
        for prompt in text_prompts:
            # Step 1: Tokenize (text -> character tokens)
            tokens = self._tokenize(prompt)
            
            # Step 2: Pad or truncate to fixed length
            if len(tokens) < self.text_len:
                # Pad with 0 (null character) - deterministic padding
                tokens = tokens + [0] * (self.text_len - len(tokens))
            else:
                # Truncate to text_len
                tokens = tokens[:self.text_len]
            
            # Step 3: Convert to tensor
            token_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)  # [text_len]
            
            # Step 4: Lookup embeddings (learnable - this is what gets trained!)
            embed_seq = self.char_embedding(token_tensor)  # [text_len, text_dim]
            
            # Step 5: Apply learned projection
            embed_seq = self.proj(embed_seq)  # [text_len, text_dim]
            
            prompt_embeds_list.append(embed_seq)
        
        # Stack to create batch
        prompt_embeds = torch.stack(prompt_embeds_list, dim=0)  # [B, text_len, text_dim]
        
        return {
            "prompt_embeds": prompt_embeds
        }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Self-Forcing model (tutorial)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of training samples (overrides config)")
    parser.add_argument("--dataset", type=str, default=None, choices=['toy', 'moving_mnist'], help="Dataset type: 'toy' or 'moving_mnist' (overrides config)")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory (overrides config)")
    parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N steps (overrides config)")
    parser.add_argument("--log_interval", type=int, default=None, help="Log metrics every N steps (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device (overrides config)")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="self-forcing", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name (optional)")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint from train0.py")

    args = parser.parse_args()
    
    # Load config file
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")
        config = {}
    
    # Override config with command-line arguments if provided
    if args.num_steps is not None:
        config.setdefault('training', {})['num_steps'] = args.num_steps
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        config.setdefault('training', {})['lr'] = args.lr
    if args.num_samples is not None:
        config.setdefault('training', {})['num_samples'] = args.num_samples
    if args.log_dir is not None:
        config.setdefault('paths', {})['log_dir'] = args.log_dir
    if args.save_interval is not None:
        config.setdefault('training', {})['save_interval'] = args.save_interval
    if args.log_interval is not None:
        config.setdefault('training', {})['log_interval'] = args.log_interval
    if args.device is not None:
        config['device'] = args.device
    if args.use_wandb:
        config.setdefault('wandb', {})['enabled'] = True
    if args.wandb_project is not None:
        config.setdefault('wandb', {})['project'] = args.wandb_project
    if args.wandb_entity is not None:
        config.setdefault('wandb', {})['entity'] = args.wandb_entity
    if args.wandb_name is not None:
        config.setdefault('wandb', {})['name'] = args.wandb_name
    
    # Extract config values with defaults
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})
    wandb_cfg = config.get('wandb', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('seed', 42)
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Training hyperparameters
    num_steps = training_cfg.get('num_steps', 1000)
    batch_size = training_cfg.get('batch_size', 16)
    lr = training_cfg.get('lr', 1e-4)
    weight_decay = training_cfg.get('weight_decay', 0.01)
    num_samples = training_cfg.get('num_samples', 20)
    save_interval = training_cfg.get('save_interval', 10)
    log_interval = training_cfg.get('log_interval', 5)
    viz_interval = training_cfg.get('viz_interval', 100)
    log_dir = paths_cfg.get('log_dir', 'logs/training')
    video_height = training_cfg.get('video_height', 64)
    video_width = training_cfg.get('video_width', 64)
    video_frames = training_cfg.get('video_frames', 9)
    
    # Model hyperparameters
    model_dim = model_cfg.get('dim', 256)
    model_ffn_dim = model_cfg.get('ffn_dim', 1024)
    model_num_heads = model_cfg.get('num_heads', 4)
    model_num_layers = model_cfg.get('num_layers', 4)
    patch_size = tuple(model_cfg.get('patch_size', [1, 4, 4]))
    text_dim = model_cfg.get('text_dim', 128)
    freq_dim = model_cfg.get('freq_dim', 256)
    num_frame_per_block = model_cfg.get('num_frame_per_block', 3)
    
    args = type('Args', (), {
        'num_steps': num_steps,
        'batch_size': batch_size,
        'lr': lr,
        'num_samples': num_samples,
        'log_dir': log_dir,
        'save_interval': save_interval,
        'log_interval': log_interval,
        'device': device,
        'config': config
    })()
    
    print("=" * 70)
    print("Self-Forcing Training Tutorial")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Create dataset
    dataset_cfg = config.get('dataset', {})
    # Command-line argument overrides config
    dataset_type = getattr(args, 'dataset', None) or dataset_cfg.get('type', 'toy')
    dataset_type = dataset_type.lower()
    
    print(f"\n1. Creating {dataset_type} dataset...")
    
    if dataset_type == 'moving_mnist':
        # Moving MNIST dataset configuration
        num_digits = dataset_cfg.get('num_digits', 1)
        digit_size = dataset_cfg.get('digit_size', 64)
        max_velocity = dataset_cfg.get('max_velocity', 2.0)
        
        dataset = MovingMNISTDataset(
            num_samples=num_samples,
            width=video_width,
            height=video_height,
            num_frames=video_frames,
            seed=seed,
            num_digits=num_digits,
            digit_size=digit_size,
            max_velocity=max_velocity
        )
    else:  # Default to 'toy'
        dataset = ToyDataset(
            num_samples=num_samples,
            width=video_width,
            height=video_height,
            num_frames=video_frames,
            seed=seed
        )
    
    print(f"   Created {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    print("\n2. Creating model...")
    # Use TinyCausalWanModel
    generator = TinyCausalWanModel(
        in_dim=model_cfg.get('in_dim', 3),
        out_dim=model_cfg.get('out_dim', 3),
        dim=model_dim,
        ffn_dim=model_ffn_dim,
        num_heads=model_num_heads,
        num_layers=model_num_layers,
        patch_size=patch_size,
        text_dim=text_dim,
        freq_dim=freq_dim,
        num_frame_per_block=num_frame_per_block,
    )
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"   Using TinyCausalWanModel (transformer backbone)")
    
    # Create text encoder
    text_encoder_cfg = config.get('text_encoder', {})
    text_encoder = SimpleTextEncoder(
        device=device,
        text_dim=text_encoder_cfg.get('text_dim', text_dim),
        text_len=text_encoder_cfg.get('text_len', 77),
        vocab_size=text_encoder_cfg.get('vocab_size', 256)
    )
    
    # Create optimizer (include text encoder parameters so they can be trained)
    optimizer = torch.optim.AdamW(
        list(generator.parameters()) + list(text_encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = SimpleScheduler()
    
    # Create trainer (most values come from config, command-line args override)
    print("\n3. Creating trainer...")
    trainer = SimplifiedTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        # Command-line args override config values
        device=getattr(args, 'device', None),
        log_dir=getattr(args, 'log_dir', None),
        save_interval=getattr(args, 'save_interval', None),
        log_interval=getattr(args, 'log_interval', None),
        viz_interval=None,  # Not exposed as CLI arg, use config
        use_wandb=getattr(args, 'use_wandb', None),
        wandb_project=getattr(args, 'wandb_project', None),
        wandb_entity=getattr(args, 'wandb_entity', None),
        wandb_name=getattr(args, 'wandb_name', None)
    )
    
    # Load pretrained checkpoint if provided
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"\n   Loading pretrained checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load generator weights
            if "generator_state_dict" in checkpoint:
                generator.load_state_dict(checkpoint["generator_state_dict"])
                print(f"   Loaded generator weights from checkpoint")

            # Optionally load text encoder weights if present
            if "text_encoder_state_dict" in checkpoint:
                text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
                print(f"   Loaded text encoder weights from checkpoint")

            # Check if this was a pretraining checkpoint
            training_type = checkpoint.get("training_type", "unknown")
            pretrain_step = checkpoint.get("step", 0)
            print(f"   Checkpoint type: {training_type}, trained for {pretrain_step} steps")
        else:
            print(f"   Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

    # Training plotter
    plotter = TrainingPlotter(save_dir=str(Path(args.log_dir) / "plots"))

    # Training loop with plotting
    print("\n4. Starting training...")
    print("-" * 70)
    
    # Create iterator that cycles through dataloader
    dataloader_iter = iter(dataloader)
    
    # Progress bar
    pbar = tqdm(range(args.num_steps), desc="Training")
    
    while trainer.step < args.num_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart iterator if dataloader is exhausted
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # DataLoader batches "prompt" (singular) from dataset as a list
        # Trainer expects "prompts" (plural), so rename it
        batch["prompts"] = batch["prompt"]
        
        # Encode prompts
        with torch.no_grad():
            conditional_dict = text_encoder(batch["prompts"])
        
        # Training step
        metrics = trainer.train_step(batch, conditional_dict)
        
        # Log to plotter
        plotter.log_metric("loss", metrics["loss"], trainer.step)
        
        # Log metrics (console + wandb) at log_interval
        if trainer.step % args.log_interval == 0:
            trainer._log_metrics(metrics)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{metrics['loss']:.8f}", "step": trainer.step})
        pbar.update(1)
        
        # Save checkpoint
        if trainer.step % args.save_interval == 0:
            trainer._save_checkpoint()
        
        # Generate and save sample videos for visualization
        if trainer.step % viz_interval == 0 and trainer.step > 0:
            print(f"\nGenerating sample videos at step {trainer.step}...")
            gen_cfg = config.get('generation', {})
            viz_prompts = config.get('viz_prompts', [
                "A red circle moving horizontally",
                "A blue square moving vertically",
                "A green triangle moving diagonally",
                "A color gradient transitioning from red to blue"
            ])
            
            # Get ground truth videos from batch if available
            ground_truth_videos = None
            if "video" in batch:
                # Batch contains ground truth videos
                # DataLoader returns list of tensors, each is [F, C, H, W]
                batch_videos = batch["video"]
                num_viz_samples = gen_cfg.get('num_viz_samples', 4)
                
                if isinstance(batch_videos, list):
                    # Stack into tensor [B, F, C, H, W]
                    # Each element in list is [F, C, H, W]
                    batch_videos = [v.to(trainer.device) for v in batch_videos[:num_viz_samples]]
                    if batch_videos:
                        ground_truth_videos = torch.stack(batch_videos)  # [B, F, C, H, W]
                elif isinstance(batch_videos, torch.Tensor):
                    # Already a tensor, might be [B, F, C, H, W] or [F, C, H, W]
                    ground_truth_videos = batch_videos[:num_viz_samples].to(trainer.device)
                    if len(ground_truth_videos.shape) == 4:
                        # If [F, C, H, W], add batch dimension
                        ground_truth_videos = ground_truth_videos.unsqueeze(0)
            
            trainer.generate_sample_videos(
                text_encoder=text_encoder,
                num_samples=gen_cfg.get('num_viz_samples', 4),
                num_frames=gen_cfg.get('viz_num_frames', 9),
                num_frames_per_block=num_frame_per_block,
                prompts=viz_prompts,
                ground_truth_videos=ground_truth_videos,
                gif_fps=gen_cfg.get('gif_fps', 2)
            )
        
        # Check if we've reached the target number of steps
        if trainer.step >= args.num_steps:
            break
    
    pbar.close()
    
    # Finalize
    print("\n5. Finalizing...")
    trainer._save_checkpoint(final=True)
    trainer._save_metrics()
    
    # Plot training curves
    plotter.plot_metric("loss", title="Training Loss")
    plotter.save_history(str(Path(args.log_dir) / "metrics_history.json"))
    
    # Generate final sample videos
    print("\nGenerating final sample videos...")
    # Try to get ground truth videos from a sample batch
    ground_truth_videos = None
    try:
        sample_batch = next(iter(dataloader))
        if "video" in sample_batch:
            batch_videos = sample_batch["video"]
            if isinstance(batch_videos, list):
                batch_videos = [v.to(trainer.device) for v in batch_videos[:4]]
                if batch_videos:
                    ground_truth_videos = torch.stack(batch_videos)
            elif isinstance(batch_videos, torch.Tensor):
                ground_truth_videos = batch_videos[:4].to(trainer.device)
                if len(ground_truth_videos.shape) == 4:
                    ground_truth_videos = ground_truth_videos.unsqueeze(0)
    except:
        pass  # No ground truth available, that's okay
    
    # Get gif_fps from config
    gen_cfg = config.get('generation', {})
    trainer.generate_sample_videos(
        text_encoder=text_encoder,
        num_samples=4,
        num_frames=9,
        num_frames_per_block=3,
        ground_truth_videos=ground_truth_videos,
        gif_fps=gen_cfg.get('gif_fps', 2)
    )
    
    # Finish wandb run
    if trainer.use_wandb:
        wandb.finish()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.log_dir}")
    print(f"Plots saved to: {Path(args.log_dir) / 'plots'}")
    print(f"Sample videos saved to: {Path(args.log_dir) / 'samples'}")
    if trainer.use_wandb:
        print(f"W&B run completed (check wandb dashboard)")
    print("\nKey points:")
    print("1. Self-Forcing simulates inference during training")
    print("2. Videos are generated block-by-block autoregressively")
    print("3. Loss is computed on the generated videos")
    print("4. This bridges the train-test gap")


if __name__ == "__main__":
    main()
