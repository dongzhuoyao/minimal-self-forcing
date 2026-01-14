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
from tqdm import tqdm

from toy_dataset import ToyDataset
from visualization import TrainingPlotter, create_video_gif, save_video_grid
from tiny_causal_wan import TinyCausalWanModel


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
        log_dir: str = "logs/training",
        save_interval: int = 10,
        log_interval: int = 5,
        viz_interval: int = 100
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
            viz_interval: Generate and save sample videos every N steps
        """
        self.generator = generator.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.viz_interval = viz_interval
        
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
        num_steps: int = 10000,
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
            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "step": self.step})
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
        """Log metrics (can be extended to use wandb, tensorboard, etc.)."""
        if self.step % self.log_interval == 0:
            print(f"Step {self.step}: Loss = {metrics['loss']:.4f}")
    
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
        prompts: Optional[List[str]] = None
    ):
        """
        Generate sample videos for visualization during training.
        
        Args:
            text_encoder: Text encoder for encoding prompts
            num_samples: Number of sample videos to generate
            num_frames: Number of frames per video
            num_frames_per_block: Frames per block for generation
            prompts: Optional list of prompts (uses defaults if None)
        """
        self.generator.eval()
        
        # Sample prompts from config or defaults
        if prompts is None:
            sample_prompts = [
                "A red circle moving horizontally",
                "A blue square rotating clockwise",
                "A yellow ball bouncing",
                "Color gradient transitioning from red to blue"
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
            
            # Save individual GIFs
            videos_list = []
            for i, video_tensor in enumerate(generated_videos):
                gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                create_video_gif(video_tensor, str(gif_path), fps=8)
                videos_list.append(video_tensor)
            
            # Save grid of all videos
            grid_path = self.samples_dir / f"step_{self.step:06d}_grid.png"
            save_video_grid(videos_list, str(grid_path), prompts=sample_prompts)
            
            print(f"  Saved sample videos to {self.samples_dir}")
        
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
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory (overrides config)")
    parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N steps (overrides config)")
    parser.add_argument("--log_interval", type=int, default=None, help="Log metrics every N steps (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device (overrides config)")
    
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
    
    # Extract config values with defaults
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('seed', 42)
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Training hyperparameters
    num_steps = training_cfg.get('num_steps', 10000)
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
    patch_size = tuple(model_cfg.get('patch_size', [1, 2, 2]))
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
    print("\n1. Creating toy dataset...")
    dataset = ToyDataset(
        num_samples=num_samples,
        width=video_width,
        height=video_height,
        num_frames=video_frames
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
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = SimplifiedTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        save_interval=save_interval,
        log_interval=log_interval,
        viz_interval=viz_interval,
        config=config
    )
    
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
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "step": trainer.step})
        pbar.update(1)
        
        # Print progress
        if trainer.step % args.log_interval == 0:
            print(f"Step {trainer.step}: Loss = {metrics['loss']:.4f}")
        
        # Save checkpoint
        if trainer.step % args.save_interval == 0:
            trainer._save_checkpoint()
        
        # Generate and save sample videos for visualization
        if trainer.step % viz_interval == 0 and trainer.step > 0:
            print(f"\nGenerating sample videos at step {trainer.step}...")
            gen_cfg = config.get('generation', {})
            viz_prompts = config.get('viz_prompts', [
                "A red circle moving horizontally",
                "A blue square rotating clockwise",
                "A yellow ball bouncing",
                "Color gradient transitioning from red to blue"
            ])
            trainer.generate_sample_videos(
                text_encoder=text_encoder,
                num_samples=gen_cfg.get('num_viz_samples', 4),
                num_frames=gen_cfg.get('viz_num_frames', 9),
                num_frames_per_block=num_frame_per_block,
                prompts=viz_prompts
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
    trainer.generate_sample_videos(
        text_encoder=text_encoder,
        num_samples=4,
        num_frames=9,
        num_frames_per_block=3
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.log_dir}")
    print(f"Plots saved to: {Path(args.log_dir) / 'plots'}")
    print(f"Sample videos saved to: {Path(args.log_dir) / 'samples'}")
    print("\nKey points:")
    print("1. Self-Forcing simulates inference during training")
    print("2. Videos are generated block-by-block autoregressively")
    print("3. Loss is computed on the generated videos")
    print("4. This bridges the train-test gap")


if __name__ == "__main__":
    main()
