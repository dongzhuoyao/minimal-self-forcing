"""
Pretraining Script for Self-Forcing Tutorial

This module provides supervised pretraining using ground truth videos.
The pretrained checkpoints can be loaded by train.py for Self-Forcing fine-tuning.

Pretraining uses standard diffusion training:
- Add noise to ground truth videos at random timesteps
- Train model to denoise (predict clean video or noise)
- This gives the model a good initialization before Self-Forcing

Full implementation features:
- Standard diffusion training with ground truth videos
- Block-by-block processing support
- Checkpoint saving compatible with train.py
- Wandb integration for logging
- Sample video generation during training

Usage:
    python train0.py                           # Use default config
    python train0.py training.num_steps=2000   # Override config values
    python train0.py paths.log_dir=logs/exp1   # Change log directory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from moving_mnist import MovingMNISTDataset, TrainingPlotter, create_video_gif, save_video_grid
from tiny_causal_wan import TinyCausalWanModel

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PretrainingTrainer:
    """
    Trainer for supervised pretraining with ground truth videos.

    Uses standard diffusion training:
    - Sample random timestep t
    - Add noise to ground truth video at timestep t
    - Train model to predict denoised video (or noise)
    """

    def __init__(
        self,
        generator: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        cfg: DictConfig,
        output_dir: Path,
        text_encoder: Optional[nn.Module] = None
    ):
        """
        Args:
            generator: The video generation model
            optimizer: Optimizer for the generator
            scheduler: Noise scheduler
            cfg: Hydra config object
            text_encoder: Optional text encoder to save in checkpoints
        """
        self.device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        self.log_dir = output_dir
        self.save_interval = cfg.training.save_interval
        self.log_interval = cfg.training.log_interval
        self.viz_interval = cfg.training.viz_interval

        # Wandb settings
        self.use_wandb = cfg.wandb.enabled and WANDB_AVAILABLE

        self.generator = generator.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.cfg = cfg

        # Training config
        self.num_frames_per_block = cfg.training.num_frames_per_block
        self.num_train_timestep = cfg.training.get('num_train_timestep', 1000)
        self.min_timestep = cfg.training.get('min_timestep', None)
        self.max_timestep = cfg.training.get('max_timestep', None)
        # Use min_step/max_step if not explicitly set (matching official impl)
        if self.min_timestep is None:
            self.min_timestep = int(0.02 * self.num_train_timestep)  # Default: 20
        if self.max_timestep is None:
            self.max_timestep = int(0.98 * self.num_train_timestep)  # Default: 980
        self.gradient_clip_norm = cfg.training.gradient_clip_norm
        self.prediction_type = cfg.training.get('prediction_type', 'flow')  # Default to flow matching

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create samples directory
        self.samples_dir = self.log_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.metrics_history = {
            "loss": [],
            "step": []
        }

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb.get('project', 'self-forcing-pretrain'),
                entity=cfg.wandb.get('entity', None),
                name=cfg.wandb.get('name', None) or f"pretrain-{self.log_dir.name}",
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(self.log_dir)
            )
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": self.device,
                "training_type": "pretraining"
            })
            print(f"Initialized wandb: project={cfg.wandb.project}")
        elif cfg.wandb.enabled and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform one pretraining step."""
        self.generator.train()
        self.optimizer.zero_grad()

        # Get ground truth video [B, F, C, H, W]
        video = batch["video"]
        if isinstance(video, list):
            video = torch.stack(video)
        video = video.to(self.device)

        batch_size, num_frames = video.shape[:2]

        # Sample random timestep per block (matching official implementation)
        # Different blocks can have different timesteps, but frames within same block share timestep
        num_blocks = num_frames // self.num_frames_per_block
        assert num_frames % self.num_frames_per_block == 0, \
            f"num_frames ({num_frames}) must be divisible by num_frames_per_block ({self.num_frames_per_block})"
        
        # Sample timestep for each frame initially [B, F]
        timesteps = torch.randint(
            self.min_timestep, self.max_timestep,
            (batch_size, num_frames),
            device=self.device,
            dtype=torch.long
        )
        
        # Make timestep the same within each block (matching official impl)
        # Reshape to [B, num_blocks, num_frames_per_block]
        timesteps_expanded = timesteps.reshape(
            batch_size, num_blocks, self.num_frames_per_block
        )
        # Copy first frame's timestep to all frames in the block
        timesteps_expanded[:, :, 1:] = timesteps_expanded[:, :, 0:1]
        # Reshape back to [B, F]
        timesteps_expanded = timesteps_expanded.reshape(batch_size, num_frames)

        # Sample noise
        noise = torch.randn_like(video)

        # Add noise to video
        noisy_video = self.scheduler.add_noise(
            video.flatten(0, 1),
            noise.flatten(0, 1),
            timesteps_expanded.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))

        # Forward pass - model returns flow_pred (flow matching)
        flow_pred, _ = self.generator(noisy_video, timesteps_expanded, conditional_dict)

        # Compute training target (Flow Matching: noise - sample)
        training_target = self.scheduler.training_target(
            video.flatten(0, 1),
            noise.flatten(0, 1),
            timesteps_expanded.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))

        # Compute loss with timestep weighting (matching official impl)
        if self.prediction_type == 'flow':
            # Flow Matching loss
            loss = F.mse_loss(
                flow_pred.float(), 
                training_target.float(), 
                reduction='none'
            )
            # Per-frame loss: mean over spatial dimensions [B, F]
            loss = loss.mean(dim=(2, 3, 4))
            # Apply timestep weighting
            weights = self.scheduler.training_weight(timesteps_expanded)
            loss = loss * weights
            loss = loss.mean()
        elif self.prediction_type == 'sample':
            # Convert flow_pred to x0_pred for sample prediction
            x0_pred = self.scheduler.convert_flow_to_x0(
                flow_pred.flatten(0, 1),
                noisy_video.flatten(0, 1),
                timesteps_expanded.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            loss = F.mse_loss(x0_pred, video)
        else:  # noise prediction
            # Convert flow_pred to noise prediction
            noise_pred = self.scheduler.convert_flow_to_noise(
                flow_pred.flatten(0, 1),
                noisy_video.flatten(0, 1),
                timesteps_expanded.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            max_norm=self.gradient_clip_norm
        )

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

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        print(f"Step {self.step}: Loss = {metrics['loss']:.8f}")

        if self.use_wandb:
            wandb.log(metrics, step=self.step)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint compatible with train.py."""
        checkpoint = {
            "step": self.step,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "training_type": "pretraining"
        }

        # Save text encoder weights if available
        if self.text_encoder is not None:
            checkpoint["text_encoder_state_dict"] = self.text_encoder.state_dict()

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
        prompts: Optional[List[str]] = None,
        ground_truth_videos: Optional[torch.Tensor] = None,
        gif_fps: int = 2
    ):
        """Generate sample videos for visualization."""
        self.generator.eval()

        if prompts is None:
            sample_prompts = [
                "A red circle moving horizontally",
                "A blue square moving vertically",
                "A green triangle moving diagonally",
                "A color gradient transitioning"
            ][:num_samples]
        else:
            sample_prompts = prompts[:num_samples]

        with torch.no_grad():
            conditional_dict = text_encoder(sample_prompts)

            batch_size = len(sample_prompts)
            height, width = 64, 64

            # Start from pure noise
            noise = torch.randn(
                batch_size, num_frames, 3, height, width,
                device=self.device
            )

            # Simple iterative denoising with Flow Matching
            denoising_steps = [1000, 750, 500, 250, 0]
            x = noise

            for i, t in enumerate(denoising_steps[:-1]):
                timestep = torch.full(
                    (batch_size, num_frames),
                    t,
                    device=self.device,
                    dtype=torch.long
                )

                flow_pred, _ = self.generator(x, timestep, conditional_dict)
                
                # Convert flow_pred to x0_pred for next step
                x0_pred = self.scheduler.convert_flow_to_x0(
                    flow_pred.flatten(0, 1),
                    x.flatten(0, 1),
                    timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frames))

                # Move towards prediction
                if i < len(denoising_steps) - 2:
                    next_t = denoising_steps[i + 1]
                    # Add noise for next timestep
                    noise_new = torch.randn_like(x0_pred)
                    next_timestep = torch.full(
                        (batch_size, num_frames),
                        next_t,
                        device=self.device,
                        dtype=torch.long
                    )
                    x = self.scheduler.add_noise(
                        x0_pred.flatten(0, 1),
                        noise_new.flatten(0, 1),
                        next_timestep.flatten(0, 1)
                    ).unflatten(0, (batch_size, num_frames))
                else:
                    x = x0_pred

            generated_videos = x

            # Normalize to [0, 1]
            generated_videos = (generated_videos + 1.0) / 2.0
            generated_videos = generated_videos.clamp(0, 1)

            # Save GIFs
            videos_list = []
            for i, video_tensor in enumerate(generated_videos):
                gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                create_video_gif(video_tensor, str(gif_path), fps=gif_fps)
                videos_list.append(video_tensor)

            # Save grid
            grid_path = self.samples_dir / f"step_{self.step:06d}_grid.png"
            save_video_grid(videos_list, str(grid_path), prompts=sample_prompts)

            print(f"  Saved sample videos to {self.samples_dir}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "samples/grid": wandb.Image(str(grid_path)),
                }, step=self.step)

                for i, (video_tensor, prompt) in enumerate(zip(videos_list, sample_prompts)):
                    gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                    wandb.log({
                        f"samples/video_{i}": wandb.Video(
                            str(gif_path),
                            format="gif",
                            caption=prompt
                        ),
                    }, step=self.step)

        self.generator.train()


class SimpleScheduler:
    """
    Simple noise scheduler for diffusion with Flow Matching support.
    
    Matches official FlowMatchScheduler structure:
    - Uses sigma-based noise schedule
    - Supports training_target (flow = noise - sample)
    - Supports training_weight (timestep-dependent weights)
    - Supports flow <-> x0 conversions
    """

    def __init__(self, num_train_timesteps=1000, sigma_max=1.0, sigma_min=0.003/1.002):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        
        # Create timesteps [0, 1000] -> [1000, 0] for compatibility
        # Don't register as buffer - will move to device dynamically
        self.timesteps = torch.linspace(num_train_timesteps, 0, num_train_timesteps)
        
        # Create sigmas for Flow Matching: sigma(t) = t * (sigma_max - sigma_min) + sigma_min
        # When t=0: sigma=sigma_min, when t=1: sigma=sigma_max
        t_normalized = torch.linspace(0, 1, num_train_timesteps)
        self.sigmas = t_normalized * (sigma_max - sigma_min) + sigma_min
        
        # Linear timestep weights (matching official impl)
        # Higher weights for middle timesteps
        self.linear_timesteps_weights = torch.ones(num_train_timesteps)
        # Optional: can adjust weights here if needed

    def add_noise(self, x, noise, timestep):
        """
        Add noise to clean data at given timestep (Flow Matching).
        
        Args:
            x: Clean data [B*F, C, H, W] or [B*F, ...]
            noise: Noise tensor [B*F, C, H, W] or [B*F, ...]
            timestep: Timestep tensor [B*F] (values in [0, 1000])
            
        Returns:
            Noisy data: (1 - sigma) * x + sigma * noise
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        
        # Move timesteps and sigmas to same device as input
        device = timestep.device
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        
        # Convert timestep to sigma
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id]
        
        # Reshape sigma for broadcasting
        if len(x.shape) == 4:
            sigma = sigma.view(-1, 1, 1, 1)
        elif len(x.shape) == 5:
            sigma = sigma.view(-1, 1, 1, 1, 1)
        
        # Flow Matching: (1 - sigma) * x + sigma * noise
        return (1 - sigma) * x + sigma * noise
    
    def training_target(self, sample, noise, timestep):
        """
        Compute training target for Flow Matching.
        
        Args:
            sample: Clean sample [B*F, C, H, W]
            noise: Noise tensor [B*F, C, H, W]
            timestep: Timestep tensor [B*F]
            
        Returns:
            Flow target: noise - sample
        """
        return noise - sample
    
    def training_weight(self, timestep):
        """
        Get timestep-dependent training weights.
        
        Args:
            timestep: Timestep tensor [B, F] or [B*F]
            
        Returns:
            Weights tensor with same shape as timestep
        """
        if timestep.ndim == 2:
            timestep_flat = timestep.flatten(0, 1)
        else:
            timestep_flat = timestep
        
        # Move timesteps and weights to same device
        device = timestep.device
        timesteps = self.timesteps.to(device)
        weights = self.linear_timesteps_weights.to(device)
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(1) - timestep_flat.unsqueeze(0)).abs(), dim=0
        )
        weights = weights[timestep_id]
        
        if timestep.ndim == 2:
            weights = weights.unflatten(0, timestep.shape)
        
        return weights
    
    def convert_flow_to_x0(self, flow_pred, xt, timestep):
        """
        Convert flow prediction to x0 prediction.
        
        Flow Matching: xt = (1 - sigma_t) * x0 + sigma_t * noise
        Flow = noise - x0
        Therefore: x0 = xt - sigma_t * flow
        
        Args:
            flow_pred: Flow prediction [B*F, C, H, W]
            xt: Noisy input [B*F, C, H, W]
            timestep: Timestep tensor [B*F]
            
        Returns:
            x0_pred: Predicted clean sample [B*F, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        
        # Move timesteps and sigmas to same device
        device = timestep.device
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id]
        
        if len(xt.shape) == 4:
            sigma_t = sigma_t.view(-1, 1, 1, 1)
        elif len(xt.shape) == 5:
            sigma_t = sigma_t.view(-1, 1, 1, 1, 1)
        
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred
    
    def convert_flow_to_noise(self, flow_pred, xt, timestep):
        """
        Convert flow prediction to noise prediction.
        
        Flow = noise - x0
        x0 = xt - sigma_t * flow
        Therefore: noise = flow + x0 = flow + (xt - sigma_t * flow) = xt - (sigma_t - 1) * flow
        
        Args:
            flow_pred: Flow prediction [B*F, C, H, W]
            xt: Noisy input [B*F, C, H, W]
            timestep: Timestep tensor [B*F]
            
        Returns:
            noise_pred: Predicted noise [B*F, C, H, W]
        """
        x0_pred = self.convert_flow_to_x0(flow_pred, xt, timestep)
        noise_pred = flow_pred + x0_pred
        return noise_pred


class SimpleTextEncoder(nn.Module):
    """Simple text encoder (same as train.py for compatibility)."""

    def __init__(self, device="cuda", text_dim=128, text_len=77, vocab_size=256):
        super().__init__()
        self.device = device
        self.text_dim = text_dim
        self.text_len = text_len
        self.vocab_size = vocab_size

        self.char_embedding = nn.Embedding(vocab_size, text_dim)
        self.proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

        self.to(device)

    def _tokenize(self, text: str) -> list:
        """Character-level tokenization."""
        tokens = []
        for char in text:
            byte_val = ord(char)
            token = min(byte_val, self.vocab_size - 1)
            tokens.append(token)
        return tokens

    def forward(self, text_prompts):
        """Encode text prompts."""
        batch_size = len(text_prompts)
        prompt_embeds_list = []

        for prompt in text_prompts:
            tokens = self._tokenize(prompt)

            if len(tokens) < self.text_len:
                tokens = tokens + [0] * (self.text_len - len(tokens))
            else:
                tokens = tokens[:self.text_len]

            token_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)
            embed_seq = self.char_embedding(token_tensor)
            embed_seq = self.proj(embed_seq)

            prompt_embeds_list.append(embed_seq)

        prompt_embeds = torch.stack(prompt_embeds_list, dim=0)

        return {
            "prompt_embeds": prompt_embeds
        }


@hydra.main(version_base=None, config_path="configs", config_name="train0")
def main(cfg: DictConfig):
    """Main pretraining script."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    # Print config
    print("=" * 70)
    print("Self-Forcing Pretraining (Supervised)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    print("=" * 70)

    # Device setup
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    print(f"Device: {device}")

    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Extract config values
    num_steps = cfg.training.num_steps
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    weight_decay = cfg.training.weight_decay
    num_samples = cfg.training.num_samples
    save_interval = cfg.training.save_interval
    log_interval = cfg.training.log_interval
    viz_interval = cfg.training.viz_interval
    log_dir = output_dir
    video_height = cfg.training.video_height
    video_width = cfg.training.video_width
    video_frames = cfg.training.video_frames

    # Create dataset
    dataset_type = cfg.dataset.type.lower()
    print(f"\n1. Creating {dataset_type} dataset...")

    if dataset_type != 'moving_mnist':
        raise ValueError(
            f"Dataset type '{dataset_type}' is not supported. "
            f"Only 'moving_mnist' is available. Please set dataset.type='moving_mnist' in your config."
        )
    
    dataset = MovingMNISTDataset(
        num_samples=num_samples,
        width=video_width,
        height=video_height,
        num_frames=video_frames,
        seed=cfg.seed,
        num_digits=cfg.dataset.num_digits,
        digit_size=cfg.dataset.digit_size,
        max_velocity=cfg.dataset.max_velocity
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
    generator = TinyCausalWanModel(
        in_dim=cfg.model.in_dim,
        out_dim=cfg.model.out_dim,
        dim=cfg.model.dim,
        ffn_dim=cfg.model.ffn_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        patch_size=tuple(cfg.model.patch_size),
        text_dim=cfg.model.text_dim,
        freq_dim=cfg.model.freq_dim,
        num_frame_per_block=cfg.model.num_frame_per_block,
    )
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")

    # Create text encoder
    text_encoder = SimpleTextEncoder(
        device=device,
        text_dim=cfg.text_encoder.text_dim,
        text_len=cfg.text_encoder.text_len,
        vocab_size=cfg.text_encoder.vocab_size
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(generator.parameters()) + list(text_encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Create scheduler with Flow Matching support
    scheduler = SimpleScheduler(
        num_train_timesteps=cfg.training.get('num_train_timestep', 1000),
        sigma_max=cfg.training.get('sigma_max', 1.0),
        sigma_min=cfg.training.get('sigma_min', 0.003/1.002)
    )

    # Create trainer
    print("\n3. Creating trainer...")
    trainer = PretrainingTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        output_dir=output_dir,
        text_encoder=text_encoder
    )

    # Training plotter
    plotter = TrainingPlotter(save_dir=str(Path(log_dir) / "plots"))

    # Training loop
    print("\n4. Starting pretraining...")
    print("-" * 70)

    dataloader_iter = iter(dataloader)
    pbar = tqdm(range(num_steps), desc="Pretraining")

    while trainer.step < num_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Convert labels from MovingMNISTDataset to text prompts (e.g., 0 -> "digit 0")
        labels = batch["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        elif not isinstance(labels, list):
            labels = [labels]
        batch["prompts"] = [f"digit {label}" if isinstance(label, int) else f"digits {label[0]},{label[1]}" for label in labels]

        # Encode prompts
        with torch.no_grad():
            conditional_dict = text_encoder(batch["prompts"])

        # Training step
        metrics = trainer.train_step(batch, conditional_dict)

        # Log to plotter
        plotter.log_metric("loss", metrics["loss"], trainer.step)

        # Log metrics
        if trainer.step % log_interval == 0:
            trainer._log_metrics(metrics)

        # Update progress bar
        pbar.set_postfix({"loss": f"{metrics['loss']:.8f}", "step": trainer.step})
        pbar.update(1)

        # Save checkpoint
        if trainer.step % save_interval == 0:
            trainer._save_checkpoint()

        # Generate samples
        if trainer.step % viz_interval == 0 and trainer.step > 0:
            print(f"\nGenerating sample videos at step {trainer.step}...")
            viz_prompts = list(cfg.viz_prompts) if cfg.viz_prompts else [
                "A red circle moving horizontally",
                "A blue square moving vertically"
            ]
            trainer.generate_sample_videos(
                text_encoder=text_encoder,
                num_samples=4,
                num_frames=video_frames,
                prompts=viz_prompts,
                gif_fps=cfg.generation.gif_fps
            )

        if trainer.step >= num_steps:
            break

    pbar.close()

    # Finalize
    print("\n5. Finalizing...")
    trainer._save_checkpoint(final=True)
    trainer._save_metrics()

    # Plot training curves
    plotter.plot_metric("loss", title="Pretraining Loss")
    plotter.save_history(str(Path(log_dir) / "metrics_history.json"))

    # Generate final samples
    print("\nGenerating final sample videos...")
    trainer.generate_sample_videos(
        text_encoder=text_encoder,
        num_samples=4,
        num_frames=video_frames,
        gif_fps=cfg.generation.gif_fps
    )

    # Finish wandb
    if trainer.use_wandb:
        wandb.finish()

    print("\n" + "=" * 70)
    print("Pretraining completed!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {log_dir}")
    print(f"Final checkpoint: {log_dir}/checkpoint_final.pt")
    print("\nTo use this checkpoint for Self-Forcing training:")
    print(f"  python train.py checkpoint={log_dir}/checkpoint_final.pt")
    print("\nKey points:")
    print("1. Pretraining uses supervised learning with ground truth videos")
    print("2. Model learns to denoise noisy videos")
    print("3. This provides a good initialization for Self-Forcing")
    print("4. Self-Forcing (train.py) can then fine-tune without ground truth")


if __name__ == "__main__":
    main()
