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
        output_dir: Path
    ):
        """
        Args:
            generator: The video generation model
            optimizer: Optimizer for the generator
            scheduler: Noise scheduler
            cfg: Hydra config object
            output_dir: Output directory for logs and checkpoints
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
        self.cfg = cfg

        # Training config
        self.num_frames_per_block = cfg.training.num_frames_per_block
        self.num_train_timestep = cfg.training.num_train_timestep
        self.min_timestep = cfg.training.min_timestep
        self.max_timestep = cfg.training.max_timestep
        # Use min_step/max_step if not explicitly set (matching official impl)
        if self.min_timestep is None:
            self.min_timestep = int(0.02 * self.num_train_timestep)  # Default: 20
        if self.max_timestep is None:
            self.max_timestep = int(0.98 * self.num_train_timestep)  # Default: 980
        self.gradient_clip_norm = cfg.training.gradient_clip_norm
        self.prediction_type = cfg.training.prediction_type

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoints directory
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

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
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name or f"train0-pretrain-{self.log_dir.name}",
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
            # Log model parameter count as a metric for better visibility
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params
            }, step=0)
            print(f"Initialized wandb: project={cfg.wandb.project}")
        elif cfg.wandb.enabled and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
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
        # Use empty conditional dict (model will create dummy embeddings if needed)
        conditional_dict = {}
        flow_pred, _ = self.generator(noisy_video, timesteps_expanded, conditional_dict)

        # Compute training target (Flow Matching: noise - sample)
        training_target = self.scheduler.training_target(
            video.flatten(0, 1),
            noise.flatten(0, 1),
            timesteps_expanded.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))

        # Compute loss with timestep weighting (matching official impl)
        if self.prediction_type == "vf":
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
            # Add learning rate to metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics_with_lr = {**metrics, 'learning_rate': current_lr}
            wandb.log(metrics_with_lr, step=self.step)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint compatible with train.py."""
        checkpoint = {
            "step": self.step,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "training_type": "pretraining"
        }

        suffix = "final" if final else f"step_{self.step:06d}"
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{suffix}.pt"
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.metrics_history = checkpoint.get("metrics_history", {"loss": [], "step": []})
        print(f"Loaded checkpoint from {checkpoint_path} (step {self.step})")

    def generate_sample_videos(
        self,
        num_samples: int = 4,
        num_frames: int = 9,
        digits: Optional[List[int]] = None,
        ground_truth_videos: Optional[torch.Tensor] = None,
        gif_fps: int = 2
    ):
        """Generate sample videos for visualization.
        
        Args:
            num_samples: Number of videos to generate
            num_frames: Number of frames per video
            digits: List of digit labels (0-9) to visualize. If None, uses empty labels.
            ground_truth_videos: Optional ground truth videos for comparison
            gif_fps: FPS for GIF output
        """
        self.generator.eval()

        # Prepare labels for visualization
        if digits is None:
            digit_labels = None
        else:
            digit_labels = digits[:num_samples]
            # Pad with None if needed
            while len(digit_labels) < num_samples:
                digit_labels.append(None)

        with torch.no_grad():
            # Use empty conditional dict (model will create dummy embeddings if needed)
            conditional_dict = {}

            batch_size = num_samples
            height, width = 32, 32

            # Start from pure noise (normalized to match data range)
            # Flow Matching starts from noise distribution, typically N(0,1)
            noise = torch.randn(
                batch_size, num_frames, 3, height, width,
                device=self.device
            )

            # Simple iterative denoising with Flow Matching
            # Use Euler steps: x_next = x_current + flow * (sigma_next - sigma_current)
            denoising_steps = [1000, 750, 500, 250, 0]
            x = noise

            for i, t in enumerate(denoising_steps[:-1]):
                timestep = torch.full(
                    (batch_size, num_frames),
                    t,
                    device=self.device,
                    dtype=torch.long
                )

                # Predict flow at current timestep
                flow_pred, _ = self.generator(x, timestep, conditional_dict)
                
                # Use Flow Matching Euler step: x_next = x_current + flow * (sigma_next - sigma_current)
                # This correctly follows the ODE dx/dt = flow
                if i < len(denoising_steps) - 2:
                    # Not the last step: step to next timestep
                    next_t = denoising_steps[i + 1]
                    next_timestep = torch.full(
                        (batch_size, num_frames),
                        next_t,
                        device=self.device,
                        dtype=torch.long
                    )
                    # Use scheduler.step for proper Flow Matching sampling
                    x = self.scheduler.step(
                        flow_pred.flatten(0, 1),
                        timestep.flatten(0, 1),
                        x.flatten(0, 1)
                    ).unflatten(0, (batch_size, num_frames))
                else:
                    # Last step: step to final (sigma=0)
                    x = self.scheduler.step(
                        flow_pred.flatten(0, 1),
                        timestep.flatten(0, 1),
                        x.flatten(0, 1),
                        to_final=True
                    ).unflatten(0, (batch_size, num_frames))

            generated_videos = x

            # Normalize to [0, 1]
            generated_videos = (generated_videos + 1.0) / 2.0
            generated_videos = generated_videos.clamp(0, 1)

            # Process ground truth videos if provided
            gt_videos_list = None
            if ground_truth_videos is not None:
                if ground_truth_videos.device != self.device:
                    ground_truth_videos = ground_truth_videos.to(self.device)

                # Normalize ground truth videos from [-1, 1] to [0, 1]
                if ground_truth_videos.min() < 0:
                    ground_truth_videos = (ground_truth_videos + 1.0) / 2.0
                ground_truth_videos = ground_truth_videos.clamp(0, 1)

                # Resize if needed
                # ground_truth_videos shape: [num_samples, F, C, H, W]
                # generated_videos shape: [num_samples, F, C, H, W]
                if ground_truth_videos.shape[2:] != generated_videos.shape[2:]:
                    gt_resized = []
                    for gt_vid in ground_truth_videos:
                        # gt_vid shape: [F, C, H, W]
                        # Squeeze out any extra batch dimensions
                        while gt_vid.dim() > 4:
                            gt_vid = gt_vid.squeeze(0)
                        
                        frames_resized = []
                        # Iterate over frames: frame shape is [C, H, W]
                        for frame_idx in range(gt_vid.shape[0]):
                            frame = gt_vid[frame_idx]  # [C, H, W]
                            
                            # Ensure frame is 3D [C, H, W]
                            if frame.dim() != 3:
                                # If somehow still has extra dims, squeeze them
                                while frame.dim() > 3:
                                    frame = frame.squeeze(0)
                            
                            frame_np = frame.permute(1, 2, 0).cpu().numpy()
                            from PIL import Image
                            img = Image.fromarray((frame_np * 255).astype(np.uint8))
                            target_size = (generated_videos.shape[-1], generated_videos.shape[-2])
                            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                            frame_resized = torch.from_numpy(np.array(img_resized)).float() / 255.0
                            frame_resized = frame_resized.permute(2, 0, 1)
                            frames_resized.append(frame_resized)
                        gt_resized.append(torch.stack(frames_resized))
                    ground_truth_videos = torch.stack(gt_resized).to(self.device)

                # Truncate to match num_frames
                if ground_truth_videos.shape[1] != num_frames:
                    ground_truth_videos = ground_truth_videos[:, :num_frames]

                # Limit to num_samples
                ground_truth_videos = ground_truth_videos[:num_samples]
                gt_videos_list = [gt_video for gt_video in ground_truth_videos]

            # Save GIFs for generated videos
            videos_list = []
            for i, video_tensor in enumerate(generated_videos):
                gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                create_video_gif(video_tensor, str(gif_path), fps=gif_fps)
                videos_list.append(video_tensor)

            # Save GIFs for ground truth videos
            gt_gif_paths = []
            if gt_videos_list is not None:
                for i, gt_video in enumerate(gt_videos_list):
                    gt_gif_path = self.samples_dir / f"step_{self.step:06d}_gt_{i:02d}.gif"
                    create_video_gif(gt_video, str(gt_gif_path), fps=gif_fps)
                    gt_gif_paths.append(gt_gif_path)

            # Save grid for generated videos
            grid_path = self.samples_dir / f"step_{self.step:06d}_grid.png"
            # Format labels for visualization: "Digit 0", "Digit 1", etc.
            if digit_labels:
                labels_for_viz = [f"Digit {d}" if d is not None else "" for d in digit_labels]
            else:
                labels_for_viz = [""] * num_samples
            save_video_grid(videos_list, str(grid_path), prompts=labels_for_viz)

            # Save comparison grid (generated vs ground truth)
            comparison_grid_path = None
            if gt_videos_list is not None:
                comparison_grid_path = self.samples_dir / f"step_{self.step:06d}_comparison_grid.png"
                comparison_videos = []
                comparison_prompts = []
                for gen_vid, gt_vid, label in zip(videos_list, gt_videos_list, labels_for_viz):
                    comparison_videos.append(gen_vid)
                    comparison_videos.append(gt_vid)
                    comparison_prompts.append(f"{label} (Generated)")
                    comparison_prompts.append(f"{label} (Ground Truth)")
                save_video_grid(comparison_videos, str(comparison_grid_path), prompts=comparison_prompts, ncols=2)

            print(f"  Saved sample videos to {self.samples_dir}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "samples/grid": wandb.Image(str(grid_path)),
                }, step=self.step)

                if comparison_grid_path is not None:
                    wandb.log({
                        "samples/comparison_grid": wandb.Image(str(comparison_grid_path)),
                    }, step=self.step)

                for i, video_tensor in enumerate(videos_list):
                    gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                    caption = f"Digit {digit_labels[i]}" if digit_labels and digit_labels[i] is not None else ""
                    wandb.log({
                        f"samples/generated_video_{i}": wandb.Video(
                            str(gif_path),
                            format="gif",
                            caption=caption
                        ),
                    }, step=self.step)

                # Log ground truth videos
                if gt_gif_paths:
                    for i, gt_gif_path in enumerate(gt_gif_paths):
                        caption = f"Digit {digit_labels[i]} (GT)" if digit_labels and digit_labels[i] is not None else "Ground Truth"
                        wandb.log({
                            f"samples/ground_truth_video_{i}": wandb.Video(
                                str(gt_gif_path),
                                format="gif",
                                caption=caption
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
    
    def step(self, flow_pred, timestep, sample, to_final=False):
        """
        Perform one Flow Matching sampling step (Euler step).
        
        Flow Matching ODE: dx/dt = flow
        Euler step: x_next = x_current + flow * (sigma_next - sigma_current)
        
        Args:
            flow_pred: Flow prediction [B*F, C, H, W] or [B, F, C, H, W]
            timestep: Current timestep [B*F] or [B, F] (values in [0, 1000])
            sample: Current sample [B*F, C, H, W] or [B, F, C, H, W]
            to_final: If True, step to final (sigma=0 for forward, sigma=1 for reverse)
            
        Returns:
            prev_sample: Updated sample [B*F, C, H, W] or [B, F, C, H, W]
        """
        original_shape = sample.shape
        if timestep.ndim == 2:
            timestep_flat = timestep.flatten(0, 1)
            sample_flat = sample.flatten(0, 1)
            flow_pred_flat = flow_pred.flatten(0, 1)
        else:
            timestep_flat = timestep
            sample_flat = sample
            flow_pred_flat = flow_pred
        
        # Move timesteps and sigmas to same device
        device = timestep_flat.device
        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        
        # Get current sigma
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id]
        
        # Get next sigma
        if to_final or (timestep_id + 1 >= len(timesteps)).any():
            # Step to final (sigma=0 for denoising)
            sigma_next = torch.zeros_like(sigma)
        else:
            sigma_next = sigmas[timestep_id + 1]
        
        # Reshape for broadcasting
        if len(sample_flat.shape) == 4:
            sigma = sigma.view(-1, 1, 1, 1)
            sigma_next = sigma_next.view(-1, 1, 1, 1)
        elif len(sample_flat.shape) == 5:
            sigma = sigma.view(-1, 1, 1, 1, 1)
            sigma_next = sigma_next.view(-1, 1, 1, 1, 1)
        
        # Euler step: x_next = x_current + flow * (sigma_next - sigma_current)
        prev_sample = sample_flat + flow_pred_flat * (sigma_next - sigma)
        
        # Reshape back if needed
        if timestep.ndim == 2:
            prev_sample = prev_sample.unflatten(0, original_shape[:2])
        
        return prev_sample


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

    # Create optimizer
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Create scheduler with Flow Matching support
    scheduler = SimpleScheduler(
        num_train_timesteps=cfg.training.num_train_timestep,
        sigma_max=cfg.training.sigma_max,
        sigma_min=cfg.training.sigma_min
    )

    # Create trainer
    print("\n3. Creating trainer...")
    trainer = PretrainingTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        output_dir=output_dir
    )

    # Training plotter
    plotter = TrainingPlotter(save_dir=str(Path(log_dir) / "plots"))

    # Get ground truth videos for visualization
    num_viz_samples = cfg.generation.num_viz_samples
    ground_truth_videos_for_viz = None
    if len(dataset) > 0:
        gt_videos_list = []
        for i in range(min(num_viz_samples, len(dataset))):
            sample = dataset[i]
            video = sample["video"]  # Shape: [F, C, H, W]
            gt_videos_list.append(video)
        # Stack to [num_viz_samples, F, C, H, W] (no extra batch dimension)
        if gt_videos_list:
            ground_truth_videos_for_viz = torch.stack(gt_videos_list, dim=0)

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

        # Training step (no text conditioning needed)
        metrics = trainer.train_step(batch)

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
            # Get digits to visualize (default to [0, 1, 2, 3] if not specified)
            num_viz_samples = cfg.generation.num_viz_samples
            viz_digits = list(cfg.viz_digits) if cfg.viz_digits else list(range(min(10, num_viz_samples)))
            trainer.generate_sample_videos(
                num_samples=num_viz_samples,
                num_frames=video_frames,
                digits=viz_digits,
                ground_truth_videos=ground_truth_videos_for_viz,
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
        num_samples=4,
        num_frames=video_frames,
        ground_truth_videos=ground_truth_videos_for_viz[:4] if ground_truth_videos_for_viz is not None else None,
        gif_fps=cfg.generation.gif_fps
    )

    # Finish wandb
    if trainer.use_wandb:
        wandb.finish()

    print("\n" + "=" * 70)
    print("Pretraining completed!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {log_dir}/checkpoints")
    print(f"Final checkpoint: {log_dir}/checkpoints/checkpoint_final.pt")
    print("\nTo use this checkpoint for Self-Forcing training:")
    print(f"  python train.py checkpoint={log_dir}/checkpoints/checkpoint_final.pt")
    print("\nKey points:")
    print("1. Pretraining uses supervised learning with ground truth videos")
    print("2. Model learns to denoise noisy videos")
    print("3. This provides a good initialization for Self-Forcing")
    print("4. Self-Forcing (train.py) can then fine-tune without ground truth")


if __name__ == "__main__":
    main()
