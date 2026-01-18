"""
Self-Forcing Training Script

This module provides the Self-Forcing training loop for video generation.
Self-Forcing is a data-free training method that simulates inference during training.

Usage:
    python train.py                                    # Use default config
    python train.py training.num_steps=2000            # Override config values
    python train.py checkpoint=path/to/checkpoint.pt   # Load pretrained
"""

import torch
import torch.nn as nn
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
from dynamic_video_sf import SelfForcingEngine

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SimplifiedTrainer:
    """
    Full Self-Forcing trainer for single GPU training.
    
    Implements the complete Self-Forcing algorithm:
    - Block-by-block autoregressive generation with KV caching simulation
    - DMD loss for data-free training
    - Gradient control (only last 21 frames compute gradients)
    - Variable frame generation support
    - Optional critic training (simplified)
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
            output_dir: Hydra output directory for logs, checkpoints, wandb
        """
        self.device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        self.output_dir = output_dir
        self.save_interval = cfg.training.save_interval
        self.log_interval = cfg.training.log_interval
        self.viz_interval = cfg.training.viz_interval

        # Wandb settings
        self.use_wandb = cfg.wandb.enabled and WANDB_AVAILABLE

        self.generator = generator.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        # Load training config values
        self.num_frames_per_block = cfg.training.num_frames_per_block
        self.denoising_steps = list(cfg.training.denoising_steps)
        self.context_noise = cfg.training.context_noise
        self.training_num_frames = cfg.training.num_frames
        self.video_height = cfg.training.video_height
        self.video_width = cfg.training.video_width
        self.gradient_clip_norm = cfg.training.gradient_clip_norm
        
        # DMD-specific configs
        self.num_train_timestep = cfg.training.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = cfg.training.guidance_scale
        self.timestep_shift = cfg.training.timestep_shift
        self.ts_schedule = cfg.training.ts_schedule
        self.ts_schedule_max = cfg.training.ts_schedule_max
        self.min_score_timestep = cfg.training.min_score_timestep
        
        # Variable frame generation (matching official impl)
        self.min_num_frames = cfg.training.min_num_frames
        self.max_num_frames = cfg.training.max_num_frames if cfg.training.max_num_frames is not None else self.training_num_frames
        assert self.min_num_frames % self.num_frames_per_block == 0
        assert self.max_num_frames % self.num_frames_per_block == 0

        # Create subdirectories in Hydra output dir
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.metrics_history = {
            "loss": [],
            "generator_loss": [],
            "dmd_gradient_norm": [],
            "step": []
        }
        
        # Moving average loss tracker (exponential moving average)
        self.average_loss = None
        self.smoothing_decay = 0.99  # EMA decay factor

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name or f"dmd2-{self.output_dir.name}",
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(self.output_dir)
            )
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "training_type": "dmd2"
            })
            # Log model parameter count as a metric for better visibility
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params
            }, step=0)
            print(f"Initialized wandb: project={cfg.wandb.project}, dir={self.output_dir}")
        elif cfg.wandb.enabled and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform one training step with Self-Forcing.
        Matches official implementation:
        1. Sample variable number of frames (21 to max_num_frames)
        2. Generate video block-by-block autoregressively
        3. Compute DMD loss on generated video
        4. Only last 21 frames compute gradients
        """
        self.generator.train()
        self.optimizer.zero_grad()

        batch_size = batch["video"].shape[0] if "video" in batch else 1
        
        # Step 1: Sample variable number of frames (matching official impl)
        min_num_blocks = self.min_num_frames // self.num_frames_per_block
        max_num_blocks = self.max_num_frames // self.num_frames_per_block
        num_generated_blocks = torch.randint(
            min_num_blocks, max_num_blocks + 1, (1,), device=self.device
        ).item()
        num_generated_frames = num_generated_blocks * self.num_frames_per_block

        # Step 2: Sample noise for generation
        noise = torch.randn(
            batch_size, num_generated_frames, 3, self.video_height, self.video_width,
            device=self.device
        )

        # Step 3: Simulate Self-Forcing inference during training
        # This generates video block-by-block with KV cache simulation
        # Use empty conditional dict (model will create dummy embeddings if needed)
        conditional_dict = {}
        generated_video, denoised_timestep_from, denoised_timestep_to = self.sf_engine.simulate_self_forcing(
            noise, conditional_dict, return_timesteps=True
        )

        # Step 4: Compute Self-Forcing loss (data-free, uses DMD)
        # Use empty unconditional dict (model will create dummy embeddings if needed)
        unconditional_dict = {}
        
        loss, log_dict = self.sf_engine.compute_self_forcing_loss(
            generated_video,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        # Step 5: Backward pass
        loss.backward()

        # Step 6: Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), 
            max_norm=self.gradient_clip_norm
        )

        # Step 7: Optimizer step
        self.optimizer.step()

        # Step 8: Update metrics
        loss_value = loss.item()
        
        # Update moving average loss (exponential moving average)
        if self.average_loss is None:
            self.average_loss = loss_value
        else:
            self.average_loss = self.smoothing_decay * self.average_loss + (1 - self.smoothing_decay) * loss_value
        
        metrics = {
            "loss": loss_value,
            "generator_loss": loss_value,
            "average_loss": self.average_loss,
            "grad_norm": grad_norm.item(),
            "num_frames": num_generated_frames,
            "step": self.step
        }
        
        # Add DMD-specific metrics
        if log_dict:
            metrics.update({k: v.item() if torch.is_tensor(v) else v 
                           for k, v in log_dict.items()})

        self.metrics_history["loss"].append(loss_value)
        self.metrics_history["generator_loss"].append(loss_value)
        if "dmd_gradient_norm" in log_dict:
            self.metrics_history["dmd_gradient_norm"].append(
                log_dict["dmd_gradient_norm"].item() if torch.is_tensor(log_dict["dmd_gradient_norm"]) 
                else log_dict["dmd_gradient_norm"]
            )
        self.metrics_history["step"].append(self.step)

        self.step += 1

        return metrics
    

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        loss_str = f"Loss = {metrics['loss']:.8f}"
        if "average_loss" in metrics:
            loss_str += f", AvgLoss = {metrics['average_loss']:.8f}"
        if "grad_norm" in metrics:
            loss_str += f", GradNorm = {metrics['grad_norm']:.4f}"
        if "dmd_gradient_norm" in metrics:
            loss_str += f", DMDGradNorm = {metrics['dmd_gradient_norm']:.4f}"
        if "pred_diff" in metrics:
            loss_str += f", PredDiff = {metrics['pred_diff']:.6f}"
        if "pred_fake_norm" in metrics:
            loss_str += f", PredFakeNorm = {metrics['pred_fake_norm']:.4f}"
        if "pred_real_norm" in metrics:
            loss_str += f", PredRealNorm = {metrics['pred_real_norm']:.4f}"
        if "original_latent_norm" in metrics:
            loss_str += f", OrigLatentNorm = {metrics['original_latent_norm']:.4f}"
        if "num_frames" in metrics:
            loss_str += f", Frames = {metrics['num_frames']}"
        print(f"Step {self.step}: {loss_str}")

        if self.use_wandb:
            # Add learning rate to metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics_with_lr = {**metrics, 'learning_rate': current_lr}
            wandb.log(metrics_with_lr, step=self.step)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "training_type": "dmd2",
            "config": OmegaConf.to_container(self.cfg, resolve=True)
        }

        suffix = "final" if final else f"step_{self.step:06d}"
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)

        if final or self.step % self.save_interval == 0:
            print(f"Saved checkpoint to {checkpoint_path}")

    def _save_metrics(self):
        """Save training metrics to file."""
        import json

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        self.step = checkpoint.get("step", 0)
        self.metrics_history = checkpoint.get("metrics_history", {
            "loss": [], "generator_loss": [], "dmd_gradient_norm": [], "step": []
        })
        
        training_type = checkpoint.get("training_type", "unknown")
        print(f"Loaded checkpoint from {checkpoint_path} (step {self.step}, type: {training_type})")

    def generate_sample_videos(
        self,
        num_samples: int = 4,
        num_frames: int = 9,
        num_frames_per_block: int = 3,
        digits: Optional[List[int]] = None,
        ground_truth_videos: Optional[torch.Tensor] = None,
        gif_fps: int = 2
    ):
        """Generate sample videos for visualization during training.
        
        Args:
            num_samples: Number of videos to generate
            num_frames: Number of frames per video
            num_frames_per_block: Frames per block for autoregressive generation
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
            noise = torch.randn(
                batch_size, num_frames, 3, 64, 64,
                device=self.device
            )

            generated_videos = self._generate_full_video(noise, conditional_dict, num_frames_per_block)

            # Normalize to [0, 1]
            generated_videos = (generated_videos + 1.0) / 2.0
            generated_videos = generated_videos.clamp(0, 1)

            # Process ground truth videos if provided
            gt_videos_list = None
            if ground_truth_videos is not None:
                if ground_truth_videos.device != self.device:
                    ground_truth_videos = ground_truth_videos.to(self.device)

                if ground_truth_videos.min() < 0:
                    ground_truth_videos = (ground_truth_videos + 1.0) / 2.0
                ground_truth_videos = ground_truth_videos.clamp(0, 1)

                if ground_truth_videos.shape[2:] != generated_videos.shape[2:]:
                    gt_resized = []
                    for gt_vid in ground_truth_videos:
                        # gt_vid shape: [F, C, H, W]
                        # Squeeze out any extra batch dimensions
                        while gt_vid.dim() > 4:
                            gt_vid = gt_vid.squeeze(0)
                        
                        frames_resized = []
                        # Iterate over frames: frame shape should be [C, H, W]
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

                if ground_truth_videos.shape[1] != num_frames:
                    ground_truth_videos = ground_truth_videos[:, :num_frames]

                ground_truth_videos = ground_truth_videos[:num_samples]
                gt_videos_list = [gt_video for gt_video in ground_truth_videos]

            # Save GIFs
            videos_list = []
            for i, video_tensor in enumerate(generated_videos):
                gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                create_video_gif(video_tensor, str(gif_path), fps=gif_fps)
                videos_list.append(video_tensor)

            gt_gif_paths = []
            if gt_videos_list is not None:
                for i, gt_video in enumerate(gt_videos_list):
                    gt_gif_path = self.samples_dir / f"step_{self.step:06d}_gt_{i:02d}.gif"
                    create_video_gif(gt_video, str(gt_gif_path), fps=gif_fps)
                    gt_gif_paths.append(gt_gif_path)

            # Save grid
            grid_path = self.samples_dir / f"step_{self.step:06d}_grid.png"
            # Format labels for visualization: "Digit 0", "Digit 1", etc.
            if digit_labels:
                labels_for_viz = [f"Digit {d}" if d is not None else "" for d in digit_labels]
            else:
                labels_for_viz = [""] * num_samples
            save_video_grid(videos_list, str(grid_path), prompts=labels_for_viz)

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
                samples_payload = {
                    "vis_sample": wandb.Image(str(grid_path)),
                }

                if comparison_grid_path is not None:
                    samples_payload["vis_comparison_gid"] = wandb.Image(str(comparison_grid_path))

                samples_table = wandb.Table(columns=[
                    "index",
                    "digit",
                    "generated_video",
                    "ground_truth_video",
                ])
                for i, _video_tensor in enumerate(videos_list):
                    gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                    caption = f"Digit {digit_labels[i]}" if digit_labels and digit_labels[i] is not None else ""
                    generated_video = wandb.Video(
                        str(gif_path),
                        format="gif",
                        caption=caption,
                    )
                    gt_video = None
                    if gt_gif_paths:
                        gt_gif_path = gt_gif_paths[i]
                        gt_caption = (
                            f"Digit {digit_labels[i]} (GT)"
                            if digit_labels and digit_labels[i] is not None
                            else "Ground Truth"
                        )
                        gt_video = wandb.Video(
                            str(gt_gif_path),
                            format="gif",
                            caption=gt_caption,
                        )
                    samples_table.add_data(i, caption, generated_video, gt_video)
                samples_payload["vis_gt"] = samples_table

                wandb.log(samples_payload, step=self.step)

        self.generator.train()

    def _generate_full_video(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        num_frames_per_block: int = 3
    ) -> torch.Tensor:
        """Generate full video for visualization."""
        batch_size, num_frames, num_channels, height, width = noise.shape

        assert num_frames % num_frames_per_block == 0
        num_blocks = num_frames // num_frames_per_block

        output = torch.zeros_like(noise)
        current_start_frame = 0

        for block_idx in range(num_blocks):
            block_noise = noise[:, current_start_frame:current_start_frame + num_frames_per_block]
            noisy_input = block_noise

            for step_idx, timestep in enumerate(self.denoising_steps):
                timestep_tensor = torch.full(
                    (batch_size, num_frames_per_block),
                    timestep,
                    device=self.device,
                    dtype=torch.long
                )

                denoised, _ = self.generator(noisy_input, timestep_tensor, conditional_dict)

                if step_idx < len(self.denoising_steps) - 1:
                    next_timestep = self.denoising_steps[step_idx + 1]
                    noise_to_add = torch.randn_like(denoised)
                    alpha = 1.0 - (next_timestep / 1000.0)
                    noisy_input = alpha * denoised + (1 - alpha) * noise_to_add
                else:
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


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """Main training script."""
    # Get Hydra output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Print config
    print("=" * 70)
    print("Self-Forcing Training")
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
    log_interval = cfg.training.log_interval
    viz_interval = cfg.training.viz_interval
    video_height = cfg.training.video_height
    video_width = cfg.training.video_width
    video_frames = cfg.training.video_frames
    num_frame_per_block = cfg.model.num_frame_per_block

    # Create dataset
    dataset_type = cfg.dataset.type.lower()
    print(f"\n1. Creating {dataset_type} dataset...")

    
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
        num_frame_per_block=num_frame_per_block,
    )
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"   Using TinyCausalWanModel (transformer backbone)")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Create scheduler
    scheduler = SimpleScheduler()

    # Create trainer with Hydra output directory
    print("\n3. Creating trainer...")
    trainer = SimplifiedTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        output_dir=output_dir
    )

    checkpoint_path = cfg.training.pretrained_checkpoint 
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Please ensure the checkpoint file exists."
        )
    print(f"\n   Loading pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
        print(f"   Loaded generator weights from checkpoint")
    else:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain 'generator_state_dict'. "
            "Please ensure this is a valid model checkpoint."
        )

    training_type = checkpoint.get("training_type", "unknown")
    pretrain_step = checkpoint.get("step", 0)
    print(f"   Checkpoint type: {training_type}, trained for {pretrain_step} steps")

    # Create Self-Forcing engine after loading weights so real_score matches generator
    trainer.sf_engine = SelfForcingEngine(
        generator=generator,
        scheduler=scheduler,
        device=device,
        denoising_steps=list(cfg.training.denoising_steps),
        num_frames_per_block=cfg.training.num_frames_per_block,
        context_noise=cfg.training.context_noise
    )
    
    # Pass DMD configs to engine
    trainer.sf_engine.num_train_timestep = trainer.num_train_timestep
    trainer.sf_engine.min_step = trainer.min_step
    trainer.sf_engine.max_step = trainer.max_step
    trainer.sf_engine.timestep_shift = trainer.timestep_shift
    trainer.sf_engine.ts_schedule = trainer.ts_schedule
    trainer.sf_engine.ts_schedule_max = trainer.ts_schedule_max
    trainer.sf_engine.min_score_timestep = trainer.min_score_timestep
    trainer.sf_engine.guidance_scale = trainer.guidance_scale
    

    # Training plotter
    plotter = TrainingPlotter(save_dir=str(trainer.plots_dir))

    # Get ground truth videos for visualization (collect upfront for consistency)
    num_viz_samples = cfg.generation.num_viz_samples
    ground_truth_videos_for_viz = None
    if len(dataset) > 0:
        gt_videos_list = []
        for i in range(min(num_viz_samples, len(dataset))):
            sample = dataset[i]
            video = sample["video"]  # Shape: [F, C, H, W]
            # Add batch dimension: [1, F, C, H, W]
            gt_videos_list.append(video.unsqueeze(0))
        # Stack to [num_viz_samples, F, C, H, W]
        if gt_videos_list:
            ground_truth_videos_for_viz = torch.stack(gt_videos_list, dim=0)

    # Training loop
    print("\n4. Starting training...")
    print("-" * 70)

    dataloader_iter = iter(dataloader)
    pbar = tqdm(range(num_steps), desc="Training")

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
        if trainer.step % trainer.save_interval == 0:
            trainer._save_checkpoint()

        # Generate sample videos
        if trainer.step % viz_interval == 0 and trainer.step > 0:
            print(f"\nGenerating sample videos at step {trainer.step}...")
            # Get digits to visualize (default to [0, 1, 2, 3] if not specified)
            num_viz_samples = cfg.generation.num_viz_samples
            viz_digits = list(cfg.viz_digits) if cfg.viz_digits else list(range(min(4, num_viz_samples)))

            # Use pre-collected ground truth videos, or fall back to batch videos
            ground_truth_videos = None
            if ground_truth_videos_for_viz is not None:
                ground_truth_videos = ground_truth_videos_for_viz.to(trainer.device)
            elif "video" in batch:
                batch_videos = batch["video"]
                if isinstance(batch_videos, list):
                    batch_videos = [v.to(trainer.device) for v in batch_videos[:num_viz_samples]]
                    if batch_videos:
                        ground_truth_videos = torch.stack(batch_videos)
                elif isinstance(batch_videos, torch.Tensor):
                    ground_truth_videos = batch_videos[:num_viz_samples].to(trainer.device)
                    if len(ground_truth_videos.shape) == 4:
                        ground_truth_videos = ground_truth_videos.unsqueeze(0)

            trainer.generate_sample_videos(
                num_samples=num_viz_samples,
                num_frames=cfg.generation.viz_num_frames,
                num_frames_per_block=num_frame_per_block,
                digits=viz_digits,
                ground_truth_videos=ground_truth_videos,
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
    plotter.plot_metric("loss", title="Training Loss")
    plotter.save_history(str(output_dir / "metrics_history.json"))

    # Generate final sample videos
    print("\nGenerating final sample videos...")
    # Use pre-collected ground truth videos, or fall back to batch videos
    final_ground_truth_videos = None
    if ground_truth_videos_for_viz is not None:
        final_ground_truth_videos = ground_truth_videos_for_viz[:4].to(trainer.device)
    else:
        try:
            sample_batch = next(iter(dataloader))
            if "video" in sample_batch:
                batch_videos = sample_batch["video"]
                if isinstance(batch_videos, list):
                    batch_videos = [v.to(trainer.device) for v in batch_videos[:4]]
                    if batch_videos:
                        final_ground_truth_videos = torch.stack(batch_videos)
                elif isinstance(batch_videos, torch.Tensor):
                    final_ground_truth_videos = batch_videos[:4].to(trainer.device)
                    if len(final_ground_truth_videos.shape) == 4:
                        final_ground_truth_videos = final_ground_truth_videos.unsqueeze(0)
        except:
            pass

    trainer.generate_sample_videos(
        num_samples=4,
        num_frames=9,
        num_frames_per_block=3,
        ground_truth_videos=final_ground_truth_videos,
        gif_fps=cfg.generation.gif_fps
    )

    # Finish wandb
    if trainer.use_wandb:
        wandb.finish()

    print("\n" + "=" * 70)
    print("Training completed!")


if __name__ == "__main__":
    main()
