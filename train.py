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
        self.use_dmd_loss = cfg.training.use_dmd_loss

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
            "step": []
        }

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb.get('project', 'self-forcing'),
                entity=cfg.wandb.get('entity', None),
                name=cfg.wandb.get('name', None) or f"self-forcing-{self.output_dir.name}",
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(self.output_dir)
            )
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": self.device,
                "output_dir": str(self.output_dir)
            })
            print(f"Initialized wandb: project={cfg.wandb.project}, dir={self.output_dir}")
        elif cfg.wandb.enabled and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform one training step."""
        self.generator.train()
        self.optimizer.zero_grad()

        batch_size = len(batch["prompts"])

        noise = torch.randn(
            batch_size, self.training_num_frames, 3, self.video_height, self.video_width,
            device=self.device
        )

        # Simulate Self-Forcing inference during training
        generated_video = self.sf_engine.simulate_self_forcing(
            noise, conditional_dict
        )

        # Compute Self-Forcing loss (data-free, uses DMD)
        loss = self.sf_engine.compute_self_forcing_loss(
            generated_video,
            conditional_dict=conditional_dict,
            use_dmd=self.use_dmd_loss
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.gradient_clip_norm)

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
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "training_type": "self-forcing"
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
        """Generate sample videos for visualization during training."""
        self.generator.eval()

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
            conditional_dict = text_encoder(sample_prompts)

            batch_size = len(sample_prompts)
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
                        frames_resized = []
                        for frame in gt_vid:
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
            save_video_grid(videos_list, str(grid_path), prompts=sample_prompts)

            comparison_grid_path = None
            if gt_videos_list is not None:
                comparison_grid_path = self.samples_dir / f"step_{self.step:06d}_comparison_grid.png"
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
                wandb.log({
                    "samples/grid": wandb.Image(str(grid_path)),
                }, step=self.step)

                if comparison_grid_path is not None:
                    wandb.log({
                        "samples/comparison_grid": wandb.Image(str(comparison_grid_path)),
                    }, step=self.step)

                for i, (video_tensor, prompt) in enumerate(zip(videos_list, sample_prompts)):
                    gif_path = self.samples_dir / f"step_{self.step:06d}_sample_{i:02d}.gif"
                    wandb.log({
                        f"samples/generated_video_{i}": wandb.Video(
                            str(gif_path),
                            format="gif",
                            caption=f"Generated: {prompt}"
                        ),
                    }, step=self.step)

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


class SimpleTextEncoder(nn.Module):
    """Simple deterministic text encoder for tutorial."""

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
        """Encode text prompts deterministically."""
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

    if dataset_type == 'moving_mnist':
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
    else:
        dataset = ToyDataset(
            num_samples=num_samples,
            width=video_width,
            height=video_height,
            num_frames=video_frames,
            seed=cfg.seed
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

    # Create Self-Forcing engine
    trainer.sf_engine = SelfForcingEngine(
        generator=generator,
        scheduler=scheduler,
        device=device,
        denoising_steps=list(cfg.training.denoising_steps),
        num_frames_per_block=cfg.training.num_frames_per_block,
        context_noise=cfg.training.context_noise
    )

    # Load pretrained checkpoint if provided
    checkpoint_path = cfg.get('checkpoint', None)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"\n   Loading pretrained checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if "generator_state_dict" in checkpoint:
                generator.load_state_dict(checkpoint["generator_state_dict"])
                print(f"   Loaded generator weights from checkpoint")

            if "text_encoder_state_dict" in checkpoint:
                text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
                print(f"   Loaded text encoder weights from checkpoint")

            training_type = checkpoint.get("training_type", "unknown")
            pretrain_step = checkpoint.get("step", 0)
            print(f"   Checkpoint type: {training_type}, trained for {pretrain_step} steps")
        else:
            print(f"   Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

    # Training plotter
    plotter = TrainingPlotter(save_dir=str(trainer.plots_dir))

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

        batch["prompts"] = batch["prompt"]

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
        if trainer.step % trainer.save_interval == 0:
            trainer._save_checkpoint()

        # Generate sample videos
        if trainer.step % viz_interval == 0 and trainer.step > 0:
            print(f"\nGenerating sample videos at step {trainer.step}...")
            viz_prompts = list(cfg.viz_prompts) if cfg.viz_prompts else [
                "A red circle moving horizontally",
                "A blue square moving vertically"
            ]

            ground_truth_videos = None
            if "video" in batch:
                batch_videos = batch["video"]
                num_viz_samples = cfg.generation.num_viz_samples

                if isinstance(batch_videos, list):
                    batch_videos = [v.to(trainer.device) for v in batch_videos[:num_viz_samples]]
                    if batch_videos:
                        ground_truth_videos = torch.stack(batch_videos)
                elif isinstance(batch_videos, torch.Tensor):
                    ground_truth_videos = batch_videos[:num_viz_samples].to(trainer.device)
                    if len(ground_truth_videos.shape) == 4:
                        ground_truth_videos = ground_truth_videos.unsqueeze(0)

            trainer.generate_sample_videos(
                text_encoder=text_encoder,
                num_samples=cfg.generation.num_viz_samples,
                num_frames=cfg.generation.viz_num_frames,
                num_frames_per_block=num_frame_per_block,
                prompts=viz_prompts,
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
        pass

    trainer.generate_sample_videos(
        text_encoder=text_encoder,
        num_samples=4,
        num_frames=9,
        num_frames_per_block=3,
        ground_truth_videos=ground_truth_videos,
        gif_fps=cfg.generation.gif_fps
    )

    # Finish wandb
    if trainer.use_wandb:
        wandb.finish()

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Checkpoints: {trainer.checkpoints_dir}")
    print(f"  - Plots: {trainer.plots_dir}")
    print(f"  - Samples: {trainer.samples_dir}")
    if trainer.use_wandb:
        print(f"  - W&B logs: {output_dir / 'wandb'}")
    print("\nKey points:")
    print("1. Self-Forcing simulates inference during training")
    print("2. Videos are generated block-by-block autoregressively")
    print("3. Loss is computed on the generated videos")
    print("4. This bridges the train-test gap")


if __name__ == "__main__":
    main()
