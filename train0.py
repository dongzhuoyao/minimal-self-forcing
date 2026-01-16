"""
Pretraining Script for Self-Forcing Tutorial

This module provides supervised pretraining using ground truth videos.
The pretrained checkpoints can be loaded by train.py for Self-Forcing fine-tuning.

Pretraining uses standard diffusion training:
- Add noise to ground truth videos at random timesteps
- Train model to denoise (predict clean video or noise)
- This gives the model a good initialization before Self-Forcing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        save_interval: Optional[int] = None,
        log_interval: Optional[int] = None,
        viz_interval: Optional[int] = None,
        use_wandb: Optional[bool] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        text_encoder: Optional[nn.Module] = None
    ):
        """
        Args:
            generator: The video generation model
            optimizer: Optimizer for the generator
            scheduler: Noise scheduler
            config: Config dictionary with hyperparameters
            device: Device to train on
            log_dir: Directory to save logs and checkpoints
            save_interval: Save checkpoint every N steps
            log_interval: Log metrics every N steps
            viz_interval: Generate sample videos every N steps
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
            wandb_name: W&B run name
            text_encoder: Optional text encoder to save in checkpoints
        """
        if config is None:
            config = {}

        training_cfg = config.get('training', {})
        paths_cfg = config.get('paths', {})
        wandb_cfg = config.get('wandb', {})

        self.device = device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = Path(log_dir or paths_cfg.get('log_dir', 'logs/pretrain'))
        self.save_interval = save_interval or training_cfg.get('save_interval', 100)
        self.log_interval = log_interval or training_cfg.get('log_interval', 10)
        self.viz_interval = viz_interval or training_cfg.get('viz_interval', 100)

        # Wandb settings
        if use_wandb is None:
            use_wandb = wandb_cfg.get('enabled', False)
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        wandb_project = wandb_project or wandb_cfg.get('project', 'self-forcing-pretrain')
        wandb_entity = wandb_entity if wandb_entity is not None else wandb_cfg.get('entity', None)
        wandb_name = wandb_name if wandb_name is not None else wandb_cfg.get('name', None)

        self.generator = generator.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.text_encoder = text_encoder  # Store for checkpoint saving

        # Training config
        self.num_frames_per_block = training_cfg.get('num_frames_per_block', 3)
        self.min_timestep = training_cfg.get('min_timestep', 0)
        self.max_timestep = training_cfg.get('max_timestep', 1000)
        self.gradient_clip_norm = training_cfg.get('gradient_clip_norm', 1.0)
        self.prediction_type = training_cfg.get('prediction_type', 'sample')  # 'sample' or 'noise'

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
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name or f"pretrain-{self.log_dir.name}",
                config=config,
                dir=str(self.log_dir)
            )
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": device,
                "training_type": "pretraining"
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
        Perform one pretraining step.

        Standard diffusion training:
        1. Get ground truth video from batch
        2. Sample random timestep
        3. Add noise to video
        4. Predict denoised video
        5. Compute MSE loss against ground truth

        Args:
            batch: Batch of data containing 'video' and 'prompts'
            conditional_dict: Conditional information (text embeddings)

        Returns:
            Dictionary of metrics
        """
        self.generator.train()
        self.optimizer.zero_grad()

        # Get ground truth video [B, F, C, H, W]
        video = batch["video"]
        if isinstance(video, list):
            video = torch.stack(video)
        video = video.to(self.device)

        batch_size, num_frames = video.shape[:2]

        # Sample random timestep for each sample in batch
        timesteps = torch.randint(
            self.min_timestep, self.max_timestep,
            (batch_size,),
            device=self.device,
            dtype=torch.long
        )

        # Expand timesteps to match video shape [B, F]
        timesteps_expanded = timesteps.unsqueeze(1).expand(batch_size, num_frames)

        # Sample noise
        noise = torch.randn_like(video)

        # Add noise to video
        noisy_video = self.scheduler.add_noise(
            video.flatten(0, 1),
            noise.flatten(0, 1),
            timesteps_expanded.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))

        # Forward pass
        pred, _ = self.generator(noisy_video, timesteps_expanded, conditional_dict)

        # Compute loss
        if self.prediction_type == 'sample':
            # Predict clean video
            loss = F.mse_loss(pred, video)
        else:
            # Predict noise
            loss = F.mse_loss(pred, noise)

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
            "training_type": "pretraining"  # Mark as pretrained checkpoint
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

            # Simple iterative denoising
            denoising_steps = [1000, 750, 500, 250, 0]
            x = noise

            for i, t in enumerate(denoising_steps[:-1]):
                timestep = torch.full(
                    (batch_size, num_frames),
                    t,
                    device=self.device,
                    dtype=torch.long
                )

                pred, _ = self.generator(x, timestep, conditional_dict)

                # Move towards prediction
                if i < len(denoising_steps) - 2:
                    next_t = denoising_steps[i + 1]
                    alpha = 1.0 - (next_t / 1000.0)
                    noise_new = torch.randn_like(pred)
                    x = alpha * pred + (1 - alpha) * noise_new
                else:
                    x = pred

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
    """Simple noise scheduler for diffusion."""

    def __init__(self):
        self.timesteps = torch.linspace(1000, 0, 1000)

    def add_noise(self, x, noise, timestep):
        """Add noise to clean data at given timestep."""
        alpha = 1.0 - (timestep.float() / 1000.0)
        alpha = alpha.clamp(0, 1)

        if len(x.shape) == 4:
            alpha = alpha.view(-1, 1, 1, 1)
        elif len(x.shape) == 5:
            alpha = alpha.view(-1, 1, 1, 1, 1)

        return alpha * x + (1 - alpha) * noise


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


def main():
    """Main pretraining script."""
    parser = argparse.ArgumentParser(description="Pretrain model for Self-Forcing (supervised)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of training samples")
    parser.add_argument("--dataset", type=str, default=None, choices=['toy', 'moving_mnist'], help="Dataset type")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory")
    parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=None, help="Log metrics every N steps")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="self-forcing-pretrain", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")
        config = {}

    # Override config with command-line arguments
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
    else:
        # Default to logs/pretrain for pretraining
        config.setdefault('paths', {})['log_dir'] = 'logs/pretrain'
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

    # Extract config values
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
    num_steps = training_cfg.get('num_steps', 1000)
    batch_size = training_cfg.get('batch_size', 16)
    lr = training_cfg.get('lr', 1e-4)
    weight_decay = training_cfg.get('weight_decay', 0.01)
    num_samples = training_cfg.get('num_samples', 100)  # More samples for pretraining
    save_interval = training_cfg.get('save_interval', 100)
    log_interval = training_cfg.get('log_interval', 10)
    viz_interval = training_cfg.get('viz_interval', 200)
    log_dir = paths_cfg.get('log_dir', 'logs/pretrain')
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

    print("=" * 70)
    print("Self-Forcing Pretraining (Supervised)")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"This creates pretrained checkpoints for train.py to use")
    print("=" * 70)

    # Create dataset
    dataset_cfg = config.get('dataset', {})
    dataset_type = args.dataset or dataset_cfg.get('type', 'toy')
    dataset_type = dataset_type.lower()

    print(f"\n1. Creating {dataset_type} dataset...")

    if dataset_type == 'moving_mnist':
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
    else:
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

    # Create text encoder
    text_encoder_cfg = config.get('text_encoder', {})
    text_encoder = SimpleTextEncoder(
        device=device,
        text_dim=text_encoder_cfg.get('text_dim', text_dim),
        text_len=text_encoder_cfg.get('text_len', 77),
        vocab_size=text_encoder_cfg.get('vocab_size', 256)
    )

    # Create optimizer (include text encoder)
    optimizer = torch.optim.AdamW(
        list(generator.parameters()) + list(text_encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Create scheduler
    scheduler = SimpleScheduler()

    # Create trainer
    print("\n3. Creating trainer...")
    trainer = PretrainingTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        log_dir=log_dir,
        save_interval=save_interval,
        log_interval=log_interval,
        viz_interval=viz_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
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

        # Rename 'prompt' to 'prompts' for consistency
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
        if trainer.step % save_interval == 0:
            trainer._save_checkpoint()

        # Generate samples
        if trainer.step % viz_interval == 0 and trainer.step > 0:
            print(f"\nGenerating sample videos at step {trainer.step}...")
            viz_prompts = config.get('viz_prompts', [
                "A red circle moving horizontally",
                "A blue square moving vertically"
            ])
            trainer.generate_sample_videos(
                text_encoder=text_encoder,
                num_samples=4,
                num_frames=video_frames,
                prompts=viz_prompts,
                gif_fps=config.get('generation', {}).get('gif_fps', 2)
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
        gif_fps=config.get('generation', {}).get('gif_fps', 2)
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
    print(f"  python train.py --checkpoint {log_dir}/checkpoint_final.pt")
    print("\nKey points:")
    print("1. Pretraining uses supervised learning with ground truth videos")
    print("2. Model learns to denoise noisy videos")
    print("3. This provides a good initialization for Self-Forcing")
    print("4. Self-Forcing (train.py) can then fine-tune without ground truth")


if __name__ == "__main__":
    main()
