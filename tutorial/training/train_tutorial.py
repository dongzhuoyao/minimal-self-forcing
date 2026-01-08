"""
Simplified Training Script for Self-Forcing Tutorial

This script demonstrates how to train a model using Self-Forcing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_impl"))

from tutorial.training.trainer import SimplifiedTrainer
from tutorial.data import ToyDataset
from tutorial.visualization import TrainingPlotter
from tutorial.algorithm import SimplifiedSelfForcingPipeline


class SimpleVideoGenerator(nn.Module):
    """
    Simplified video generator for tutorial.
    
    In practice, this would be a complex transformer model.
    """
    
    def __init__(self, channels=3, height=64, width=64):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Simple conv layers
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(32)
    
    def forward(self, x, timestep, conditional_dict):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, F, C, H, W)
            timestep: Timestep tensor (B, F)
            conditional_dict: Conditional information
        
        Returns:
            denoised: Denoised output (B, F, C, H, W)
            extra: Extra information
        """
        batch_size, num_frames = x.shape[:2]
        
        # Process each frame
        outputs = []
        for f in range(num_frames):
            frame = x[:, f]  # (B, C, H, W)
            
            # Simple denoising
            out = self.relu(self.norm(self.conv1(frame)))
            out = self.relu(self.conv2(out))
            out = self.conv3(out)
            
            # Residual connection
            out = out + frame
            
            outputs.append(out)
        
        denoised = torch.stack(outputs, dim=1)  # (B, F, C, H, W)
        return denoised, None


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
    """Simple text encoder for tutorial."""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
    
    def forward(self, text_prompts):
        """Encode text prompts (simplified)."""
        batch_size = len(text_prompts)
        # Return dummy embeddings
        return {
            "text_embeddings": torch.randn(
                batch_size, 77, 768, device=self.device
            )
        }


def main():
    parser = argparse.ArgumentParser(description="Train Self-Forcing model (tutorial)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of training samples")
    parser.add_argument("--log_dir", type=str, default="tutorial/logs/training", help="Log directory")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=5, help="Log metrics every N steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Self-Forcing Training Tutorial")
    print("=" * 70)
    
    # Create dataset
    print("\n1. Creating toy dataset...")
    dataset = ToyDataset(
        num_samples=args.num_samples,
        width=64,
        height=64,
        num_frames=9
    )
    print(f"   Created {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    print("\n2. Creating model...")
    generator = SimpleVideoGenerator(channels=3, height=64, width=64)
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = SimpleScheduler()
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = SimplifiedTrainer(
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )
    
    # Create text encoder
    text_encoder = SimpleTextEncoder(device=args.device)
    
    # Training plotter
    plotter = TrainingPlotter(save_dir=str(Path(args.log_dir) / "plots"))
    
    # Training loop with plotting
    print("\n4. Starting training...")
    print("-" * 70)
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Encode prompts
            with torch.no_grad():
                conditional_dict = text_encoder(batch["prompts"])
            
            # Training step
            metrics = trainer.train_step(batch, conditional_dict)
            epoch_losses.append(metrics["loss"])
            
            # Log to plotter
            plotter.log_metric("loss", metrics["loss"], trainer.step)
            
            # Print progress
            if trainer.step % args.log_interval == 0:
                print(f"Step {trainer.step}: Loss = {metrics['loss']:.4f}")
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1}/{args.num_epochs} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        trainer._save_checkpoint()
    
    # Finalize
    print("\n5. Finalizing...")
    trainer._save_checkpoint(final=True)
    trainer._save_metrics()
    
    # Plot training curves
    plotter.plot_metric("loss", title="Training Loss")
    plotter.save_history(str(Path(args.log_dir) / "metrics_history.json"))
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.log_dir}")
    print(f"Plots saved to: {Path(args.log_dir) / 'plots'}")
    print("\nKey points:")
    print("1. Self-Forcing simulates inference during training")
    print("2. Videos are generated block-by-block autoregressively")
    print("3. Loss is computed on the generated videos")
    print("4. This bridges the train-test gap")


if __name__ == "__main__":
    main()
