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

# Add root directory to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))


from trainer import SimplifiedTrainer
from data import ToyDataset
from visualization import TrainingPlotter
from model import TinyCausalWanModel


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
    
    def __init__(self, device="cuda", text_dim=128):
        super().__init__()
        self.device = device
        self.text_dim = text_dim
    
    def forward(self, text_prompts):
        """Encode text prompts (simplified)."""
        batch_size = len(text_prompts)
        # Return dummy embeddings matching TinyCausalWanModel's expected format
        # Shape: [B, text_len, text_dim] where text_len=77 (standard)
        text_len = 77
        return {
            "prompt_embeds": torch.randn(
                batch_size, text_len, self.text_dim, device=self.device
            )
        }


def main():
    parser = argparse.ArgumentParser(description="Train Self-Forcing model (tutorial)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of training samples")
    parser.add_argument("--log_dir", type=str, default="logs/training", help="Log directory")
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
    # Use TinyCausalWanModel
    generator = TinyCausalWanModel(
        in_dim=3,  # RGB channels
        out_dim=3,
        dim=256,  # Hidden dimension 
        ffn_dim=1024,  # FFN dimension 
        num_heads=4,  # Attention heads 
        num_layers=4,  # Transformer layers 
        patch_size=(1, 2, 2),  # Patch size for embedding
        text_dim=128,  # Text embedding dimension
        freq_dim=256,  # Time embedding dimension
        num_frame_per_block=3,  # Frames per block for causal mask
    )
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"   Using TinyCausalWanModel (transformer backbone)")
    
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
    
    # Create text encoder (matching TinyCausalWanModel's text_dim)
    text_encoder = SimpleTextEncoder(device=args.device, text_dim=128)
    
    # Training plotter
    plotter = TrainingPlotter(save_dir=str(Path(args.log_dir) / "plots"))
    
    # Training loop with plotting
    print("\n4. Starting training...")
    print("-" * 70)
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # DataLoader batches "prompt" (singular) from dataset as a list
            # Trainer expects "prompts" (plural), so rename it
            batch["prompts"] = batch["prompt"]
            
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
