"""
Example demonstrating the Self-Forcing algorithm.

This script shows how the algorithm works step-by-step.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_impl"))

from tutorial.algorithm import (
    SimplifiedSelfForcingPipeline,
    SimpleKVCache,
    explain_self_forcing,
    create_algorithm_diagram,
    visualize_kv_cache_growth
)
from tutorial.data import ToyDataset
from tutorial.visualization import save_video_grid


class SimpleVideoGenerator(nn.Module):
    """
    Simplified video generator for demonstration.
    
    In practice, this would be a complex transformer model.
    """
    
    def __init__(self, channels=3, height=64, width=64):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Simple conv layers for demonstration
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x, timestep, conditional_dict):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, F, C, H, W)
            timestep: Timestep tensor (B, F)
            conditional_dict: Conditional information
        
        Returns:
            denoised: Denoised output (B, F, C, H, W)
            extra: Extra information (None for simplicity)
        """
        batch_size, num_frames = x.shape[:2]
        
        # Process each frame
        outputs = []
        for f in range(num_frames):
            frame = x[:, f]  # (B, C, H, W)
            
            # Simple denoising (in practice, this would use timestep embedding)
            out = self.relu(self.conv1(frame))
            out = self.conv2(out)
            
            # Add residual
            out = out + frame
            
            outputs.append(out)
        
        denoised = torch.stack(outputs, dim=1)  # (B, F, C, H, W)
        return denoised, None


class SimpleScheduler:
    """Simple noise scheduler for demonstration."""
    
    def __init__(self):
        self.timesteps = torch.linspace(1000, 0, 1000)
    
    def add_noise(self, x, noise, timestep):
        """
        Add noise to clean data.
        
        Args:
            x: Clean data
            noise: Noise to add
            timestep: Timestep (higher = more noise)
        
        Returns:
            Noisy data
        """
        # Simplified: linear noise schedule
        alpha = 1.0 - (timestep.float() / 1000.0)
        alpha = alpha.clamp(0, 1)
        
        # Reshape for broadcasting
        if len(x.shape) == 4:  # (B*F, C, H, W)
            alpha = alpha.view(-1, 1, 1, 1)
        elif len(x.shape) == 5:  # (B, F, C, H, W)
            alpha = alpha.view(-1, 1, 1, 1, 1)
        
        return alpha * x + (1 - alpha) * noise


def demonstrate_algorithm():
    """Demonstrate the Self-Forcing algorithm."""
    print("=" * 70)
    print("Self-Forcing Algorithm Demonstration")
    print("=" * 70)
    
    # Print explanation
    print("\n" + explain_self_forcing())
    
    # Create components
    print("\n" + "-" * 70)
    print("Creating Components")
    print("-" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create simple generator
    generator = SimpleVideoGenerator(channels=3, height=64, width=64)
    generator = generator.to(device)
    print("✓ Generator created")
    
    # Create scheduler
    scheduler = SimpleScheduler()
    print("✓ Scheduler created")
    
    # Create pipeline
    pipeline = SimplifiedSelfForcingPipeline(
        generator=generator,
        scheduler=scheduler,
        num_frames_per_block=3,
        denoising_steps=[1000, 750, 500, 250],
        device=device
    )
    print("✓ Pipeline created")
    
    # Demonstrate autoregressive generation
    print("\n" + "-" * 70)
    print("Demonstrating Autoregressive Generation")
    print("-" * 70)
    
    batch_size = 1
    num_frames = 9  # 3 blocks × 3 frames per block
    channels, height, width = 3, 64, 64
    
    # Create noise
    noise = torch.randn(batch_size, num_frames, channels, height, width, device=device)
    print(f"✓ Created noise tensor: {noise.shape}")
    
    # Create dummy conditional dict
    conditional_dict = {
        "text_embeddings": torch.randn(batch_size, 77, 768, device=device)
    }
    
    # Simulate inference
    print("\nSimulating inference (block-by-block generation)...")
    try:
        generated_video, trajectory = pipeline.simulate_inference(
            noise,
            conditional_dict,
            return_trajectory=True
        )
        print(f"✓ Generated video: {generated_video.shape}")
        print(f"✓ Trajectory has {len(trajectory)} blocks")
        
        # Visualize blocks
        if len(trajectory) > 0:
            print("\nVisualizing generated blocks...")
            output_dir = Path("tutorial/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to visualization format (normalize to [-1, 1] range)
            blocks_for_viz = [block for block in trajectory]
            save_video_grid(
                blocks_for_viz,
                str(output_dir / "algorithm_blocks.png"),
                prompts=[f"Block {i+1}" for i in range(len(blocks_for_viz))]
            )
            print(f"✓ Saved block visualization to {output_dir / 'algorithm_blocks.png'}")
    
    except Exception as e:
        print(f"⚠ Note: Full simulation requires model with KV cache support")
        print(f"  Error: {e}")
        print("  This is expected - the tutorial version is simplified")
    
    # Demonstrate KV cache
    print("\n" + "-" * 70)
    print("Demonstrating KV Cache")
    print("-" * 70)
    
    cache = SimpleKVCache(
        batch_size=1,
        max_length=100,
        num_heads=12,
        head_dim=64
    )
    print("✓ KV Cache created")
    
    # Simulate cache growth
    cache_sizes = []
    for block_idx in range(7):  # 7 blocks
        # Simulate adding frames
        k = torch.randn(1, 3, 12, 64)  # 3 frames per block
        v = torch.randn(1, 3, 12, 64)
        cache.update(k, v)
        cache_sizes.append(cache.current_length)
        print(f"  Block {block_idx + 1}: Cache size = {cache.current_length}")
    
    # Visualize cache growth
    output_dir = Path("tutorial/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_kv_cache_growth(
        cache_sizes,
        str(output_dir / "kv_cache_growth.png")
    )
    print(f"✓ Saved cache growth visualization")
    
    # Create algorithm diagram
    print("\n" + "-" * 70)
    print("Creating Algorithm Diagram")
    print("-" * 70)
    
    create_algorithm_diagram(str(output_dir / "algorithm_diagram.png"))
    print(f"✓ Saved algorithm diagram")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Self-Forcing simulates inference during training")
    print("2. Videos are generated block-by-block autoregressively")
    print("3. KV cache enables efficient autoregressive generation")
    print("4. This bridges the train-test gap")
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    demonstrate_algorithm()
