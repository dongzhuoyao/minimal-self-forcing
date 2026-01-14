"""
Generate videos using a trained checkpoint.

Usage:
    python generate.py --checkpoint logs/training/checkpoint_final.pt --prompts "A red circle moving horizontally"
"""

import torch
import argparse
from pathlib import Path
import sys

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from algorithm import SimplifiedSelfForcingPipeline
from model import TinyCausalWanModel
from trainer import SimpleScheduler, SimpleTextEncoder
from visualization import create_video_gif, save_video_grid


def main():
    parser = argparse.ArgumentParser(description="Generate videos using trained checkpoint")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="logs/training/checkpoint_final.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--prompts", 
        type=str, 
        nargs="+", 
        default=["A red circle moving horizontally"],
        help="Text prompts for video generation"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=9, 
        help="Number of frames to generate (must be divisible by num_frames_per_block)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/generated",
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num_frames_per_block",
        type=int,
        default=3,
        help="Number of frames per block (must divide num_frames)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Validate num_frames
    if args.num_frames % args.num_frames_per_block != 0:
        raise ValueError(
            f"num_frames ({args.num_frames}) must be divisible by "
            f"num_frames_per_block ({args.num_frames_per_block})"
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Video Generation with Self-Forcing")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompts: {args.prompts}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    # Create model (must match training configuration)
    print("\n1. Creating model...")
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
        num_frame_per_block=args.num_frames_per_block,  # Frames per block
    ).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Load checkpoint
    print(f"\n2. Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
        print(f"   Loaded generator from checkpoint")
        if "step" in checkpoint:
            print(f"   Checkpoint step: {checkpoint['step']}")
        if "epoch" in checkpoint:
            print(f"   Checkpoint epoch: {checkpoint['epoch']}")
    else:
        # Try loading as direct state dict
        try:
            generator.load_state_dict(checkpoint)
            print(f"   Loaded generator state dict")
        except Exception as e:
            raise ValueError(f"Could not load checkpoint: {e}")
    
    generator.eval()  # Set to evaluation mode
    
    # Create scheduler and text encoder
    print("\n3. Setting up scheduler and text encoder...")
    scheduler = SimpleScheduler()
    text_encoder = SimpleTextEncoder(device=device, text_dim=128)
    
    # Create pipeline
    print("\n4. Creating inference pipeline...")
    pipeline = SimplifiedSelfForcingPipeline(
        generator=generator,
        scheduler=scheduler,
        num_frames_per_block=args.num_frames_per_block,
        denoising_steps=[1000, 750, 500, 250],  # Denoising timesteps
        device=device
    )
    
    # Generate videos
    print(f"\n5. Generating {len(args.prompts)} video(s)...")
    print("-" * 70)
    
    # Encode prompts
    conditional_dict = text_encoder(args.prompts)
    batch_size = len(args.prompts)
    
    # Create noise
    # Shape: [B, F, C, H, W] where H=W=64 for tutorial
    noise = torch.randn(
        batch_size, args.num_frames, 3, 64, 64, 
        device=device
    )
    
    # Generate videos
    with torch.no_grad():
        generated_videos = pipeline.simulate_inference(noise, conditional_dict)
    
    # Convert from latents to pixel space (denormalize)
    # The model outputs in [-1, 1] range, convert to [0, 1]
    generated_videos = (generated_videos + 1.0) / 2.0
    generated_videos = generated_videos.clamp(0, 1)
    
    # Save videos
    print("\n6. Saving generated videos...")
    videos_list = []
    for i, video_tensor in enumerate(generated_videos):
        # Save individual GIF (pass tensor directly, function handles conversion)
        gif_path = output_dir / f"generated_{i:03d}.gif"
        create_video_gif(video_tensor, str(gif_path), fps=8)
        print(f"   Saved: {gif_path} (Prompt: {args.prompts[i]})")
        
        # Keep tensor for grid (save_video_grid also expects tensors)
        videos_list.append(video_tensor)
    
    # Save grid of all videos
    grid_path = output_dir / "generated_grid.png"
    save_video_grid(videos_list, str(grid_path), prompts=args.prompts)
    print(f"   Saved grid: {grid_path}")
    
    print("\n" + "=" * 70)
    print("Generation completed!")
    print("=" * 70)
    print(f"\nGenerated videos saved to: {output_dir}")
    print(f"\nTips:")
    print("- Try different prompts to see how the model responds")
    print("- Adjust --num_frames to generate longer/shorter videos")
    print("- Use --seed to get reproducible results")


if __name__ == "__main__":
    main()
