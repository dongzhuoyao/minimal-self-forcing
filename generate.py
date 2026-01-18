"""
Generate videos using a trained checkpoint.

Usage:
    python generate.py --checkpoint logs/training/checkpoint_final.pt --prompts "Your prompt here"
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import yaml
from typing import List, Dict, Tuple

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from tiny_causal_wan import TinyCausalWanModel
from trainer import SimpleScheduler, SimpleTextEncoder
from moving_mnist import create_video_gif, save_video_grid


class SimpleKVCache:
    """
    Simplified KV Cache for autoregressive generation.
    
    In autoregressive video generation, we generate frames sequentially.
    KV cache stores the key-value pairs from previous frames to avoid
    recomputing them for each new frame.
    """
    
    def __init__(self, batch_size: int, max_length: int, num_heads: int, head_dim: int):
        """
        Args:
            batch_size: Batch size
            max_length: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize empty cache
        self.k_cache = torch.zeros(batch_size, max_length, num_heads, head_dim)
        self.v_cache = torch.zeros(batch_size, max_length, num_heads, head_dim)
        self.current_length = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Update cache with new key-value pairs.
        
        Args:
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
        """
        seq_len = k.shape[1]
        end_idx = self.current_length + seq_len
        
        self.k_cache[:, self.current_length:end_idx] = k
        self.v_cache[:, self.current_length:end_idx] = v
        self.current_length = end_idx
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cache."""
        return (
            self.k_cache[:, :self.current_length],
            self.v_cache[:, :self.current_length]
        )
    
    def reset(self):
        """Reset cache."""
        self.current_length = 0


class SimplifiedSelfForcingPipeline:
    """
    Simplified Self-Forcing Training Pipeline.
    
    The key idea: Instead of training on clean latents directly, simulate
    the inference process during training. This bridges the train-test gap.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        scheduler,
        num_frames_per_block: int = 3,
        denoising_steps: List[int] = [1000, 750, 500, 250],
        device: str = "cuda"
    ):
        """
        Args:
            generator: The video generation model
            scheduler: Noise scheduler for diffusion
            num_frames_per_block: Number of frames generated per block
            denoising_steps: List of timesteps for denoising
            device: Device to run on
        """
        self.generator = generator
        self.scheduler = scheduler
        self.num_frames_per_block = num_frames_per_block
        self.denoising_steps = denoising_steps
        self.device = device
        
        # KV cache for autoregressive generation
        self.kv_cache = None
    
    def initialize_kv_cache(self, batch_size: int, max_frames: int):
        """
        Initialize KV cache for autoregressive generation.
        
        In real implementation, this would be more complex, but for tutorial
        we use a simplified version.
        """
        # Simplified: assume model has these attributes
        # In practice, these would come from model config
        num_heads = 12
        head_dim = 64
        max_length = max_frames * 100  # Approximate
        
        self.kv_cache = SimpleKVCache(
            batch_size=batch_size,
            max_length=max_length,
            num_heads=num_heads,
            head_dim=head_dim
        )
    
    def simulate_inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Simulate the inference process during training.
        
        This is the core of Self-Forcing: we generate videos autoregressively
        block by block, just like during inference, but we do it during training.
        
        Args:
            noise: Initial noise tensor of shape (B, T, C, H, W)
            conditional_dict: Conditional information (e.g., text embeddings)
            return_trajectory: Whether to return intermediate states
        
        Returns:
            Generated video latents
        """
        batch_size, num_frames, channels, height, width = noise.shape
        
        # Calculate number of blocks
        num_blocks = num_frames // self.num_frames_per_block
        
        # Initialize KV cache
        self.initialize_kv_cache(batch_size, num_frames)
        
        # Output tensor
        output = torch.zeros_like(noise)
        current_start_frame = 0
        
        trajectory = [] if return_trajectory else None
        
        # Generate block by block (autoregressive)
        for block_idx in range(num_blocks):
            # Get frames for this block
            block_frames = self.num_frames_per_block
            block_noise = noise[:, current_start_frame:current_start_frame + block_frames]
            
            # Denoise this block
            denoised_block = self._denoise_block(
                block_noise,
                conditional_dict,
                current_start_frame
            )
            
            # Store output
            output[:, current_start_frame:current_start_frame + block_frames] = denoised_block
            
            if return_trajectory:
                trajectory.append(denoised_block.detach().clone())
            
            # Update KV cache for next block
            # In practice, this would involve running the model with timestep=0
            # to update the cache with the denoised frames
            self._update_cache(denoised_block, conditional_dict, current_start_frame)
            
            current_start_frame += block_frames
        
        if return_trajectory:
            return output, trajectory
        return output
    
    def _denoise_block(
        self,
        block_noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        start_frame: int
    ) -> torch.Tensor:
        """
        Denoise a single block of frames.
        
        This performs the diffusion denoising process for one block.
        """
        noisy_input = block_noise
        batch_size, num_frames = block_noise.shape[:2]
        
        # Denoising loop
        for step_idx, timestep in enumerate(self.denoising_steps):
            # Create timestep tensor
            timestep_tensor = torch.full(
                (batch_size, num_frames),
                timestep,
                device=self.device,
                dtype=torch.long
            )
            
            # Forward pass through generator
            # In simplified version, we assume generator takes:
            # - noisy_input: (B, F, C, H, W)
            # - timestep: (B, F)
            # - conditional_dict: dict
            # - kv_cache: optional
            
            if hasattr(self.generator, 'forward_with_cache'):
                # If generator supports KV cache
                denoised, _ = self.generator.forward_with_cache(
                    noisy_input,
                    timestep_tensor,
                    conditional_dict,
                    kv_cache=self.kv_cache,
                    start_position=start_frame
                )
            else:
                # Simplified: just forward pass
                denoised, _ = self.generator(noisy_input, timestep_tensor, conditional_dict)
            
            # If not last step, add noise for next iteration
            if step_idx < len(self.denoising_steps) - 1:
                next_timestep = self.denoising_steps[step_idx + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    torch.full(
                        (batch_size * num_frames,),
                        next_timestep,
                        device=self.device,
                        dtype=torch.long
                    )
                ).unflatten(0, (batch_size, num_frames))
            else:
                # Last step: return denoised result
                return denoised
        
        return denoised
    
    def _update_cache(
        self,
        denoised_frames: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        start_frame: int
    ):
        """
        Update KV cache with denoised frames.
        
        This simulates what happens during inference: after generating a block,
        we run the model again with timestep=0 to update the cache.
        """
        if self.kv_cache is None:
            return
        
        # In practice, this would involve:
        # 1. Running generator with timestep=0 on denoised frames
        # 2. Extracting and storing KV pairs
        # For tutorial, we skip the actual implementation
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate videos using trained checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint file (overrides config)"
    )
    parser.add_argument(
        "--prompts", 
        type=str, 
        nargs="+", 
        default=None,
        help="Text prompts for video generation (overrides config)"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=None, 
        help="Number of frames to generate (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for generated videos (overrides config)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--num_frames_per_block",
        type=int,
        default=None,
        help="Number of frames per block (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)"
    )
    
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
    paths_cfg = config.setdefault('paths', {})
    gen_cfg = config.setdefault('generation', {})
    model_cfg = config.setdefault('model', {})
    
    checkpoint = args.checkpoint or paths_cfg.get('checkpoint', 'logs/training/checkpoint_final.pt')
    prompts = args.prompts or [""]
    num_frames = args.num_frames or gen_cfg.get('num_frames', 9)
    output_dir = args.output_dir or gen_cfg.get('output_dir', 'outputs/generated')
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    num_frames_per_block = args.num_frames_per_block or gen_cfg.get('num_frames_per_block', 3)
    seed = args.seed or config.get('seed', 42)
    denoising_steps = gen_cfg.get('denoising_steps', [1000, 750, 500, 250])
    
    # Model hyperparameters
    model_dim = model_cfg.get('dim', 256)
    model_ffn_dim = model_cfg.get('ffn_dim', 1024)
    model_num_heads = model_cfg.get('num_heads', 4)
    model_num_layers = model_cfg.get('num_layers', 4)
    patch_size = tuple(model_cfg.get('patch_size', [1, 4, 4]))
    text_dim = model_cfg.get('text_dim', 128)
    freq_dim = model_cfg.get('freq_dim', 256)
    
    # Text encoder config
    text_encoder_cfg = config.get('text_encoder', {})
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Validate num_frames
    if num_frames % num_frames_per_block != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by "
            f"num_frames_per_block ({num_frames_per_block})"
        )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Video Generation with Self-Forcing")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Prompts: {prompts}")
    print(f"Number of frames: {num_frames}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Setup device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    # Create model (must match training configuration)
    print("\n1. Creating model...")
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
        num_frame_per_block=num_frames_per_block,
    ).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Load checkpoint
    print(f"\n2. Loading checkpoint from {checkpoint}...")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if "generator_state_dict" in checkpoint_data:
        generator.load_state_dict(checkpoint_data["generator_state_dict"])
        print(f"   Loaded generator from checkpoint")
        if "step" in checkpoint_data:
            print(f"   Checkpoint step: {checkpoint_data['step']}")
        if "epoch" in checkpoint_data:
            print(f"   Checkpoint epoch: {checkpoint_data['epoch']}")
    else:
        # Try loading as direct state dict
        try:
            generator.load_state_dict(checkpoint_data)
            print(f"   Loaded generator state dict")
        except Exception as e:
            raise ValueError(f"Could not load checkpoint: {e}")
    
    generator.eval()  # Set to evaluation mode
    
    # Create scheduler and text encoder
    print("\n3. Setting up scheduler and text encoder...")
    scheduler = SimpleScheduler()
    text_encoder = SimpleTextEncoder(
        device=device,
        text_dim=text_encoder_cfg.get('text_dim', text_dim),
        text_len=text_encoder_cfg.get('text_len', 77),
        vocab_size=text_encoder_cfg.get('vocab_size', 256)
    )
    
    # Create pipeline
    print("\n4. Creating inference pipeline...")
    pipeline = SimplifiedSelfForcingPipeline(
        generator=generator,
        scheduler=scheduler,
        num_frames_per_block=num_frames_per_block,
        denoising_steps=denoising_steps,
        device=device
    )
    
    # Generate videos
    print(f"\n5. Generating {len(prompts)} video(s)...")
    print("-" * 70)
    
    # Encode prompts
    conditional_dict = text_encoder(prompts)
    batch_size = len(prompts)
    
    # Create noise
    # Shape: [B, F, C, H, W] where H=W=32 for tutorial
    noise = torch.randn(
        batch_size, num_frames, 3, 32, 32, 
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
        print(f"   Saved: {gif_path} (Prompt: {prompts[i]})")
        
        # Keep tensor for grid (save_video_grid also expects tensors)
        videos_list.append(video_tensor)
    
    # Save grid of all videos
    grid_path = output_dir / "generated_grid.png"
    save_video_grid(videos_list, str(grid_path), prompts=prompts)
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
