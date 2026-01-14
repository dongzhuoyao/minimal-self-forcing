"""
Self-Forcing Algorithm Implementation

This module contains the core Self-Forcing logic for autoregressive video generation:
- Block-by-block autoregressive generation with KV caching
- DMD (Distribution Matching Distillation) loss computation
- Gradient control for efficient training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List


class SelfForcingEngine:
    """
    Self-Forcing engine for autoregressive video generation.
    
    Implements the core Self-Forcing algorithm:
    - Simulates inference during training (bridging train-test gap)
    - Block-by-block generation with KV caching
    - DMD loss for data-free training
    """
    
    def __init__(
        self,
        generator: torch.nn.Module,
        scheduler: object,
        device: str,
        denoising_steps: List[int] = [1000, 750, 500, 250],
        num_frames_per_block: int = 3,
        context_noise: int = 0
    ):
        """
        Initialize Self-Forcing engine.
        
        Args:
            generator: Video generation model
            scheduler: Noise scheduler for diffusion
            device: Device to run on
            denoising_steps: List of denoising timesteps
            num_frames_per_block: Number of frames per block
            context_noise: Noise level for cache update
        """
        self.generator = generator
        self.scheduler = scheduler
        self.device = device
        self.denoising_steps = denoising_steps
        self.num_frames_per_block = num_frames_per_block
        self.context_noise = context_noise
    
    def simulate_self_forcing(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Simulate Self-Forcing inference during training.
        
        This implements block-by-block autoregressive generation with KV caching,
        matching the official Self-Forcing implementation.
        
        Key features:
        - Block-by-block generation (not all frames at once)
        - KV cache for efficient autoregressive generation
        - Gradient control: only last 21 frames compute gradients
        - Random timestep selection per block for efficiency
        
        Args:
            noise: Initial noise tensor of shape [B, F, C, H, W]
            conditional_dict: Conditional information
            
        Returns:
            Generated video latents of shape [B, F, C, H, W]
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # Calculate number of blocks
        assert num_frames % self.num_frames_per_block == 0, \
            f"num_frames ({num_frames}) must be divisible by num_frame_per_block ({self.num_frames_per_block})"
        num_blocks = num_frames // self.num_frames_per_block
        
        # Initialize output tensor
        output = torch.zeros_like(noise)
        
        # Gradient control: only compute gradients for last 21 frames
        num_output_frames = num_frames
        start_gradient_frame_index = max(0, num_output_frames - 21)
        
        # Random exit flags: which timestep to compute gradients on (for efficiency)
        # In full impl, this is synchronized across distributed processes
        num_denoising_steps = len(self.denoising_steps)
        exit_flags = [
            torch.randint(0, num_denoising_steps, (1,), device=self.device).item()
            for _ in range(num_blocks)
        ]
        
        # Block-by-block generation
        current_start_frame = 0
        all_num_frames = [self.num_frames_per_block] * num_blocks
        
        for block_index, current_num_frames in enumerate(all_num_frames):
            # Get noise for this block
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames
            ]
            
            # Spatial denoising loop (multiple timesteps)
            for index, current_timestep in enumerate(self.denoising_steps):
                exit_flag = (index == exit_flags[block_index])
                
                # Create timestep tensor for this block
                timestep = torch.full(
                    (batch_size, current_num_frames),
                    current_timestep,
                    device=self.device,
                    dtype=torch.long
                )
                
                if not exit_flag:
                    # Intermediate timesteps: no gradients (for efficiency)
                    with torch.no_grad():
                        denoised_pred, _ = self.generator(
                            noisy_input, timestep, conditional_dict
                        )
                    
                    # Add noise for next timestep
                    if index < len(self.denoising_steps) - 1:
                        next_timestep = self.denoising_steps[index + 1]
                        noise_to_add = torch.randn_like(denoised_pred)
                        alpha = 1.0 - (next_timestep / 1000.0)
                        noisy_input = alpha * denoised_pred + (1 - alpha) * noise_to_add
                else:
                    # Selected timestep: compute gradients only for last 21 frames
                    if current_start_frame < start_gradient_frame_index:
                        # Early blocks: no gradients
                        with torch.no_grad():
                            denoised_pred, _ = self.generator(
                                noisy_input, timestep, conditional_dict
                            )
                    else:
                        # Later blocks: gradients enabled
                        denoised_pred, _ = self.generator(
                            noisy_input, timestep, conditional_dict
                        )
                    break
            
            # Store output for this block
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            # Update KV cache (simplified - in full impl, this reruns with timestep=0)
            # For tutorial, we simulate cache update by running generator again with no_grad
            # This matches the official implementation where cache is updated after each block
            context_timestep = torch.full(
                (batch_size, current_num_frames),
                self.context_noise,
                device=self.device,
                dtype=torch.long
            )
            
            # Add context noise for cache update (if context_noise > 0)
            if self.context_noise > 0:
                context_noisy = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    self.context_noise * torch.ones(
                        batch_size * current_num_frames,
                        device=self.device,
                        dtype=torch.long
                    )
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                context_noisy = denoised_pred
            
            # Update cache (detached, no gradients)
            # In full implementation, this would update the actual KV cache structure
            # For tutorial, we simulate this by running generator (cache update happens internally)
            with torch.no_grad():
                _ = self.generator(
                    context_noisy, context_timestep, conditional_dict
                )
            
            # Move to next block
            current_start_frame += current_num_frames
        
        # Return only last 21 frames for loss computation (matching official impl)
        if output.shape[1] > 21:
            return output[:, -21:]
        return output
    
    def compute_self_forcing_loss(
        self,
        generated_video: torch.Tensor,
        prompts: List[str],
        conditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        use_dmd: bool = True
    ) -> torch.Tensor:
        """
        Compute Self-Forcing loss.
        
        Self-Forcing is data-free and does NOT require ground truth videos.
        Uses DMD (Distribution Matching Distillation) for distribution matching.
        
        Args:
            generated_video: Generated video tensor [B, F, C, H, W]
            prompts: List of text prompts
            conditional_dict: Conditional information (text embeddings)
            use_dmd: If True, use DMD loss; if False, use simplified temporal loss
            
        Returns:
            Loss tensor
        """
        if use_dmd:
            return self.compute_dmd_loss(generated_video, conditional_dict, prompts)
        else:
            # Fallback: simplified temporal consistency loss
            temporal_loss = 0.0
            for t in range(generated_video.shape[1] - 1):
                frame_diff = generated_video[:, t+1] - generated_video[:, t]
                temporal_loss += torch.mean(frame_diff ** 2)
            temporal_loss = temporal_loss / (generated_video.shape[1] - 1)
            
            reg_loss = torch.mean(generated_video ** 2)
            return temporal_loss + 0.1 * reg_loss
    
    def compute_dmd_loss(
        self,
        generated_video: torch.Tensor,
        conditional_dict: Optional[Dict[str, torch.Tensor]],
        prompts: List[str]
    ) -> torch.Tensor:
        """
        Compute DMD (Distribution Matching Distillation) loss.
        
        Simplified version for tutorial:
        - Uses generator itself as score network (self-distillation)
        - Computes DMD gradient and matches distributions
        
        Based on DMD paper: https://arxiv.org/abs/2311.18828
        
        Args:
            generated_video: Generated video tensor [B, F, C, H, W]
            conditional_dict: Conditional information (text embeddings)
            prompts: List of text prompts
            
        Returns:
            DMD loss tensor
        """
        original_latent = generated_video
        batch_size, num_frames = generated_video.shape[:2]
        device = generated_video.device
        
        # Create unconditional dict (null/empty embeddings)
        assert conditional_dict is not None, "conditional_dict is None"
        
        # Get the correct key name (text encoder uses "prompt_embeds")
        embed_key = "prompt_embeds"
        
        unconditional_dict = {
            embed_key: torch.zeros_like(conditional_dict[embed_key])
        }
        
        # Step 1: Randomly sample timestep for DMD
        # Use timesteps from denoising_steps to match training format
        # Sample a random timestep from the denoising steps (avoid first and last)
        if len(self.denoising_steps) > 2:
            timestep_value = self.denoising_steps[
                torch.randint(1, len(self.denoising_steps) - 1, (1,), device=device).item()
            ]
        else:
            timestep_value = self.denoising_steps[0]
        
        # Create timestep tensor with shape [B, F] matching the video shape
        timestep = torch.full(
            (batch_size, num_frames),
            timestep_value,
            device=device,
            dtype=torch.long
        )
        
        # Step 2: Add noise to generated video
        noise = torch.randn_like(generated_video)
        noisy_latent = self.scheduler.add_noise(
            generated_video.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))
        
        # Step 3: Compute KL gradient using generator as score network
        # In full DMD, this would use separate real_score and fake_score networks
        # For tutorial, we use generator itself (self-distillation)
        with torch.no_grad():
            # Conditional prediction
            _, pred_cond = self.generator(noisy_latent, timestep, conditional_dict)
            
            # Unconditional prediction (for classifier-free guidance)
            _, pred_uncond = self.generator(noisy_latent, timestep, unconditional_dict)
            
            # Classifier-free guidance (simplified, guidance_scale=1.0)
            guidance_scale = 1.0
            pred_score = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        # Step 4: Compute DMD gradient
        # DMD gradient = difference between fake and real scores
        # In simplified version: gradient = prediction difference
        grad = pred_score - original_latent.detach()
        
        # Step 5: Normalize gradient (DMD paper eq. 8)
        p_real = (original_latent - pred_score.detach())
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        normalizer = normalizer.clamp(min=1e-8)  # Avoid division by zero
        grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        
        # Step 6: Compute DMD loss (DMD paper eq. 7)
        # Loss = 0.5 * MSE(original_latent, original_latent - grad)
        dmd_loss = 0.5 * F.mse_loss(
            original_latent,
            (original_latent - grad).detach(),
            reduction='mean'
        )
        
        return dmd_loss


def demo_single_forward():
    """
    Dummy demo: Single forward pass with TinyCausalWan backbone.
    
    This demonstrates how to use the SelfForcingEngine with a TinyCausalWanModel.
    """
    print("=" * 70)
    print("Self-Forcing Engine Demo: Single Forward Pass")
    print("=" * 70)
    
    # Import here to avoid circular imports
    from tiny_causal_wan import TinyCausalWanModel
    
    # Simple scheduler for demo
    class SimpleScheduler:
        """Simple noise scheduler for demo."""
        def add_noise(self, x, noise, timestep):
            """Add noise to clean data."""
            alpha = 1.0 - (timestep.float() / 1000.0)
            alpha = alpha.clamp(0, 1)
            if len(x.shape) == 4:
                alpha = alpha.view(-1, 1, 1, 1)
            elif len(x.shape) == 5:
                alpha = alpha.view(-1, 1, 1, 1, 1)
            return alpha * x + (1 - alpha) * noise
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Device: {device}")
    
    # Model configuration
    batch_size = 2
    num_frames = 9  # Must be divisible by 3 (num_frames_per_block)
    channels = 3
    height = 64
    width = 64
    
    print(f"\n2. Model Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Frames: {num_frames}")
    print(f"   Resolution: {height}x{width}")
    print(f"   Channels: {channels}")
    
    # Create TinyCausalWanModel
    print(f"\n3. Creating TinyCausalWanModel...")
    generator = TinyCausalWanModel(
        in_dim=3,
        out_dim=3,
        dim=256,  # Smaller for demo
        ffn_dim=1024,
        num_heads=4,
        num_layers=4,
        patch_size=(1, 4, 4),
        text_dim=128,
        freq_dim=256,
        num_frame_per_block=3,
    ).to(device)
    
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Create scheduler
    scheduler = SimpleScheduler()
    
    # Create Self-Forcing engine
    print(f"\n4. Creating SelfForcingEngine...")
    sf_engine = SelfForcingEngine(
        generator=generator,
        scheduler=scheduler,
        device=device,
        denoising_steps=[1000, 750, 500, 250],
        num_frames_per_block=3,
        context_noise=0
    )
    
    # Create dummy inputs
    print(f"\n5. Creating dummy inputs...")
    noise = torch.randn(
        batch_size, num_frames, channels, height, width,
        device=device
    )
    print(f"   Noise shape: {noise.shape}")
    
    # Create dummy conditional dict (text embeddings)
    text_len = 77
    text_dim = 128
    conditional_dict = {
        "prompt_embeds": torch.randn(
            batch_size, text_len, text_dim,
            device=device
        )
    }
    print(f"   Conditional dict shape: {conditional_dict['prompt_embeds'].shape}")
    
    # Run forward pass
    print(f"\n6. Running Self-Forcing forward pass...")
    generator.eval()  # Set to eval mode for deterministic behavior
    with torch.no_grad():
        generated_video = sf_engine.simulate_self_forcing(
            noise, conditional_dict
        )
    
    print(f"   Generated video shape: {generated_video.shape}")
    print(f"   Generated video range: [{generated_video.min():.3f}, {generated_video.max():.3f}]")
    
    # Compute loss
    print(f"\n7. Computing Self-Forcing loss...")
    prompts = ["A red circle moving horizontally"] * batch_size
    generator.train()  # Set to train mode for loss computation
    
    loss = sf_engine.compute_self_forcing_loss(
        generated_video,
        prompts,
        conditional_dict=conditional_dict,
        use_dmd=True
    )
    
    print(f"   Loss value: {loss.item():.6f}")
    
    print(f"\n8. Demo completed successfully!")
    print("=" * 70)
    
    return {
        "generated_video": generated_video,
        "loss": loss,
        "model": generator
    }


if __name__ == "__main__":
    demo_single_forward()

