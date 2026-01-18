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
        context_noise: int = 0,
        same_step_across_blocks: bool = True
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
        self.generator = generator  # fake_score (trainable)
        self.scheduler = scheduler
        self.device = device
        self.denoising_steps = denoising_steps
        self.num_frames_per_block = num_frames_per_block
        self.context_noise = context_noise
        self.same_step_across_blocks = same_step_across_blocks
        
        # Create frozen copy of generator for real_score (teacher network)
        # Matching original implementation: real_score is a separate frozen model
        # In original impl, real_score is loaded from a separate model_name (e.g., "Wan2.1-T2V-1.3B")
        # For simplified version, we use a deep copy of the generator
        import copy
        self.real_score = copy.deepcopy(generator)
        for param in self.real_score.parameters():
            param.requires_grad = False
        self.real_score.eval()
    
    def update_real_score(self, ema_decay: float = 0.0):
        """
        Update the frozen real_score model with weights from generator.
        This is used for self-distillation where teacher is updated periodically.
        
        Args:
            ema_decay: EMA decay factor. If 0, copy weights directly. If > 0, use EMA update.
        """
        if ema_decay == 0.0:
            # Direct copy (matching original implementation where real_score is separate)
            self.real_score.load_state_dict(self.generator.state_dict())
        else:
            # EMA update for self-distillation
            with torch.no_grad():
                for real_param, gen_param in zip(self.real_score.parameters(), self.generator.parameters()):
                    real_param.data.mul_(ema_decay).add_(gen_param.data, alpha=1 - ema_decay)
    
    def simulate_self_forcing(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        return_timesteps: bool = False
    ):
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
        if self.same_step_across_blocks:
            exit_flag = torch.randint(0, num_denoising_steps, (1,), device=self.device).item()
            exit_flags = [exit_flag] * num_blocks
        else:
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
            output = output[:, -21:]
        
        # Compute denoised timestep range (for DMD timestep scheduling)
        denoised_timestep_from = None
        denoised_timestep_to = None
        if return_timesteps and exit_flags:
            # If exit steps differ across blocks, timestep scheduling is ambiguous.
            all_same_exit = all(exit_flags[0] == v for v in exit_flags)
            if not all_same_exit:
                if getattr(self, "ts_schedule", False) or getattr(self, "ts_schedule_max", False):
                    raise ValueError(
                        "Per-block exit timesteps detected by AI while ts_schedule/ts_schedule_max is enabled. "
                        "Disable timestep scheduling or enforce a shared exit timestep across blocks."
                    )
            else:
                # Find the exit timestep used (for timestep scheduling in DMD)
                exit_timestep_idx = exit_flags[0]
                if exit_timestep_idx < len(self.denoising_steps) - 1:
                    denoised_timestep_to = 1000 - self.denoising_steps[exit_timestep_idx + 1]
                    denoised_timestep_from = 1000 - self.denoising_steps[exit_timestep_idx]
                else:
                    denoised_timestep_to = 0
                    denoised_timestep_from = 1000 - self.denoising_steps[exit_timestep_idx]
        
        if return_timesteps:
            return output, denoised_timestep_from, denoised_timestep_to
        return output
    
    def compute_self_forcing_loss(
        self,
        generated_video: torch.Tensor,
        conditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None
    ):
        """
        Compute Self-Forcing loss.
        
        Self-Forcing is data-free and does NOT require ground truth videos.
        Uses DMD (Distribution Matching Distillation) for distribution matching.
        
        Args:
            generated_video: Generated video tensor [B, F, C, H, W]
            conditional_dict: Conditional information (text embeddings)
            unconditional_dict: Unconditional information (null embeddings)
            denoised_timestep_from: Start timestep for DMD scheduling
            denoised_timestep_to: End timestep for DMD scheduling
            
        Returns:
            Loss tensor and log dict
        """
        return self.compute_dmd_loss(
            generated_video, 
            conditional_dict, 
            unconditional_dict,
            denoised_timestep_from,
            denoised_timestep_to
        )
    
    def compute_dmd_loss(
        self,
        generated_video: torch.Tensor,
        conditional_dict: Optional[Dict[str, torch.Tensor]],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
        embed_key: str = "prompt_embeds"
    ):
        """
        Compute DMD (Distribution Matching Distillation) loss.
        
        Full implementation matching official DMD:
        - Uses generator itself as score network (self-distillation)
        - Computes DMD gradient and matches distributions
        - Supports timestep scheduling based on denoised timesteps
        
        Based on DMD paper: https://arxiv.org/abs/2311.18828
        
        Args:
            generated_video: Generated video tensor [B, F, C, H, W]
            conditional_dict: Conditional information (text embeddings)
            unconditional_dict: Unconditional information (null embeddings)
            denoised_timestep_from: Start timestep for scheduling
            denoised_timestep_to: End timestep for scheduling
            embed_key: Key for embeddings in dict
            num_train_timestep: Total training timesteps
            min_step: Minimum timestep for sampling
            max_step: Maximum timestep for sampling
            timestep_shift: Timestep shift factor
            ts_schedule: Use timestep scheduling
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            DMD loss tensor and log dict
        """
        original_latent = generated_video
        batch_size, num_frames = generated_video.shape[:2]
        device = generated_video.device
        
        # Create unconditional dict if not provided
        assert conditional_dict is not None, "conditional_dict is None"
        if unconditional_dict is None:
            unconditional_dict = {
                embed_key: torch.zeros_like(conditional_dict[embed_key])
            }
        
        # Step 1: Sample timestep for DMD (with optional scheduling)
        with torch.no_grad():
            ts_schedule = getattr(self, 'ts_schedule', False)
            ts_schedule_max = getattr(self, 'ts_schedule_max', False)
            num_train_timestep = getattr(self, 'num_train_timestep', 1000)
            min_step = getattr(self, 'min_step', 20)
            max_step = getattr(self, 'max_step', 980)
            min_score_timestep = getattr(self, 'min_score_timestep', 0)
            timestep_shift = getattr(self, 'timestep_shift', 1.0)
            
            # Determine min_timestep: use denoised_timestep_to if ts_schedule is enabled, else min_score_timestep
            if ts_schedule and denoised_timestep_to is not None:
                min_timestep = denoised_timestep_to
            else:
                min_timestep = min_score_timestep
            
            # Determine max_timestep: use denoised_timestep_from if ts_schedule_max is enabled, else num_train_timestep
            if ts_schedule_max and denoised_timestep_from is not None:
                max_timestep = denoised_timestep_from
            else:
                max_timestep = num_train_timestep
            
            # Sample uniform timestep
            timestep_value = torch.randint(
                min_timestep, max_timestep + 1, (1,), device=device
            ).item()
            
            # Apply timestep shift if needed (matching original DMD implementation)
            if timestep_shift > 1.0:
                timestep_value = int(
                    timestep_shift * (timestep_value / 1000.0) / 
                    (1 + (timestep_shift - 1) * (timestep_value / 1000.0)) * 1000
                )
                # Clamp to [min_step, max_step] after shift
                timestep_value = max(min_step, min(max_step, timestep_value))
        
        # Create timestep tensor with shape [B, F] matching the video shape
        timestep = torch.full(
            (batch_size, num_frames),
            timestep_value,
            device=device,
            dtype=torch.long
        )
        
        # Step 2: Add noise to generated video (no gradients through noise path)
        with torch.no_grad():
            noise = torch.randn_like(generated_video)
            noisy_latent = self.scheduler.add_noise(
                generated_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
        noisy_latent = noisy_latent.detach()
        
        # Step 3: Compute KL gradient using separate fake_score and real_score networks
        # Matching original DMD implementation:
        # - fake_score: trainable generator (student)
        # - real_score: frozen copy of generator (teacher)
        
        # Fake score: compute WITH gradients (student network)
        self.generator.train()
        pred_fake_cond, _ = self.generator(noisy_latent, timestep, conditional_dict)
        
        # Unconditional prediction (for classifier-free guidance)
        # Matching original DMD implementation:
        # - If real_guidance_scale and fake_guidance_scale are set: use them
        # - Otherwise: real_guidance_scale = guidance_scale, fake_guidance_scale = 0.0
        guidance_scale = getattr(self, 'guidance_scale', 1.0)
        
        # Check if real_guidance_scale and fake_guidance_scale are explicitly set
        has_separate_scales = hasattr(self, 'real_guidance_scale') and hasattr(self, 'fake_guidance_scale')
        
        if has_separate_scales:
            fake_guidance_scale = self.fake_guidance_scale
            real_guidance_scale = self.real_guidance_scale
        else:
            # Default: fake_guidance_scale = 0.0, real_guidance_scale = guidance_scale
            fake_guidance_scale = 0.0
            real_guidance_scale = guidance_scale
        
        # Fake score: compute WITH gradients (student network)
        if fake_guidance_scale != 0.0:
            pred_fake_uncond, _ = self.generator(noisy_latent, timestep, unconditional_dict)
            pred_fake = pred_fake_cond + fake_guidance_scale * (pred_fake_cond - pred_fake_uncond)
        else:
            pred_fake = pred_fake_cond
        
        # Real score: compute WITHOUT gradients (teacher network, frozen)
        # Use frozen copy of generator (real_score)
        with torch.no_grad():
            pred_real_cond, _ = self.real_score(noisy_latent, timestep, conditional_dict)
            
            if real_guidance_scale != 0.0:
                pred_real_uncond, _ = self.real_score(noisy_latent, timestep, unconditional_dict)
                pred_real = pred_real_cond + real_guidance_scale * (pred_real_cond - pred_real_uncond)
            else:
                pred_real = pred_real_cond
        
        # Explicitly detach pred_real to ensure it doesn't contribute to gradients
        pred_real = pred_real.detach()
        
        # Step 4: Compute DMD gradient (DMD paper eq. 7)
        # grad = pred_fake - pred_real
        # pred_fake has gradients, pred_real is detached, so grad has gradients from pred_fake
        grad = (pred_fake - pred_real)
        
        # Debug: Check if pred_fake and pred_real are identical (which would make grad=0)
        pred_diff = torch.mean(torch.abs(pred_fake - pred_real)).item()
        
        # Step 5: Normalize gradient (DMD paper eq. 8)
        p_real = (original_latent - pred_real)
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        normalizer = normalizer.clamp(min=1e-8)  # Avoid division by zero
        grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        
        # Step 6: Compute DMD loss (DMD paper eq. 7)
        # Matching original implementation: loss = 0.5 * MSE(original_latent, original_latent - grad)
        # IMPORTANT: original_latent has gradients (from generator), grad is detached
        dmd_loss = 0.5 * F.mse_loss(
            original_latent.double(),
            (original_latent.double() - grad.double()).detach(),
            reduction='mean'
        )
        
        # Log dict for metrics
        log_dict = {
            "dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.float().mean().detach(),
            "pred_diff": pred_diff,  # Debug: difference between pred_fake and pred_real
            "pred_fake_norm": torch.mean(torch.abs(pred_fake)).detach(),
            "pred_real_norm": torch.mean(torch.abs(pred_real)).detach(),
            "original_latent_norm": torch.mean(torch.abs(original_latent)).detach(),
        }
        
        return dmd_loss, log_dict


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
    height = 32
    width = 32
    
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
    generator.train()  # Set to train mode to enable gradients for loss computation
    # Note: simulate_self_forcing internally controls gradients (only last 21 frames)
    generated_video = sf_engine.simulate_self_forcing(
        noise, conditional_dict
    )
    
    print(f"   Generated video shape: {generated_video.shape}")
    print(f"   Generated video range: [{generated_video.min():.3f}, {generated_video.max():.3f}]")
    
    # Compute loss
    print(f"\n7. Computing Self-Forcing loss...")
    # Generator is already in train mode
    
    loss, _ = sf_engine.compute_self_forcing_loss(
        generated_video,
        conditional_dict=conditional_dict
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
