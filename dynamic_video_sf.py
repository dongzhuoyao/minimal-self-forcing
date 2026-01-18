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


def compute_logit_normal_timestep_sampling(
    batch_size: int,
    num_frames: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample timestep values u in [0, 1] using logit-normal distribution.
    
    This is used for flow matching / vector field prediction to sample timesteps 
    with non-uniform distribution, which helps with convergence on large-scale models.
    
    Args:
        batch_size: Number of samples in batch
        num_frames: Number of frames per video
        logit_mean: Mean of the underlying normal distribution
        logit_std: Standard deviation of the underlying normal distribution
        device: Device to place samples on
        
    Returns:
        u: [batch_size, num_frames] tensor of values in [0, 1]
    """
    # Sample from normal distribution
    normal_samples = torch.randn(batch_size, num_frames, device=device) * logit_std + logit_mean
    # Apply sigmoid to map to (0, 1)
    u = torch.sigmoid(normal_samples)
    return u


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
    
    def _convert_prediction_to_x0(
        self,
        pred: torch.Tensor,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model prediction to x0 estimate based on prediction_type.
        
        This centralizes the conversion logic used across:
        - simulate_self_forcing: block-by-block generation
        - compute_dmd_loss: DMD loss computation
        - _generate_full_video: full video generation
        
        Args:
            pred: Model prediction (shape: [B, F, C, H, W] or [B*F, C, H, W])
            noisy_latent: Noisy input at timestep t (same shape as pred)
            timestep: Timestep tensor (shape: [B, F] or [B*F])
            
        Returns:
            x0: Denoised estimate (same shape as pred)
        """
        prediction_type = str(getattr(self, "prediction_type", "vf")).lower()
        num_train_timestep = getattr(self, 'num_train_timestep', 1000)
        
        if prediction_type == "vf":
            # Vector field prediction: convert to x0 estimate using flow matching formula
            # x0 = x_t + t * v_t, where v_t is the predicted vector field
            # Reshape timestep to match pred dimensions for broadcasting
            original_shape = pred.shape
            if len(timestep.shape) == 1:
                # Flattened case: [B*F] -> [B*F, 1, 1, 1]
                t = timestep.float() / float(num_train_timestep)
                t = t.view(-1, 1, 1, 1)
            else:
                # Unflattened case: [B, F] -> [B, F, 1, 1, 1]
                batch_size, num_frames = timestep.shape
                t = timestep.float() / float(num_train_timestep)
                t = t.view(batch_size, num_frames, 1, 1, 1)
            
            x0 = noisy_latent + t * pred
            return x0
        elif prediction_type == "x0":
            # Direct x0 prediction: use prediction as-is
            return pred
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}. Supported types: 'vf', 'x0'")
    
    def _simulate_self_forcing_core(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        exit_flags: list,
        start_gradient_frame_index: int,
        batch_size: int,
        num_blocks: int
    ) -> torch.Tensor:
        """
        Core Self-Forcing simulation logic.
        Always converts prediction to x0 first (using _convert_prediction_to_x0),
        then uses scheduler.add_noise() for next timestep.
        Matches original implementation.
        
        Returns:
            output: Tensor of shape [B, F, C, H, W] with gradients preserved
        """
        current_start_frame = 0
        all_num_frames = [self.num_frames_per_block] * num_blocks
        output_blocks = []
        
        for block_index, current_num_frames in enumerate(all_num_frames):
            # Get noise for this block
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames
            ]
            
            # Denoising loop (matches original: always uses scheduler.add_noise)
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
                        pred, _ = self.generator(
                            noisy_input, timestep, conditional_dict
                        )
                    
                    # Convert prediction to x0 estimate (handles both vf and x0)
                    denoised_pred = self._convert_prediction_to_x0(pred, noisy_input, timestep)
                    
                    # Add noise for next timestep (matches original: always stochastic)
                    if index < len(self.denoising_steps) - 1:
                        next_timestep = self.denoising_steps[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                batch_size * current_num_frames,
                                device=self.device,
                                dtype=torch.long
                            )
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Selected timestep: compute gradients only for last 21 frames
                    if current_start_frame < start_gradient_frame_index:
                        # Early blocks: no gradients
                        with torch.no_grad():
                            pred, _ = self.generator(
                                noisy_input, timestep, conditional_dict
                            )
                    else:
                        # Later blocks: gradients enabled
                        pred, _ = self.generator(
                            noisy_input, timestep, conditional_dict
                        )
                    
                    # Convert prediction to x0 estimate (handles both vf and x0)
                    denoised_pred = self._convert_prediction_to_x0(pred, noisy_input, timestep)
                    break
            
            # Store output block (maintains gradient flow)
            output_blocks.append(denoised_pred)
            
            # Update KV cache
            context_timestep = torch.full(
                (batch_size, current_num_frames),
                self.context_noise,
                device=self.device,
                dtype=torch.long
            )
            
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
            
            with torch.no_grad():
                _ = self.generator(
                    context_noisy, context_timestep, conditional_dict
                )
            
            current_start_frame += current_num_frames
        
        # Concatenate blocks along frame dimension (maintains gradient flow)
        output = torch.cat(output_blocks, dim=1)
        return output
    
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
        prediction_type = str(getattr(self, "prediction_type", "vf")).lower()
        
        if not hasattr(self, "_logged_prediction_type"):
            print(f"simulate_self_forcing: prediction_type={prediction_type}; always converts to x0 first, then uses scheduler.add_noise() (matches original).")
            self._logged_prediction_type = True

        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # Calculate number of blocks
        assert num_frames % self.num_frames_per_block == 0, \
            f"num_frames ({num_frames}) must be divisible by num_frame_per_block ({self.num_frames_per_block})"
        num_blocks = num_frames // self.num_frames_per_block
        
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
        
        # Use unified function: always converts to x0 first, then uses scheduler.add_noise()
        output = self._simulate_self_forcing_core(
            noise, conditional_dict, exit_flags,
            start_gradient_frame_index, batch_size, num_blocks
        )
        
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
    
    def generate_full_video(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        num_frames_per_block: int = 3,
        denoising_steps: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Generate full video for visualization with KV cache support.
        
        Args:
            noise: Initial noise tensor of shape [B, F, C, H, W]
            conditional_dict: Conditional information
            num_frames_per_block: Frames per block for autoregressive generation
            denoising_steps: Optional list of denoising timesteps. If None, uses self.denoising_steps
        """
        prediction_type = str(getattr(self, "prediction_type", "vf")).lower()
        
        if not hasattr(self, "_logged_generate_full_video"):
            print(f"generate_full_video: prediction_type={prediction_type}; always converts to x0 first, then uses scheduler.add_noise() (matches original).")
            self._logged_generate_full_video = True

        # Use provided denoising_steps or fall back to self.denoising_steps
        steps_to_use = denoising_steps if denoising_steps is not None else self.denoising_steps

        batch_size, num_frames, num_channels, height, width = noise.shape

        assert num_frames % num_frames_per_block == 0
        num_blocks = num_frames // num_frames_per_block

        # Initialize KV cache for autoregressive generation
        # Calculate frame_seq_length from patch_size
        patch_size = getattr(self.generator, 'patch_size', (1, 4, 4))
        h_patched = height // patch_size[1]
        w_patched = width // patch_size[2]
        frame_seq_length = h_patched * w_patched
        
        # Calculate KV cache size (use a reasonable default if local_attn_size not available)
        local_attn_size = getattr(self.generator, 'local_attn_size', -1)
        if local_attn_size != -1:
            kv_cache_size = local_attn_size * frame_seq_length
        else:
            # Default: enough for all frames
            kv_cache_size = num_frames * frame_seq_length
        
        num_layers = getattr(self.generator, 'num_layers', 4)
        num_heads = getattr(self.generator, 'num_heads', 16)
        head_dim = getattr(self.generator, 'dim', 2048) // num_heads
        
        # Initialize KV cache (one per transformer block)
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], 
                                dtype=noise.dtype, device=self.device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], 
                                dtype=noise.dtype, device=self.device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device)
            })
        
        # Initialize cross-attention cache (if needed)
        crossattn_cache = None
        if hasattr(self.generator, 'use_cross_attn') and getattr(self.generator, 'use_cross_attn', False):
            crossattn_cache = []
            for _ in range(num_layers):
                crossattn_cache.append({
                    "k": torch.zeros([batch_size, 512, num_heads, head_dim], 
                                    dtype=noise.dtype, device=self.device),
                    "v": torch.zeros([batch_size, 512, num_heads, head_dim], 
                                    dtype=noise.dtype, device=self.device),
                    "is_init": False
                })

        output = torch.zeros_like(noise)
        current_start_frame = 0
        current_start = 0  # Position in sequence for KV cache

        for block_idx in range(num_blocks):
            block_noise = noise[:, current_start_frame:current_start_frame + num_frames_per_block]
            noisy_input = block_noise

            # Unified approach: always convert to x0 first, then use scheduler.add_noise()
            for step_idx, timestep in enumerate(steps_to_use):
                timestep_tensor = torch.full(
                    (batch_size, num_frames_per_block),
                    timestep,
                    device=self.device,
                    dtype=torch.long
                )

                # Pass KV cache and current_start for autoregressive context
                pred, _ = self.generator(
                    noisy_input, 
                    timestep_tensor, 
                    conditional_dict,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start
                )

                # Convert prediction to x0 estimate (handles both vf and x0)
                denoised = self._convert_prediction_to_x0(pred, noisy_input, timestep_tensor)

                if step_idx < len(steps_to_use) - 1:
                    # Add noise for next timestep (matches original: always uses scheduler.add_noise)
                    next_timestep = steps_to_use[step_idx + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised.flatten(0, 1),
                        torch.randn_like(denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            batch_size * num_frames_per_block,
                            device=self.device,
                            dtype=torch.long
                        )
                    ).unflatten(0, denoised.shape[:2])
                else:
                    output[:, current_start_frame:current_start_frame + num_frames_per_block] = denoised

            # Update KV cache after each block (matches simulate_self_forcing)
            context_timestep = torch.full(
                (batch_size, num_frames_per_block),
                self.context_noise,
                device=self.device,
                dtype=torch.long
            )
            
            if self.context_noise > 0:
                context_noisy = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    self.context_noise * torch.ones(
                        batch_size * num_frames_per_block,
                        device=self.device,
                        dtype=torch.long
                    )
                ).unflatten(0, denoised.shape[:2])
            else:
                context_noisy = denoised
            
            # Update cache with denoised frames (no gradients needed for visualization)
            with torch.no_grad():
                _ = self.generator(
                    context_noisy,
                    context_timestep,
                    conditional_dict,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start
                )
            
            # Update positions for next block
            current_start_frame += num_frames_per_block
            current_start += num_frames_per_block * frame_seq_length

        return output
    
    def generate_sample_videos(
        self,
        num_samples: int = 4,
        num_frames: int = 9,
        num_frames_per_block: int = 3,
        digits: Optional[List[int]] = None,
        ground_truth_videos: Optional[torch.Tensor] = None,
        gif_fps: int = 2,
        samples_dir: Optional[str] = None,
        step: int = 0,
        use_wandb: bool = False,
        video_height: int = 32,
        video_width: int = 32,
        viz_denoising_steps: Optional[List] = None
    ):
        """Generate sample videos for visualization during training.
        
        Args:
            num_samples: Number of videos to generate
            num_frames: Number of frames per video
            num_frames_per_block: Frames per block for autoregressive generation
            digits: List of digit labels (0-9) to visualize. If None, uses empty labels.
            ground_truth_videos: Optional ground truth videos for comparison
            gif_fps: FPS for GIF output
            samples_dir: Directory to save samples (Path or str)
            step: Current training step (for filenames)
            use_wandb: Whether to log to wandb
            video_height: Height of generated videos
            video_width: Width of generated videos
            viz_denoising_steps: Optional denoising steps for visualization. 
                Can be:
                - A single list of ints: [1000, 500] (single configuration)
                - A list of lists: [[1000], [1000, 500], [1000, 750, 500, 250]] (multiple configurations)
                - None: uses self.denoising_steps
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

        # Import visualization functions (optional)
        try:
            from moving_mnist import create_video_gif, save_video_grid
        except ImportError:
            print("Warning: moving_mnist not available, skipping video visualization")
            self.generator.train()
            return

        # Import wandb (optional)
        try:
            import wandb
            WANDB_AVAILABLE = True
        except ImportError:
            WANDB_AVAILABLE = False
            if use_wandb:
                print("Warning: wandb not available, skipping wandb logging")

        # Normalize viz_denoising_steps to list of lists
        if viz_denoising_steps is None:
            viz_denoising_steps = [self.denoising_steps]
        elif isinstance(viz_denoising_steps[0], int):
            # Single configuration: convert to list of lists
            viz_denoising_steps = [viz_denoising_steps]
        # Otherwise, it's already a list of lists

        # Convert samples_dir to Path if string
        from pathlib import Path
        if samples_dir is not None:
            samples_dir = Path(samples_dir)
            samples_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("Warning: samples_dir not provided, skipping file saving")
            self.generator.train()
            return

        with torch.no_grad():
            # Use empty conditional dict (model will create dummy embeddings if needed)
            conditional_dict = {}

            batch_size = num_samples
            # Use the same noise for all configurations for fair comparison
            noise = torch.randn(
                batch_size, num_frames, 3, video_height, video_width,
                device=self.device
            )

            # Process ground truth videos if provided (only need to do this once)
            gt_videos_list = None
            if ground_truth_videos is not None:
                if ground_truth_videos.device != self.device:
                    ground_truth_videos = ground_truth_videos.to(self.device)

                if ground_truth_videos.min() < 0:
                    ground_truth_videos = (ground_truth_videos + 1.0) / 2.0
                ground_truth_videos = ground_truth_videos.clamp(0, 1)

                if ground_truth_videos.shape[1] != num_frames:
                    ground_truth_videos = ground_truth_videos[:, :num_frames]

                ground_truth_videos = ground_truth_videos[:num_samples]
                gt_videos_list = [gt_video for gt_video in ground_truth_videos]

            # Generate videos for each configuration
            all_configs_videos = {}
            all_configs_gif_paths = {}
            
            for config_idx, steps_config in enumerate(viz_denoising_steps):
                num_steps = len(steps_config)
                config_name = f"{num_steps}step"
                
                print(f"  Generating videos with {num_steps}-step sampling: {steps_config}")
                
                generated_videos = self.generate_full_video(
                    noise, conditional_dict, num_frames_per_block, 
                    denoising_steps=steps_config
                )

                # Normalize to [0, 1]
                generated_videos = (generated_videos + 1.0) / 2.0
                generated_videos = generated_videos.clamp(0, 1)
                
                all_configs_videos[config_name] = generated_videos
                
                # Save GIFs for this configuration
                config_gif_paths = []
                videos_list = []
                for i, video_tensor in enumerate(generated_videos):
                    gif_path = samples_dir / f"step_{step:06d}_sample_{i:02d}_{config_name}.gif"
                    create_video_gif(video_tensor, str(gif_path), fps=gif_fps)
                    config_gif_paths.append(gif_path)
                    videos_list.append(video_tensor)
                
                all_configs_gif_paths[config_name] = config_gif_paths
                
                # Save grid for this configuration
                grid_path = samples_dir / f"step_{step:06d}_grid_{config_name}.png"
                if digit_labels:
                    labels_for_viz = [f"Digit {d}" if d is not None else "" for d in digit_labels]
                else:
                    labels_for_viz = [""] * num_samples
                save_video_grid(videos_list, str(grid_path), prompts=labels_for_viz)

            # Save ground truth videos (only once, shared across all configs)
            gt_gif_paths = []
            if gt_videos_list is not None:
                for i, gt_video in enumerate(gt_videos_list):
                    gt_gif_path = samples_dir / f"step_{step:06d}_gt_{i:02d}.gif"
                    create_video_gif(gt_video, str(gt_gif_path), fps=gif_fps)
                    gt_gif_paths.append(gt_gif_path)

            # Create comparison grids for each configuration
            for config_name, videos_list in all_configs_videos.items():
                if gt_videos_list is not None:
                    comparison_grid_path = samples_dir / f"step_{step:06d}_comparison_grid_{config_name}.png"
                    comparison_videos = []
                    comparison_prompts = []
                    if digit_labels:
                        labels_for_viz = [f"Digit {d}" if d is not None else "" for d in digit_labels]
                    else:
                        labels_for_viz = [""] * num_samples
                    for gen_vid, gt_vid, label in zip(videos_list, gt_videos_list, labels_for_viz):
                        comparison_videos.append(gen_vid)
                        comparison_videos.append(gt_vid)
                        comparison_prompts.append(f"{label} (Generated {config_name})")
                        comparison_prompts.append(f"{label} (Ground Truth)")
                    save_video_grid(comparison_videos, str(comparison_grid_path), prompts=comparison_prompts, ncols=2)

            print(f"  Saved sample videos to {samples_dir} (configurations: {', '.join(all_configs_videos.keys())})")

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                samples_payload = {}
                
                # Log grids for each configuration
                for config_name, videos_list in all_configs_videos.items():
                    grid_path = samples_dir / f"step_{step:06d}_grid_{config_name}.png"
                    samples_payload[f"vis_sample_{config_name}"] = wandb.Image(str(grid_path))
                    
                    if gt_videos_list is not None:
                        comparison_grid_path = samples_dir / f"step_{step:06d}_comparison_grid_{config_name}.png"
                        samples_payload[f"vis_comparison_{config_name}"] = wandb.Image(str(comparison_grid_path))

                # Create a combined table with all configurations
                samples_table = wandb.Table(columns=[
                    "index",
                    "digit",
                    "config",
                    "generated_video",
                    "ground_truth_video",
                ])
                for config_name, gif_paths in all_configs_gif_paths.items():
                    for i, gif_path in enumerate(gif_paths):
                        caption = f"Digit {digit_labels[i]}" if digit_labels and digit_labels[i] is not None else ""
                        generated_video = wandb.Video(
                            str(gif_path),
                            format="gif",
                            caption=f"{caption} ({config_name})",
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
                        samples_table.add_data(i, caption, config_name, generated_video, gt_video)
                samples_payload["vis_all_configs"] = samples_table

                wandb.log(samples_payload, step=step)

        self.generator.train()
    
    def compute_dmd_loss(
        self,
        generated_video: torch.Tensor,
        conditional_dict: Optional[Dict[str, torch.Tensor]],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
        embed_key: str = "prompt_embeds",
        compute_generator_gradient: bool = True
    ):
        """
        Compute DMD (Distribution Matching Distillation) loss.
        
        Full implementation matching official DMD:
        - Uses generator itself as score network (self-distillation)
        - Computes DMD gradient and matches distributions
        - Supports timestep scheduling based on denoised timesteps
        - Uses logit-normal timestep sampling for vector field prediction (flow matching style)
        
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
            timestep_shift: Timestep shift factor (only used for non-vf prediction types)
            ts_schedule: Use timestep scheduling
            guidance_scale: Classifier-free guidance scale
            prediction_type: "vf" (vector field) uses logit-normal sampling, others use uniform
            logit_mean: Mean for logit-normal distribution (default: 0.0)
            logit_std: Std for logit-normal distribution (default: 1.0)
            
        Returns:
            DMD loss tensor and log dict
        """
        if not compute_generator_gradient:
            generated_video = generated_video.detach()
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
            prediction_type = getattr(self, "prediction_type", "vf").lower()
            
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
            
            # Use logit-normal sampling for vector field prediction (flow matching style)
            # Otherwise use uniform sampling (original DMD)
            if prediction_type == "vf":
                # Logit-normal timestep sampling (similar to minimal-dmd-main flow matching)
                logit_mean = getattr(self, 'logit_mean', 0.0)
                logit_std = getattr(self, 'logit_std', 1.0)
                
                # Sample u in [0, 1] using logit-normal distribution
                u = compute_logit_normal_timestep_sampling(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    logit_mean=logit_mean,
                    logit_std=logit_std,
                    device=device
                )
                
                # Convert u to timestep indices in [min_timestep, max_timestep]
                timestep_float = min_timestep + u * (max_timestep - min_timestep)
                timestep = timestep_float.long()
                
                # Clamp to valid range
                timestep = torch.clamp(timestep, min_step, max_step)
            else:
                # Uniform timestep sampling (original DMD)
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
        if compute_generator_gradient:
            pred_fake_cond, _ = self.generator(noisy_latent, timestep, conditional_dict)
        else:
            with torch.no_grad():
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
            raise NotImplementedError("currently not supported for CFG, as I don't know how to do it when prediction_type is 'vf'")
            if compute_generator_gradient:
                pred_fake_uncond, _ = self.generator(noisy_latent, timestep, unconditional_dict)
            else:
                with torch.no_grad():
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

        # Step 4: Convert predictions to x0 estimates based on prediction_type
        pred_fake_x0 = self._convert_prediction_to_x0(pred_fake, noisy_latent, timestep)
        pred_real_x0 = self._convert_prediction_to_x0(pred_real, noisy_latent, timestep)
        
        # Step 5: Compute DMD gradient (DMD paper eq. 7)
        # grad = (p_real - p_fake); pred_fake_x0 has gradients, pred_real_x0 is detached
        p_real = (original_latent - pred_real_x0)
        p_fake = (original_latent - pred_fake_x0)
        grad = (p_real - p_fake)
        
        # Debug: Check if pred_fake and pred_real are identical (which would make grad=0)
        pred_diff = torch.mean(torch.abs(pred_fake_x0 - pred_real_x0)).item()
        guidance_loss = F.mse_loss(
            pred_fake_x0.detach(),
            pred_real_x0.detach(),
            reduction="mean"
        )
        
        # Step 6: Normalize gradient (DMD paper eq. 8)
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        normalizer = normalizer.clamp(min=1e-8)  # Avoid division by zero
        grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        # Step 7: Compute DMD loss (DMD paper eq. 7)
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
            "pred_fake_norm": torch.mean(torch.abs(pred_fake_x0)).detach(),
            "pred_real_norm": torch.mean(torch.abs(pred_real_x0)).detach(),
            "original_latent_norm": torch.mean(torch.abs(original_latent)).detach(),
            "guidance_loss": guidance_loss.detach(),
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
    
    loss, _ = sf_engine.compute_dmd_loss(
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
