# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A simplified, educational implementation of the Self-Forcing algorithm for autoregressive video diffusion models. Uses Moving MNIST dataset for training and visualization. This is a data-free training method that bridges the train-test gap by simulating inference during training.

## Commands

### Training
```bash
# Basic training with Hydra config
python train.py

# Override config values
python train.py training.num_steps=2000 training.batch_size=8 training.lr=1e-4

# Load from pretrained checkpoint
python train.py training.pretrained_checkpoint=path/to/checkpoint.pt
```

### Generation
```bash
python generate.py --checkpoint logs/training/checkpoint_final.pt --prompts "Your prompt"
python generate.py --checkpoint path/to/checkpoint.pt --num_frames 12 --output_dir outputs/
```

### Testing Components
```bash
# Test Self-Forcing engine
python dynamic_video_sf.py

# Test model architecture
python tiny_causal_wan.py

# Test Moving MNIST dataset
python moving_mnist.py --visualize --num_viz 6
```

## Architecture

### Core Components

**TinyCausalWanModel** (`tiny_causal_wan.py`): Transformer-based video generation model
- Patch embedding via Conv3d
- RoPE (Rotary Position Embedding) for spatial-temporal encoding
- Block-wise causal attention mask (each block attends to all previous blocks)
- Text-to-video cross-attention
- Input/output shape: `[B, F, C, H, W]`

**SelfForcingEngine** (`dynamic_video_sf.py`): Implements block-by-block autoregressive generation
- Simulates inference during training
- KV caching for efficient autoregressive generation
- DMD (Distribution Matching Distillation) loss for data-free training
- Only last 21 frames compute gradients (efficiency optimization)

**SimplifiedTrainer** (`train.py`): Training loop with Hydra configuration
- Variable frame generation (min_num_frames to max_num_frames)
- wandb integration for experiment tracking
- Checkpointing and visualization

### Data Flow

1. Training generates noise `[B, F, C, H, W]` (F must be divisible by `num_frames_per_block`)
2. SelfForcingEngine generates video block-by-block autoregressively
3. DMD loss computed using frozen teacher network (real_score) vs trainable student (generator)
4. Gradient flows only through last 21 frames for efficiency

### Key Configuration (configs/train.yaml)

- `training.num_frames_per_block`: Frames per autoregressive block (default: 3)
- `training.denoising_steps`: List of timesteps for spatial denoising (e.g., [1000, 750, 500, 250])
- `training.min_num_frames` / `max_num_frames`: Variable frame generation range
- `model.patch_size`: Patch size for embedding [temporal, height, width]

## Important Patterns

### Tensor Formats
- Model input/output: `[B, F, C, H, W]` where F=frames, C=channels
- Visualization expects: `[F, C, H, W]` or `[B, F, C, H, W]`
- Model outputs in `[-1, 1]` range, convert to `[0, 1]` for visualization: `(x + 1.0) / 2.0`

### Frame Count Requirements
- Number of frames must be divisible by `num_frames_per_block` (default: 3)
- Example valid frame counts: 9, 12, 15, 21

### Checkpoint Format
Checkpoints contain:
- `generator_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `step`: Training step
- `training_type`: "dmd2" for Self-Forcing
- `config`: Hydra config dict

### DMD Loss
The DMD loss uses:
- `real_score`: Frozen copy of generator (teacher network)
- `generator`: Trainable student network
- Loss: `0.5 * MSE(original_latent, original_latent - grad)` where grad is normalized KL gradient
