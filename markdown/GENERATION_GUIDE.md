# Video Generation Guide

This guide shows you how to generate videos using a trained checkpoint.

## Quick Start

### Basic Usage

Generate a video with a single prompt:

```bash
conda activate sf
python generate.py --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A red circle moving horizontally"
```

### Generate Multiple Videos

Generate multiple videos with different prompts:

```bash
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A red circle moving horizontally" \
              "A blue square rotating clockwise" \
              "A green triangle bouncing"
```

### Customize Generation

```bash
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A yellow ball bouncing" \
    --num_frames 12 \
    --output_dir outputs/my_generations \
    --seed 123
```

## Command Line Arguments

- `--checkpoint`: Path to checkpoint file (default: `logs/training/checkpoint_final.pt`)
- `--prompts`: One or more text prompts (default: `["A red circle moving horizontally"]`)
- `--num_frames`: Number of frames to generate, must be divisible by `num_frames_per_block` (default: `9`)
- `--output_dir`: Directory to save generated videos (default: `outputs/generated`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda` if available)
- `--num_frames_per_block`: Frames per block for autoregressive generation (default: `3`)
- `--seed`: Random seed for reproducibility (default: `42`)

## Examples

### Example 1: Generate with Latest Checkpoint

```bash
# Use the final checkpoint from training
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A red circle moving horizontally"
```

### Example 2: Generate with Specific Step Checkpoint

```bash
# Use a checkpoint from a specific training step
python generate.py \
    --checkpoint tutorial/logs/training/checkpoint_step_000050.pt \
    --prompts "A blue square rotating" \
    --num_frames 9
```

### Example 3: Generate Longer Videos

```bash
# Generate 21 frames (7 blocks × 3 frames per block)
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A green triangle moving diagonally" \
    --num_frames 21
```

### Example 4: Reproducible Generation

```bash
# Use a specific seed for reproducible results
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A yellow ball bouncing" \
    --seed 42
```

## Output Files

The script generates:

1. **Individual GIFs**: `generated_000.gif`, `generated_001.gif`, etc.
   - One GIF per prompt
   - Saved in the output directory

2. **Grid Image**: `generated_grid.png`
   - Shows all generated videos in a grid
   - Includes prompts as labels

## Understanding the Generation Process

1. **Model Loading**: The script loads your trained checkpoint
2. **Noise Sampling**: Random noise is sampled for each video
3. **Text Encoding**: Prompts are encoded into embeddings
4. **Autoregressive Generation**: Videos are generated block-by-block using Self-Forcing
5. **Post-processing**: Generated latents are converted to pixel space and saved

## Troubleshooting

### Checkpoint Not Found

If you get a "checkpoint not found" error:

```bash
# List available checkpoints
ls tutorial/logs/training/checkpoint_*.pt

# Use an available checkpoint
python generate.py --checkpoint tutorial/logs/training/checkpoint_step_000010.pt ...
```

### CUDA Out of Memory

If you run out of GPU memory:

```bash
# Use CPU instead (slower but works)
python generate.py --device cpu --prompts "A red circle" ...

# Or reduce batch size by generating one video at a time
python generate.py --prompts "A red circle" ...
```

### Invalid Number of Frames

`num_frames` must be divisible by `num_frames_per_block` (default: 3):

```bash
# Valid: 9, 12, 15, 18, 21, etc.
python generate.py --num_frames 12 ...

# Invalid: 10, 11, 13, etc. (not divisible by 3)
```

## Tips

1. **Better Prompts**: Use descriptive prompts that match your training data style
2. **Experiment**: Try different seeds to see variation in generation
3. **Checkpoint Selection**: Earlier checkpoints may produce different results than later ones
4. **Frame Count**: More frames = longer videos but more computation time

## Integration with Training

After training, you can immediately generate videos:

```bash
# Train for a few steps
python trainer.py --num_steps 10000

# Generate videos with the trained model
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A red circle moving horizontally"
```
