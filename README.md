# Self-Forcing Tutorial Codebase

This is a simplified, educational version of the Self-Forcing codebase designed for learning and experimentation.

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision numpy pillow matplotlib imageio

# Optional: Install CLIP for text-video alignment evaluation
pip install git+https://github.com/openai/CLIP.git
```

## Training

Train the Self-Forcing model:

```bash
python trainer.py --num_steps 1000 --batch_size 16 --lr 1e-5
```

**Key Arguments:**
- `--num_steps`: Number of training steps (default: 1000)
- `--batch_size`: Batch size (default: 16, from config)
- `--lr`: Learning rate (default: 1e-4, from config)
- `--num_samples`: Number of training samples in toy dataset (default: 20)
- `--log_dir`: Directory to save logs and checkpoints (default: logs/training)
- `--save_interval`: Save checkpoint every N steps (default: 100)
- `--log_interval`: Log metrics every N steps (default: 5)
- `--device`: Device to use, cuda or cpu (default: auto-detects CUDA)

**Examples:**
```bash
# Basic training with defaults
python trainer.py --num_steps 1000

# Custom batch size and learning rate
python trainer.py --num_steps 1000 --batch_size 32 --lr 2e-4

# Full example with all common arguments
python trainer.py --num_steps 1000 --batch_size 16 --lr 1e-4 --num_samples 50
```

## Sampling/Generation

Generate videos using a trained checkpoint:

```bash
python generate.py --checkpoint logs/training/checkpoint_final.pt --prompts "A red circle moving horizontally"
```

**Key Arguments:**
- `--checkpoint`: Path to checkpoint file (default: logs/training/checkpoint_final.pt)
- `--prompts`: Text prompts for video generation (can specify multiple)
- `--num_frames`: Number of frames to generate (default: 9)
- `--output_dir`: Output directory for generated videos (default: outputs/generated)
- `--device`: Device to use, cuda or cpu (default: auto-detects CUDA)
- `--seed`: Random seed for reproducibility (default: 42)

**Example:**
```bash
python generate.py \
    --checkpoint logs/training/checkpoint_final.pt \
    --prompts "A red circle moving horizontally" "A blue square rotating" \
    --num_frames 12 \
    --output_dir outputs/my_generations
```

## Documentation

For more detailed information, see the documentation in the `markdown/` folder:

- **[HOW_TO_RUN.md](markdown/HOW_TO_RUN.md)**: Complete guide for training, generation, evaluation, and visualization
- **[GENERATION_GUIDE.md](markdown/GENERATION_GUIDE.md)**: Detailed guide for video generation
- **[QUICK_START.md](markdown/QUICK_START.md)**: Quick start examples
- **[ALGORITHM_SUMMARY.md](markdown/ALGORITHM_SUMMARY.md)**: Self-Forcing algorithm explanation
- **[CONDA_SETUP.md](markdown/CONDA_SETUP.md)**: Conda environment setup
- **[TUTORIAL_SUMMARY.md](markdown/TUTORIAL_SUMMARY.md)**: Tutorial overview
- **[TUTORIAL_PLAN.md](markdown/TUTORIAL_PLAN.md)**: Tutorial plan and structure

## Directory Structure

```
├── visualization/     # Visualization tools
├── configs/           # Configuration files
├── trainer.py         # Training script
├── generate.py        # Generation/sampling script
├── tiny_causal_wan.py # Model definition
├── metrics.py         # Evaluation metrics
├── toy_dataset.py     # Toy dataset generator
└── markdown/          # Detailed documentation
```

## Features

- **Toy Dataset**: Generate synthetic video data without external dependencies
- **Self-Forcing Training**: Simplified training loop with Self-Forcing algorithm
- **Video Generation**: Generate videos from text prompts
- **Evaluation Metrics**: Simple metrics for assessing video quality
- **Visualization Tools**: Visualize videos and training progress


# Update

- using pretrained checkpoints 
- check the upper bound of VAE for frame reconstruction; using Video-VAE
