# How to Run Tutorial Commands

This guide provides all the commands needed to train, generate, evaluate, and visualize in the Self-Forcing tutorial.

## Table of Contents

1. [Training](#training)
2. [Visualization](#visualization)
3. [Evaluation](#evaluation)
4. [Generation/Inference](#generationinference)
5. [Quick Reference](#quick-reference)

---

## Training

### Basic Training Command

```bash
python tutorial/training/train_tutorial.py --num_epochs 5 --batch_size 2
```

### Full Training Command with All Options

```bash
python tutorial/training/train_tutorial.py \
    --num_epochs 5 \
    --batch_size 2 \
    --lr 1e-4 \
    --num_samples 20 \
    --log_dir tutorial/logs/training \
    --save_interval 10 \
    --log_interval 5 \
    --device cuda
```

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_epochs` | int | 5 | Number of training epochs |
| `--batch_size` | int | 2 | Batch size for training |
| `--lr` | float | 1e-4 | Learning rate |
| `--num_samples` | int | 20 | Number of training samples in toy dataset |
| `--log_dir` | str | tutorial/logs/training | Directory to save logs and checkpoints |
| `--save_interval` | int | 10 | Save checkpoint every N steps |
| `--log_interval` | int | 5 | Log metrics every N steps |
| `--device` | str | cuda/cpu | Device to use (auto-detects CUDA if available) |

### Training Output

After training, you'll find:
- **Checkpoints**: Saved in `{log_dir}/checkpoints/`
- **Metrics**: Saved in `{log_dir}/metrics_history.json`
- **Plots**: Saved in `{log_dir}/plots/` (e.g., `loss.png`)

### Example Training Session

```bash
# Quick training run (5 epochs, 2 samples per batch)
python tutorial/training/train_tutorial.py --num_epochs 5 --batch_size 2 --num_samples 20

# Longer training run
python tutorial/training/train_tutorial.py --num_epochs 20 --batch_size 4 --num_samples 100 --log_dir tutorial/logs/long_training
```

---

## Visualization

### Visualize Multiple Videos from Toy Dataset

```bash
python tutorial/visualization/visualize_toy_dataset.py \
    --num_samples 6 \
    --output_dir tutorial/outputs/toy_dataset_visualization
```

### Visualize a Single Video

```bash
python tutorial/visualization/visualize_toy_dataset.py \
    --single 0 \
    --output_dir tutorial/outputs/toy_dataset_visualization
```

### Visualization Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_samples` | int | 6 | Number of videos to visualize |
| `--output_dir` | str | tutorial/outputs/toy_dataset_visualization | Output directory for visualizations |
| `--single` | int | None | Visualize a single video by index (optional) |

### Visualization Output

The script generates:
- **Video Grid**: `{output_dir}/video_grid.png` - Grid of middle frames from all videos
- **Individual GIFs**: `{output_dir}/video_XXX.gif` - Animated GIF for each video
- **Frame Images**: `{output_dir}/frames_sample_0/` - Individual frames from first video

### Python API for Visualization

You can also use visualization functions directly in Python:

```python
from tutorial.visualization import save_video_grid, create_video_gif
from tutorial.data import ToyDataset

# Load dataset
dataset = ToyDataset(num_samples=9)
videos = [dataset[i]["video"] for i in range(9)]
prompts = [dataset[i]["prompt"] for i in range(9)]

# Create video grid
save_video_grid(videos, "outputs/grid.png", prompts=prompts, ncols=3)

# Create individual GIFs
for i, video in enumerate(videos):
    create_video_gif(video, f"outputs/video_{i}.gif", fps=8)
```

---

## Evaluation

### Python Script for Evaluation

Currently, evaluation is done via Python code. Create a script or use interactive Python:

```python
from tutorial.evaluation import compute_all_metrics
from tutorial.data import ToyDataset

# Load videos and prompts
dataset = ToyDataset(num_samples=10)
videos = [dataset[i]["video"] for i in range(10)]
prompts = [dataset[i]["prompt"] for i in range(10)]

# Compute all metrics
results = compute_all_metrics(videos, prompts)

# Print results
print("Evaluation Results:")
for metric_name, value in results.items():
    if value is not None:
        print(f"  {metric_name}: {value:.4f}")
```

### Available Metrics

The `compute_all_metrics` function computes:

- **Frame Consistency**: Temporal smoothness between consecutive frames
- **CLIP Score**: Text-video alignment (requires CLIP installation)
- **PSNR**: Peak Signal-to-Noise Ratio (requires ground truth)
- **SSIM**: Structural Similarity Index (requires ground truth)
- **FPS**: Generation speed (requires generation times)

### Individual Metric Usage

```python
from tutorial.evaluation import (
    FrameConsistencyMetric,
    CLIPScoreMetric,
    VisualQualityMetric,
    GenerationSpeedMetric
)

# Frame consistency
consistency_metric = FrameConsistencyMetric()
score = consistency_metric.compute(video)

# CLIP score (requires CLIP: pip install git+https://github.com/openai/CLIP.git)
clip_metric = CLIPScoreMetric()
clip_score = clip_metric.compute(video, "A red circle moving horizontally")

# Visual quality (requires ground truth)
quality_metric = VisualQualityMetric()
psnr = quality_metric.compute_psnr(generated_video, ground_truth_video)
ssim = quality_metric.compute_ssim(generated_video, ground_truth_video)

# Generation speed
speed_metric = GenerationSpeedMetric()
fps = speed_metric.compute_fps(num_frames=16, generation_time=2.5)
```

### Evaluation with Ground Truth

```python
from tutorial.evaluation import compute_all_metrics
from tutorial.data import ToyDataset

# Generate dataset
dataset = ToyDataset(num_samples=10)
videos = [dataset[i]["video"] for i in range(10)]
prompts = [dataset[i]["prompt"] for i in range(10)]

# For this example, use the same videos as ground truth
# In practice, you'd have separate ground truth videos
ground_truth_videos = videos.copy()

# Compute metrics with ground truth
results = compute_all_metrics(
    videos=videos,
    prompts=prompts,
    ground_truth_videos=ground_truth_videos,
    generation_times=[2.0] * 10  # Example generation times
)

print(results)
```

### Visualize Evaluation Results

```python
from tutorial.visualization import plot_evaluation_results
from tutorial.evaluation import compute_all_metrics
from tutorial.data import ToyDataset

# Compute metrics
dataset = ToyDataset(num_samples=10)
videos = [dataset[i]["video"] for i in range(10)]
prompts = [dataset[i]["prompt"] for i in range(10)]
results = compute_all_metrics(videos, prompts)

# Plot results
plot_evaluation_results(
    results,
    save_path="tutorial/logs/plots/evaluation_results.png"
)
```

---

## Generation/Inference

### Python API for Generation

Currently, generation/inference is done via Python code using the Self-Forcing pipeline:

```python
import torch
from tutorial.algorithm import SimplifiedSelfForcingPipeline
from tutorial.model import TinyCausalWanModel
from tutorial.training.train_tutorial import SimpleScheduler, SimpleTextEncoder

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model
generator = TinyCausalWanModel(
    in_dim=3,
    out_dim=3,
    dim=256,
    ffn_dim=1024,
    num_heads=4,
    num_layers=4,
    patch_size=(1, 2, 2),
    text_dim=128,
    freq_dim=256,
    num_frame_per_block=3,
).to(device)

# Load checkpoint (if available)
# checkpoint = torch.load("tutorial/logs/training/checkpoints/checkpoint_epoch_5.pth")
# generator.load_state_dict(checkpoint["generator_state_dict"])

# Create scheduler
scheduler = SimpleScheduler()

# Create text encoder
text_encoder = SimpleTextEncoder(device=device, text_dim=128)

# Create pipeline
pipeline = SimplifiedSelfForcingPipeline(
    generator=generator,
    scheduler=scheduler,
    num_frames_per_block=3,
    denoising_steps=[1000, 750, 500, 250],
    device=device
)

# Generate video
prompt = "A red circle moving horizontally"
conditional_dict = text_encoder([prompt])

# Initialize noise
num_frames = 9
noise = torch.randn(1, num_frames, 3, 64, 64, device=device)

# Generate
with torch.no_grad():
    generated_video = pipeline.simulate_inference(noise, conditional_dict)

# Save generated video
from tutorial.visualization import create_video_gif
create_video_gif(generated_video[0], "outputs/generated_video.gif", fps=8)
```

### Generation Parameters

The `SimplifiedSelfForcingPipeline` accepts:

- `generator`: The video generation model
- `scheduler`: Noise scheduler for diffusion
- `num_frames_per_block`: Number of frames generated per block (default: 3)
- `denoising_steps`: List of timesteps for denoising (default: [1000, 750, 500, 250])
- `device`: Device to run on (default: "cuda")

### Batch Generation

```python
# Generate multiple videos
prompts = [
    "A red circle moving horizontally",
    "A blue square rotating",
    "A green triangle bouncing"
]

conditional_dict = text_encoder(prompts)
batch_size = len(prompts)
noise = torch.randn(batch_size, num_frames, 3, 64, 64, device=device)

with torch.no_grad():
    generated_videos = pipeline.simulate_inference(noise, conditional_dict)

# Save all generated videos
for i, video in enumerate(generated_videos):
    create_video_gif(video, f"outputs/generated_{i}.gif", fps=8)
```

---

## Quick Reference

### Complete Workflow Example

```bash
# 1. Train the model
python tutorial/training/train_tutorial.py \
    --num_epochs 10 \
    --batch_size 2 \
    --num_samples 50 \
    --log_dir tutorial/logs/my_training

# 2. Visualize training dataset
python tutorial/visualization/visualize_toy_dataset.py \
    --num_samples 9 \
    --output_dir tutorial/outputs/dataset_viz

# 3. Evaluate (create eval.py script)
python eval.py

# 4. Generate videos (create generate.py script)
python generate.py
```

### Example Evaluation Script (`eval.py`)

```python
#!/usr/bin/env python3
"""Evaluate videos from toy dataset."""

from tutorial.evaluation import compute_all_metrics, plot_evaluation_results
from tutorial.data import ToyDataset

def main():
    # Load dataset
    dataset = ToyDataset(num_samples=20)
    videos = [dataset[i]["video"] for i in range(20)]
    prompts = [dataset[i]["prompt"] for i in range(20)]
    
    # Compute metrics
    print("Computing metrics...")
    results = compute_all_metrics(videos, prompts)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric_name, value in results.items():
        if value is not None:
            print(f"{metric_name:25s}: {value:.4f}")
    
    # Plot results
    plot_evaluation_results(
        results,
        save_path="tutorial/logs/plots/evaluation_results.png"
    )
    print("\nPlot saved to tutorial/logs/plots/evaluation_results.png")

if __name__ == "__main__":
    main()
```

### Example Generation Script (`generate.py`)

```python
#!/usr/bin/env python3
"""Generate videos using trained model."""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "original_impl"))

from tutorial.algorithm import SimplifiedSelfForcingPipeline
from tutorial.model import TinyCausalWanModel
from tutorial.training.train_tutorial import SimpleScheduler, SimpleTextEncoder
from tutorial.visualization import create_video_gif, save_video_grid

def main():
    parser = argparse.ArgumentParser(description="Generate videos")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--prompts", type=str, nargs="+", 
                       default=["A red circle moving horizontally"],
                       help="Text prompts")
    parser.add_argument("--num_frames", type=int, default=9, help="Number of frames")
    parser.add_argument("--output_dir", type=str, default="tutorial/outputs/generated",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model
    device = args.device
    generator = TinyCausalWanModel(
        in_dim=3, out_dim=3, dim=256, ffn_dim=1024,
        num_heads=4, num_layers=4, patch_size=(1, 2, 2),
        text_dim=128, freq_dim=256, num_frame_per_block=3,
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create scheduler and text encoder
    scheduler = SimpleScheduler()
    text_encoder = SimpleTextEncoder(device=device, text_dim=128)
    
    # Create pipeline
    pipeline = SimplifiedSelfForcingPipeline(
        generator=generator,
        scheduler=scheduler,
        num_frames_per_block=3,
        denoising_steps=[1000, 750, 500, 250],
        device=device
    )
    
    # Generate videos
    print(f"Generating {len(args.prompts)} videos...")
    conditional_dict = text_encoder(args.prompts)
    batch_size = len(args.prompts)
    noise = torch.randn(batch_size, args.num_frames, 3, 64, 64, device=device)
    
    with torch.no_grad():
        generated_videos = pipeline.simulate_inference(noise, conditional_dict)
    
    # Save videos
    videos_list = [v for v in generated_videos]
    for i, (video, prompt) in enumerate(zip(videos_list, args.prompts)):
        gif_path = f"{args.output_dir}/generated_{i:03d}.gif"
        create_video_gif(video, gif_path, fps=8)
        print(f"Saved: {gif_path} (Prompt: {prompt})")
    
    # Save grid
    grid_path = f"{args.output_dir}/generated_grid.png"
    save_video_grid(videos_list, grid_path, prompts=args.prompts)
    print(f"Saved grid: {grid_path}")

if __name__ == "__main__":
    main()
```

### Running Example Scripts

```bash
# Evaluate
python eval.py

# Generate with default prompt
python generate.py

# Generate with custom prompts
python generate.py --prompts "A red circle" "A blue square" "A green triangle"

# Generate with checkpoint
python generate.py --checkpoint tutorial/logs/training/checkpoints/checkpoint_epoch_10.pth \
    --prompts "A red circle moving horizontally" \
    --output_dir tutorial/outputs/my_generations
```

---

## Additional Notes

### Dependencies

Make sure you have installed the required dependencies:

```bash
# Basic dependencies
pip install torch torchvision numpy pillow matplotlib imageio

# Optional: For CLIP score evaluation
pip install git+https://github.com/openai/CLIP.git
```

### Directory Structure

After running commands, you'll typically have:

```
tutorial/
├── logs/
│   ├── training/
│   │   ├── checkpoints/          # Model checkpoints
│   │   ├── plots/                 # Training plots
│   │   └── metrics_history.json  # Training metrics
│   └── plots/
│       └── evaluation_results.png # Evaluation plots
└── outputs/
    ├── toy_dataset_visualization/ # Dataset visualizations
    └── generated/                  # Generated videos
```

### Tips

1. **Training**: Start with small `num_samples` (10-20) for quick testing
2. **Visualization**: Use `--num_samples 6` or `9` for nice grid layouts
3. **Evaluation**: CLIP score requires GPU and CLIP installation
4. **Generation**: Make sure to load a trained checkpoint for best results
5. **Device**: Use `--device cpu` if you don't have CUDA available

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `num_samples`
2. **CLIP not available**: Install with `pip install git+https://github.com/openai/CLIP.git`
3. **Import errors**: Make sure you're running from the project root directory
4. **Checkpoint not found**: Train the model first or use a different checkpoint path

### Getting Help

- Check `tutorial/README.md` for more details
- See `tutorial/QUICK_START.md` for quick examples
- Review `tutorial/algorithm/README.md` for algorithm details
