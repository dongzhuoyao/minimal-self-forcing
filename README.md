# Self-Forcing Tutorial Codebase

This is a simplified, educational version of the Self-Forcing codebase designed for learning and experimentation.

## Features

- **Toy Dataset**: Generate synthetic video data without external dependencies
- **Easy Evaluation**: Simple metrics for assessing video quality
- **Visualization Tools**: Visualize videos, training progress, and comparisons
- **Simplified Code**: Clear, well-commented code for educational purposes

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision numpy pillow matplotlib imageio

# Optional: Install CLIP for text-video alignment evaluation
pip install git+https://github.com/openai/CLIP.git
```

### Generate Toy Dataset

```python
from data import ToyDataset

# Create a toy dataset with 100 synthetic videos
dataset = ToyDataset(num_samples=100, width=256, height=256, num_frames=16)

# Save prompts for training
dataset.save_prompts("data/prompts/toy_prompts.txt")
```

### Visualize Videos

```python
from visualization import create_video_gif, save_video_grid
from data import ToyDataset

dataset = ToyDataset(num_samples=9)
videos = [dataset[i]["video"] for i in range(9)]
prompts = [dataset[i]["prompt"] for i in range(9)]

# Create a grid of videos
save_video_grid(videos, "outputs/grid.png", prompts=prompts)

# Create individual GIFs
for i, video in enumerate(videos):
    create_video_gif(video, f"outputs/video_{i}.gif", fps=8)
```

### Evaluate Videos

```python
from evaluation import compute_all_metrics
from data import ToyDataset

dataset = ToyDataset(num_samples=10)
videos = [dataset[i]["video"] for i in range(10)]
prompts = [dataset[i]["prompt"] for i in range(10)]

# Compute all metrics
results = compute_all_metrics(videos, prompts)

print("Evaluation Results:")
for metric_name, value in results.items():
    if value is not None:
        print(f"  {metric_name}: {value:.4f}")
```

## Directory Structure

```
├── algorithm/                   # Self-Forcing algorithm implementation
│   ├── self_forcing_algorithm.py  # Core algorithm
│   ├── visualization.py         # Algorithm visualization
│   └── README.md               # Algorithm documentation
├── data/
│   ├── toy_dataset.py          # Synthetic dataset generator
│   └── prompts/                 # Generated prompts
├── evaluation/
│   └── metrics.py               # Evaluation metrics
├── visualization/
│   ├── video_utils.py           # Video visualization tools
│   └── training_plots.py        # Training progress plots
├── configs/
│   └── tutorial_config.yaml     # Simplified config
└── training/
    ├── trainer.py               # Simplified trainer
    └── train_tutorial.py        # Training script
```

## Components

### 0. Self-Forcing Algorithm (`algorithm/`)

**Core algorithm implementation** for understanding how Self-Forcing works:
- `SimplifiedSelfForcingPipeline`: Autoregressive generation with KV caching
- `SimpleKVCache`: KV cache for efficient generation
- `SelfForcingLoss`: Distribution matching loss
- Visualization tools for understanding the algorithm

**Key Concept**: Self-Forcing bridges the train-test gap by simulating inference during training.

**Usage:**
```python
from algorithm import SimplifiedSelfForcingPipeline, explain_self_forcing

# Understand the algorithm
print(explain_self_forcing())

# Use the pipeline
pipeline = SimplifiedSelfForcingPipeline(generator, scheduler)
generated_video = pipeline.simulate_inference(noise, conditional_dict)
```

See `algorithm/README.md` for detailed explanation.

### 1. Toy Dataset (`data/toy_dataset.py`)

Generate synthetic videos with simple animations:
- Moving shapes (circles, squares, triangles)
- Rotating shapes
- Color transitions
- Bouncing balls

**Usage:**
```python
from data import ToyVideoGenerator

generator = ToyVideoGenerator(width=256, height=256, num_frames=16)
video, prompt = generator.generate_moving_shape(
    shape="circle",
    color=(255, 0, 0),
    direction="horizontal"
)
```

### 2. Evaluation Metrics (`evaluation/metrics.py`)

Available metrics:
- **Frame Consistency**: Temporal smoothness between frames
- **CLIP Score**: Text-video alignment (requires CLIP)
- **PSNR/SSIM**: Visual quality (requires ground truth)
- **Generation Speed**: FPS measurement

**Usage:**
```python
from evaluation import FrameConsistencyMetric, CLIPScoreMetric

consistency_metric = FrameConsistencyMetric()
score = consistency_metric.compute(video)

clip_metric = CLIPScoreMetric()
clip_score = clip_metric.compute(video, "A red circle moving")
```

### 2. Training (`training/`)

**Simplified training loop** for Self-Forcing:
- `SimplifiedTrainer`: Training loop with Self-Forcing simulation
- `train_tutorial.py`: Complete training script
- Checkpoint saving and metrics logging
- Integration with visualization tools

**Usage:**
```python
from trainer import SimplifiedTrainer

trainer = SimplifiedTrainer(
    generator=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn
)
trainer.train(dataloader, num_epochs=10)
```

### 3. Visualization (`visualization/`)

Tools for visualizing videos and training:
- `save_video_grid()`: Create grid of video frames
- `create_video_gif()`: Generate GIF from video tensor
- `create_video_comparison()`: Side-by-side comparison
- `TrainingPlotter`: Plot training metrics

**Usage:**
```python
from visualization import save_video_grid, create_video_gif

save_video_grid(videos, "grid.png", prompts=prompts)
create_video_gif(video, "output.gif", fps=8)
```

## Examples

### Run Training Example

```bash
python train_tutorial.py --num_epochs 5 --batch_size 2
```

This demonstrates:
- How to train with Self-Forcing
- Simulating inference during training
- Training loop with toy dataset
- Checkpoint saving and metrics logging


## Integration with Original Codebase

To use the tutorial components with the original Self-Forcing codebase:

1. **Use toy dataset for training:**
   ```python
   from data import ToyDataset
   dataset = ToyDataset(num_samples=100)
   dataset.save_prompts("prompts/toy_prompts.txt")
   # Use in training config
   ```

2. **Evaluate generated videos:**
   ```python
   from evaluation import compute_all_metrics
   results = compute_all_metrics(generated_videos, prompts)
   ```

3. **Visualize results:**
   ```python
   from visualization import save_video_grid
   save_video_grid(videos, "results.png", prompts=prompts)
   ```

## Next Steps

1. **Explore the toy dataset**: Generate different types of synthetic videos
2. **Experiment with evaluation**: Try different metrics and see how they behave
3. **Visualize training**: Use `TrainingPlotter` to track training progress
4. **Integrate with model**: Connect tutorial components with actual model training/inference

## Notes

- The toy dataset is designed for quick experimentation and learning
- For production use, replace with real video datasets
- Some metrics (like CLIP score) require additional dependencies
- The visualization tools work with any video tensor in the format `(T, C, H, W)`

## Contributing

Feel free to extend the tutorial codebase with:
- More toy video generators
- Additional evaluation metrics
- New visualization tools
- More example scripts
