# Quick Start Guide

## 1. Generate Toy Dataset

```python
from tutorial.data import ToyDataset

# Create 10 synthetic videos
dataset = ToyDataset(num_samples=10, width=256, height=256, num_frames=16)

# Save prompts for training
dataset.save_prompts("tutorial/data/prompts/toy_prompts.txt")
```

## 2. Visualize Videos

```python
from tutorial.visualization import save_video_grid, create_video_gif
from tutorial.data import ToyDataset

dataset = ToyDataset(num_samples=9)
videos = [dataset[i]["video"] for i in range(9)]
prompts = [dataset[i]["prompt"] for i in range(9)]

# Create grid
save_video_grid(videos, "outputs/grid.png", prompts=prompts)

# Create GIFs
for i, video in enumerate(videos):
    create_video_gif(video, f"outputs/video_{i}.gif", fps=8)
```

## 3. Evaluate Videos

```python
from tutorial.evaluation import compute_all_metrics
from tutorial.data import ToyDataset

dataset = ToyDataset(num_samples=10)
videos = [dataset[i]["video"] for i in range(10)]
prompts = [dataset[i]["prompt"] for i in range(10)]

results = compute_all_metrics(videos, prompts)
print(results)
```

## 4. Run Examples

```bash
# Inference example
python tutorial/examples/inference_example.py --mode inference

# Evaluation example
python tutorial/examples/inference_example.py --mode evaluation
```

## Integration with Original Codebase

The tutorial components can be used alongside the original Self-Forcing codebase:

1. **Generate prompts for training:**
   ```python
   from tutorial.data import ToyDataset
   dataset = ToyDataset(num_samples=1000)
   dataset.save_prompts("prompts/toy_prompts.txt")
   ```

2. **Evaluate after inference:**
   ```python
   from tutorial.evaluation import compute_all_metrics
   results = compute_all_metrics(generated_videos, prompts)
   ```

3. **Visualize training progress:**
   ```python
   from tutorial.visualization import TrainingPlotter
   plotter = TrainingPlotter()
   plotter.log_metric("loss", loss_value, step)
   plotter.plot_metric("loss")
   ```
