# Tutorial Codebase Plan for Self-Forcing

## Overview
This document outlines the plan for extracting a tutorial codebase from `./original_impl` that focuses on:
1. **Toy Dataset**: Simple, easy-to-understand synthetic data
2. **Easy Evaluation**: Straightforward metrics for quality assessment
3. **Visualization**: Tools to visualize training progress and generated videos

## Proposed Structure

```
tutorial/
├── README.md                    # Tutorial guide
├── requirements.txt             # Simplified dependencies
├── configs/
│   └── tutorial_config.yaml     # Simplified config for tutorial
├── data/
│   ├── toy_dataset.py          # Synthetic dataset generator
│   └── prompts/                 # Simple text prompts
├── models/
│   └── simple_model.py         # Simplified model wrapper
├── trainer.py       # Training script (includes SimplifiedTrainer class and main function)
├── inference/
│   └── inference_tutorial.py  # Simplified inference script
├── evaluation/
│   ├── metrics.py              # Simple evaluation metrics
│   └── evaluator.py            # Evaluation pipeline
├── visualization/
│   ├── video_utils.py          # Video visualization tools
│   ├── training_plots.py       # Training progress visualization
│   └── compare_videos.py       # Side-by-side video comparison
└── examples/
    ├── train_example.py        # Example training script
    └── inference_example.py     # Example inference script
```

## Key Components

### 1. Toy Dataset (`data/toy_dataset.py`)
**Purpose**: Generate synthetic video data for quick experimentation

**Features**:
- Simple geometric animations (moving shapes, color transitions)
- Text prompts describing the animations
- Configurable number of frames, resolution
- No external dependencies (pure synthetic data)
- Can generate ODE pairs for training

**Example prompts**:
- "A red circle moving from left to right"
- "A blue square rotating clockwise"
- "Color gradient transitioning from blue to red"

### 2. Evaluation (`evaluation/`)
**Purpose**: Simple metrics to assess video quality

**Metrics**:
- **CLIP Score**: Text-video alignment (using CLIP)
- **Frame Consistency**: Temporal smoothness between frames
- **Visual Quality**: Simple perceptual metrics (SSIM, PSNR if ground truth available)
- **Generation Speed**: FPS measurement

**Simplified Evaluator**:
- Easy-to-use API
- Batch evaluation support
- Results visualization

### 3. Visualization (`visualization/`)
**Purpose**: Tools to visualize and understand the model

**Tools**:
- **Video Grid**: Display multiple videos in a grid
- **Training Curves**: Plot loss, metrics over time
- **Latent Visualization**: Visualize intermediate latents (if applicable)
- **Comparison Tool**: Side-by-side comparison of videos
- **Progress Bar**: Real-time generation progress

### 4. Simplified Training (`training/`)
**Purpose**: Easy-to-understand training loop

**Features**:
- Clear step-by-step comments
- Minimal dependencies
- Progress logging
- Checkpoint saving
- Visualization integration

### 5. Simplified Inference (`inference/`)
**Purpose**: Easy-to-use inference pipeline

**Features**:
- Simple API
- Progress visualization
- Video saving
- Batch processing support

## Implementation Strategy

### Phase 1: Core Components
1. Create toy dataset generator
2. Implement basic evaluation metrics
3. Create visualization utilities
4. Simplify training script

### Phase 2: Integration
1. Integrate components into a working pipeline
2. Create example scripts
3. Add documentation

### Phase 3: Polish
1. Add error handling
2. Improve visualization
3. Add more examples
4. Create comprehensive README

## Dependencies Simplification

**Keep**:
- torch, torchvision (core)
- numpy, PIL (data processing)
- matplotlib, imageio (visualization)
- transformers (for CLIP evaluation)

**Remove/Simplify**:
- Complex distributed training (keep single GPU)
- Advanced memory optimization (keep basic)
- Multiple model variants (keep one simple path)
- Complex VAE optimizations (use standard)

## Usage Examples

### Training
```python
from trainer import SimplifiedTrainer
from data import ToyDataset

dataset = ToyDataset(num_samples=100)
# Use trainer.main() or SimplifiedTrainer class directly
# See trainer.py for usage examples
```

### Inference
```python
from inference import generate_video
from visualization import display_video

video = generate_video("A red circle moving left to right")
display_video(video)
```

### Evaluation
```python
from evaluation import evaluate_videos

results = evaluate_videos(
    videos=generated_videos,
    prompts=prompts,
    metrics=["clip_score", "frame_consistency"]
)
print(results)
```

## Benefits

1. **Educational**: Clear, well-commented code for learning
2. **Fast Iteration**: Synthetic data enables quick experiments
3. **Easy Debugging**: Simplified codebase easier to understand
4. **Visual Feedback**: Immediate visualization of results
5. **Self-Contained**: Minimal external dependencies

## Next Steps

1. Implement toy dataset generator
2. Create simplified evaluation metrics
3. Build visualization tools
4. Extract and simplify training/inference code
5. Create example scripts and documentation
