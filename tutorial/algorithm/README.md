# Self-Forcing Algorithm Tutorial

This directory contains a simplified, educational implementation of the Self-Forcing algorithm.

## What is Self-Forcing?

Self-Forcing is a training technique for autoregressive video diffusion models that bridges the **train-test gap** by simulating the inference process during training.

### The Problem

Traditional autoregressive models suffer from:
- **Training**: Model sees clean latents, learns to denoise
- **Inference**: Model generates autoregressively with KV caching
- **Gap**: Distributions don't match!

### The Solution

Self-Forcing solves this by:
1. **Simulating inference during training**: Generate videos block-by-block with KV caching
2. **Distribution matching**: Use distribution matching losses instead of pixel-level loss
3. **KV cache consistency**: Model learns to work with cached states

## Key Components

### 1. Autoregressive Generation (`SimplifiedSelfForcingPipeline`)

Generate videos block-by-block, maintaining KV cache:

```python
from tutorial.algorithm import SimplifiedSelfForcingPipeline

pipeline = SimplifiedSelfForcingPipeline(
    generator=your_model,
    scheduler=your_scheduler,
    num_frames_per_block=3,
    denoising_steps=[1000, 750, 500, 250]
)

# Simulate inference during training
generated_video = pipeline.simulate_inference(noise, conditional_dict)
```

### 2. KV Cache (`SimpleKVCache`)

Efficient caching for autoregressive generation:

```python
from tutorial.algorithm import SimpleKVCache

cache = SimpleKVCache(
    batch_size=1,
    max_length=1000,
    num_heads=12,
    head_dim=64
)

# Update cache with new frames
cache.update(new_keys, new_values)

# Get cached values
k_cached, v_cached = cache.get_cache()
```

### 3. Self-Forcing Loss (`SelfForcingLoss`)

Distribution matching loss:

```python
from tutorial.algorithm import SelfForcingLoss

loss_fn = SelfForcingLoss(loss_type="mse")
loss = loss_fn.compute_loss(generated_video, target_video)
```

## Algorithm Flow

```
Training Step:
1. Sample noise for entire video
2. Initialize KV cache
3. For each block:
   a. Denoise the block (using KV cache)
   b. Store denoised frames
   c. Update KV cache with denoised frames
   d. Move to next block
4. Compute distribution matching loss
5. Backpropagate through entire autoregressive process
```

## Visualization

Visualize how the algorithm works:

```python
from tutorial.algorithm import (
    visualize_autoregressive_generation,
    visualize_kv_cache_growth,
    create_algorithm_diagram
)

# Algorithm flow diagram
create_algorithm_diagram("outputs/algorithm_diagram.png")

# Visualize block-by-block generation
visualize_autoregressive_generation(video_blocks)

# Visualize KV cache growth
visualize_kv_cache_growth(cache_sizes)
```

## Educational Explanation

Get a detailed explanation:

```python
from tutorial.algorithm import explain_self_forcing

print(explain_self_forcing())
```

## Key Differences from Full Implementation

The tutorial version simplifies:

1. **KV Cache**: Simplified structure (full version has complex attention caching)
2. **Distribution Matching**: Can use MSE (full version uses DMD/SiD/CausVid)
3. **Model Interface**: Assumes standard interface (full version has Wan-specific details)

## Integration with Original Codebase

To use the full algorithm:

```python
# Use original implementation
from original_impl.pipeline.self_forcing_training import SelfForcingTrainingPipeline
from original_impl.model.dmd import DMD

# Full implementation with all features
pipeline = SelfForcingTrainingPipeline(...)
model = DMD(config, device)
```

## Learning Path

1. **Start here**: Understand the simplified algorithm
2. **Visualize**: Use visualization tools to see how it works
3. **Experiment**: Try with toy dataset
4. **Scale up**: Move to full implementation

## References

- Paper: [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009)
- Full Implementation: `original_impl/`
