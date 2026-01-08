# Self-Forcing Algorithm Tutorial Summary

## What Was Added

I've created a comprehensive **algorithm tutorial** that explains and demonstrates the core Self-Forcing algorithm.

## Components Created

### 1. Core Algorithm (`tutorial/algorithm/self_forcing_algorithm.py`)

**SimplifiedSelfForcingPipeline**: The main algorithm implementation
- Simulates inference during training
- Generates videos block-by-block autoregressively
- Maintains KV cache between blocks
- Educational comments explaining each step

**SimpleKVCache**: Simplified KV cache implementation
- Shows how key-value pairs are cached
- Demonstrates cache growth during generation
- Easy to understand structure

**SelfForcingLoss**: Loss function explanation
- Explains distribution matching concept
- Shows how it differs from standard diffusion loss
- Placeholder for full implementation

**explain_self_forcing()**: Educational explanation
- Markdown-formatted explanation
- Covers problem, solution, and algorithm flow
- Perfect for learning

### 2. Visualization Tools (`tutorial/algorithm/visualization.py`)

**Visualization Functions**:
- `visualize_autoregressive_generation()`: Show block-by-block generation
- `visualize_kv_cache_growth()`: Plot cache growth over time
- `visualize_denoising_process()`: Show denoising steps
- `create_algorithm_diagram()`: Create flow diagram

### 3. Example Script (`tutorial/examples/algorithm_example.py`)

Complete demonstration showing:
- How to use the algorithm
- KV cache usage
- Block-by-block generation
- Visualization creation

## Key Concepts Explained

### The Problem
- Traditional autoregressive models have train-test gap
- Training: sees clean latents
- Inference: generates autoregressively with KV cache
- Gap: distributions don't match!

### The Solution: Self-Forcing
1. **Simulate inference during training**: Generate block-by-block with KV cache
2. **Distribution matching**: Match distributions instead of pixels
3. **KV cache consistency**: Model learns to work with cached states

### Algorithm Flow
```
1. Initialize KV cache
2. For each block:
   a. Denoise block (using KV cache)
   b. Store denoised frames
   c. Update KV cache
   d. Move to next block
3. Compute distribution matching loss
4. Backpropagate through entire process
```

## Usage Examples

### Understand the Algorithm
```python
from tutorial.algorithm import explain_self_forcing
print(explain_self_forcing())
```

### Use the Pipeline
```python
from tutorial.algorithm import SimplifiedSelfForcingPipeline

pipeline = SimplifiedSelfForcingPipeline(
    generator=your_model,
    scheduler=your_scheduler,
    num_frames_per_block=3,
    denoising_steps=[1000, 750, 500, 250]
)

generated_video = pipeline.simulate_inference(noise, conditional_dict)
```

### Visualize Algorithm
```python
from tutorial.algorithm import create_algorithm_diagram, visualize_kv_cache_growth

create_algorithm_diagram("outputs/diagram.png")
visualize_kv_cache_growth(cache_sizes, "outputs/cache_growth.png")
```

### Run Example
```bash
python tutorial/examples/algorithm_example.py
```

## Educational Value

1. **Clear Explanation**: Step-by-step explanation of the algorithm
2. **Simplified Implementation**: Easy to understand code
3. **Visualizations**: See how the algorithm works
4. **Examples**: Ready-to-run demonstrations

## Integration Points

The algorithm tutorial connects to:
- **Toy Dataset**: Use synthetic data to test algorithm
- **Evaluation**: Evaluate generated videos
- **Visualization**: Visualize algorithm behavior
- **Original Codebase**: Understand full implementation

## Files Created

```
tutorial/
├── algorithm/
│   ├── __init__.py
│   ├── self_forcing_algorithm.py    # Core algorithm
│   ├── visualization.py             # Visualization tools
│   └── README.md                    # Algorithm docs
├── examples/
│   └── algorithm_example.py         # Demonstration script
└── README.md                         # Updated with algorithm section
```

## Next Steps

1. **Read the explanation**: `explain_self_forcing()`
2. **Run the example**: `python tutorial/examples/algorithm_example.py`
3. **Experiment**: Try with toy dataset
4. **Visualize**: Create your own visualizations
5. **Scale up**: Move to full implementation in `original_impl/`

## Key Differences from Full Implementation

| Tutorial Version | Full Implementation |
|----------------|-------------------|
| Simplified KV cache | Complex attention caching |
| Basic loss (MSE) | Distribution matching (DMD/SiD/CausVid) |
| Standard model interface | Wan-specific details |
| Single GPU | Distributed training |

The tutorial version focuses on **understanding** the algorithm, while the full implementation has all optimizations and features.
