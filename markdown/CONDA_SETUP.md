# Conda Environment Setup

This guide will help you set up a conda environment for the Self-Forcing Tutorial.

## Quick Start

1. **Create the conda environment:**
   ```bash
   cd tutorial
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate sf
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; import matplotlib; print('✓ All dependencies installed successfully!')"
   ```

## Manual Installation (Alternative)

If you prefer to install manually:

```bash
conda create -n sf python=3.10
conda activate sf
conda install pytorch torchvision numpy matplotlib pillow -c pytorch -c conda-forge
pip install imageio imageio-ffmpeg tqdm
```

## Requirements

The tutorial requires:
- **Python**: 3.9-3.11 (3.10 recommended)
- **PyTorch**: >=2.0.0
- **NumPy**: >=1.20.0
- **Matplotlib**: >=3.5.0
- **Pillow**: >=8.0.0
- **imageio**: >=2.9.0 (for GIF creation)
- **imageio-ffmpeg**: >=0.4.0 (for video processing)
- **tqdm**: >=4.64.0 (for progress bars)

## GPU Support (Optional)

If you have a CUDA-capable GPU and want to use it:

```bash
conda activate sf
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

Or for CUDA 12.1:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Testing the Installation

Run a simple test to verify everything works:

```bash
python -c "from toy_dataset import ToyDataset; dataset = ToyDataset(num_samples=3); print('✓ Installation successful!')"
```

This should import the tutorial modules without errors.

## Troubleshooting

### Import errors
If you get `ModuleNotFoundError: No module named 'tutorial'`, make sure you're running scripts from the repository root:
```bash
cd /path/to/self-forcing
python -c "from toy_dataset import ToyDataset; print('✓ Import successful!')"
```

### CUDA issues
If you have CUDA installation issues, you can use CPU-only PyTorch:
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### Matplotlib backend issues
If matplotlib displays don't work, you may need to set a backend:
```bash
export MPLBACKEND=Agg  # For non-interactive environments
```
