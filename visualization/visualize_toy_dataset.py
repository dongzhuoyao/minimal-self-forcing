"""
Visualize Toy Dataset Videos

This script demonstrates how to visualize videos from the toy dataset.
"""


import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
# Also add original_impl if needed
sys.path.insert(0, str(root_dir / "original_impl"))

from toy_dataset import ToyDataset
from visualization import (
    save_video_grid,
    create_video_gif,
    display_video,
    save_video_frames,
    tensor_to_numpy
)
import matplotlib.pyplot as plt


def visualize_toy_dataset(
    num_samples: int = 6,
    output_dir: str = "outputs/toy_dataset_visualization"
):
    """
    Visualize toy dataset videos.
    
    Args:
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print(f"Creating toy dataset with {num_samples} samples...")
    dataset = ToyDataset(
        num_samples=num_samples,
        width=64,
        height=64,
        num_frames=9
    )
    
    # Collect videos and prompts
    videos = []
    prompts = []
    
    print(f"\nLoading {num_samples} samples...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        # Convert from (T, C, H, W) to (B, T, C, H, W) for visualization functions
        video = sample["video"].unsqueeze(0)  # Add batch dimension
        videos.append(video)
        prompts.append(sample["prompt"])
        print(f"  Sample {i}: {sample['prompt']}")
    
    # 1. Save video grid (middle frames)
    print(f"\n1. Saving video grid...")
    grid_path = f"{output_dir}/video_grid.png"
    save_video_grid(videos, grid_path, prompts=prompts, ncols=3)
    
    # 2. Create GIFs for each video
    print(f"\n2. Creating GIFs...")
    for i, (video, prompt) in enumerate(zip(videos, prompts)):
        gif_path = f"{output_dir}/video_{i:03d}.gif"
        create_video_gif(video, gif_path, fps=1)
        print(f"  Saved GIF: {gif_path} ({prompt[:50]}...)")
    
    # 3. Save individual frames for first video
    print(f"\n3. Saving individual frames for first video...")
    frames_dir = f"{output_dir}/frames_sample_0"
    save_video_frames(videos[0], frames_dir)
    
    # 4. Display first video (if in interactive environment)
    print(f"\n4. Displaying first video...")
    print(f"   Prompt: {prompts[0]}")
    try:
        display_video(videos[0], title=prompts[0])
    except Exception as e:
        print(f"   Note: Interactive display not available ({e})")
        print(f"   Check the saved GIFs and frames instead!")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    return videos, prompts


def visualize_single_video(
    dataset_idx: int = 0,
    output_path: str = "tutorial/outputs/toy_dataset_visualization/single_video.gif"
):
    """
    Visualize a single video from the dataset.
    
    Args:
        dataset_idx: Index of the video in the dataset
        output_path: Path to save the GIF
    """
    # Create dataset
    dataset = ToyDataset(num_samples=10, width=64, height=64, num_frames=9)
    
    # Get sample
    sample = dataset[dataset_idx]
    video = sample["video"].unsqueeze(0)  # Add batch dimension
    prompt = sample["prompt"]
    
    print(f"Visualizing video {dataset_idx}:")
    print(f"  Prompt: {prompt}")
    print(f"  Shape: {video.shape}")
    
    # Create GIF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    create_video_gif(video, output_path, fps=1)
    print(f"  Saved GIF to: {output_path}")
    
    # Display
    try:
        display_video(video, title=prompt)
    except Exception as e:
        print(f"  Note: Interactive display not available ({e})")
    
    return video, prompt


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize toy dataset videos")
    parser.add_argument("--num_samples", type=int, default=6,
                       help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/toy_dataset_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--single", type=int, default=None,
                       help="Visualize a single video by index")
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Visualize single video
        visualize_single_video(
            dataset_idx=args.single,
            output_path=f"{args.output_dir}/single_video_{args.single}.gif"
        )
    else:
        # Visualize multiple videos
        visualize_toy_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
