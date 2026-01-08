"""
Example Inference Script for Tutorial

This script demonstrates how to use the tutorial codebase for inference.
"""

import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_impl"))

from tutorial.data import ToyDataset
from tutorial.visualization import create_video_gif, save_video_grid
from tutorial.evaluation import compute_all_metrics, CLIPScoreMetric, FrameConsistencyMetric


def simple_inference_example():
    """Simple example of generating videos from prompts."""
    print("=" * 60)
    print("Simple Inference Example")
    print("=" * 60)
    
    # Create toy dataset
    print("\n1. Creating toy dataset...")
    dataset = ToyDataset(num_samples=5, width=256, height=256, num_frames=16)
    
    # Simulate inference (in real usage, you would use the actual model)
    print("\n2. Simulating video generation...")
    print("Note: This is a placeholder. In real usage, you would:")
    print("  - Load your trained model")
    print("  - Encode text prompts")
    print("  - Generate videos using the pipeline")
    
    # For demonstration, we'll use the ground truth videos
    generated_videos = []
    prompts = []
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        generated_videos.append(sample["video"])
        prompts.append(sample["prompt"])
        print(f"  Generated video {i+1}: {sample['prompt']}")
    
    # Visualize results
    print("\n3. Visualizing results...")
    output_dir = Path("tutorial/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video grid
    save_video_grid(
        generated_videos,
        str(output_dir / "video_grid.png"),
        prompts=prompts
    )
    
    # Save individual GIFs
    for i, (video, prompt) in enumerate(zip(generated_videos, prompts)):
        gif_path = output_dir / f"video_{i:02d}.gif"
        create_video_gif(video, str(gif_path), fps=8)
        print(f"  Saved GIF: {gif_path}")
    
    # Evaluate
    print("\n4. Evaluating videos...")
    results = compute_all_metrics(generated_videos, prompts)
    
    print("\nEvaluation Results:")
    print("-" * 40)
    for metric_name, value in results.items():
        if value is not None:
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)


def evaluation_example():
    """Example of evaluating generated videos."""
    print("\n" + "=" * 60)
    print("Evaluation Example")
    print("=" * 60)
    
    # Create dataset
    dataset = ToyDataset(num_samples=10, width=256, height=256, num_frames=16)
    
    # Get videos and prompts
    videos = [dataset[i]["video"] for i in range(len(dataset))]
    prompts = [dataset[i]["prompt"] for i in range(len(dataset))]
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_all_metrics(videos, prompts)
    
    print("\nEvaluation Results:")
    print("-" * 40)
    for metric_name, value in results.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    # Individual metric examples
    print("\n" + "-" * 40)
    print("Individual Metric Examples:")
    print("-" * 40)
    
    # Frame consistency
    consistency_metric = FrameConsistencyMetric()
    consistency_score = consistency_metric.compute(videos[0])
    print(f"\nFrame Consistency (first video): {consistency_score:.4f}")
    
    # CLIP score
    clip_metric = CLIPScoreMetric()
    if clip_metric.available:
        clip_score = clip_metric.compute(videos[0], prompts[0])
        print(f"CLIP Score (first video): {clip_score:.4f}")
    else:
        print("\nCLIP Score: Not available (install CLIP to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["inference", "evaluation"],
        help="Example mode to run"
    )
    args = parser.parse_args()
    
    if args.mode == "inference":
        simple_inference_example()
    else:
        evaluation_example()
