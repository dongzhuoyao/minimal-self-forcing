"""
Toy Dataset Generator for Self-Forcing Tutorial

This module generates synthetic video data for quick experimentation and learning.
No external video data is required - everything is generated programmatically.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional
import json
import os
from pathlib import Path

# Import visualization functions (optional, will fail gracefully if not available)
try:
    from visualization import (
        save_video_grid,
        create_video_gif,
        display_video,
        save_video_frames
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ToyVideoGenerator:
    """Generate simple synthetic videos for tutorial purposes."""
    
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        num_frames: int = 16,
        fps: int = 8
    ):
        """
        Args:
            width: Video width in pixels
            height: Video height in pixels
            num_frames: Number of frames per video
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
    
    def generate_moving_shape(
        self,
        shape: str = "circle",
        color: Tuple[int, int, int] = (255, 0, 0),
        direction: str = "horizontal"
    ) -> Tuple[np.ndarray, str]:
        """
        Generate a video of a shape moving across the screen.
        
        Args:
            shape: "circle", "square", or "triangle"
            color: RGB color tuple
            direction: "horizontal", "vertical", or "diagonal"
        
        Returns:
            video: numpy array of shape (num_frames, height, width, 3)
            prompt: text description of the video
        """
        video = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        
        # Generate movement trajectory
        if direction == "horizontal":
            x_positions = np.linspace(50, self.width - 50, self.num_frames)
            y_positions = np.full(self.num_frames, self.height // 2)
        elif direction == "vertical":
            x_positions = np.full(self.num_frames, self.width // 2)
            y_positions = np.linspace(50, self.height - 50, self.num_frames)
        else:  # diagonal
            x_positions = np.linspace(50, self.width - 50, self.num_frames)
            y_positions = np.linspace(50, self.height - 50, self.num_frames)
        
        # Draw frames
        for frame_idx in range(self.num_frames):
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            x, y = int(x_positions[frame_idx]), int(y_positions[frame_idx])
            size = 30
            
            if shape == "circle":
                draw.ellipse(
                    [x - size, y - size, x + size, y + size],
                    fill=color
                )
            elif shape == "square":
                draw.rectangle(
                    [x - size, y - size, x + size, y + size],
                    fill=color
                )
            else:  # triangle
                points = [
                    (x, y - size),
                    (x - size, y + size),
                    (x + size, y + size)
                ]
                draw.polygon(points, fill=color)
            
            video[frame_idx] = np.array(img)
        
        color_name = self._color_to_name(color)
        # Convert direction to adverb form
        direction_word = "horizontally" if direction == "horizontal" else "vertically" if direction == "vertical" else "diagonally"
        prompt = f"A {color_name} {shape} moving {direction_word}"
        
        return video, prompt
    
    def generate_color_transition(
        self,
        start_color: Tuple[int, int, int] = (255, 0, 0),
        end_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> Tuple[np.ndarray, str]:
        """Generate a video with color gradient transition."""
        video = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        
        for frame_idx in range(self.num_frames):
            t = frame_idx / (self.num_frames - 1)
            r = int(start_color[0] * (1 - t) + end_color[0] * t)
            g = int(start_color[1] * (1 - t) + end_color[1] * t)
            b = int(start_color[2] * (1 - t) + end_color[2] * t)
            
            color = (r, g, b)
            img = Image.new('RGB', (self.width, self.height), color=color)
            video[frame_idx] = np.array(img)
        
        start_name = self._color_to_name(start_color)
        end_name = self._color_to_name(end_color)
        prompt = f"A color gradient transitioning from {start_name} to {end_name}"
        
        return video, prompt
    
    def _color_to_name(self, color: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to color name."""
        color_map = {
            (255, 0, 0): "red",
            (0, 255, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
            (255, 0, 255): "magenta",
            (0, 255, 255): "cyan",
            (255, 255, 255): "white",
            (0, 0, 0): "black",
        }
        return color_map.get(color, f"rgb{color}")


class ToyDataset:
    """Dataset class for toy synthetic videos."""
    
    def __init__(
        self,
        num_samples: int = 100,
        width: int = 256,
        height: int = 256,
        num_frames: int = 16,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of video samples to generate
            width: Video width
            height: Video height
            num_frames: Number of frames per video
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.generator = ToyVideoGenerator(width, height, num_frames)
        self.videos = []
        self.prompts = []
        
        np.random.seed(seed)
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate all video samples."""
        print(f"Generating {self.num_samples} toy videos...")
        
        for i in range(self.num_samples):
            # Randomly select animation type
            anim_type = np.random.choice([
                "moving_shape",
                "color_transition"
            ])
            
            if anim_type == "moving_shape":
                shape = np.random.choice(["circle", "square", "triangle"])
                color = self._random_color()
                direction = np.random.choice(["horizontal", "vertical", "diagonal"])
                video, prompt = self.generator.generate_moving_shape(
                    shape, color, direction
                )
            
            else:  # color_transition
                start_color = self._random_color()
                end_color = self._random_color()
                video, prompt = self.generator.generate_color_transition(
                    start_color, end_color
                )
            
            self.videos.append(video)
            self.prompts.append(prompt)
        
        print(f"Generated {len(self.videos)} videos")
    
    def _random_color(self) -> Tuple[int, int, int]:
        """Generate a random RGB color."""
        colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0),  # yellow
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
        ]
        return colors[np.random.randint(len(colors))]
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        """Get a video sample."""
        video = self.videos[idx]
        prompt = self.prompts[idx]
        
        # Convert to torch tensor and normalize to [-1, 1]
        video_tensor = torch.from_numpy(video).float()
        video_tensor = video_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        return {
            "video": video_tensor,
            "prompt": prompt,
            "idx": idx
        }
    
    def save_prompts(self, output_path: str):
        """Save prompts to a text file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for prompt in self.prompts:
                f.write(prompt + '\n')
        print(f"Saved prompts to {output_path}")
    
    def save_metadata(self, output_path: str):
        """Save dataset metadata to JSON."""
        metadata = {
            "num_samples": self.num_samples,
            "width": self.generator.width,
            "height": self.generator.height,
            "num_frames": self.generator.num_frames,
            "fps": self.generator.fps,
            "prompts": self.prompts
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {output_path}")
    
    def visualize(
        self,
        num_samples: Optional[int] = None,
        output_dir: str = "outputs/toy_dataset_visualization",
        save_gifs: bool = True,
        save_grid: bool = True,
        save_frames: bool = True,
        display: bool = False
    ):
        """
        Visualize videos from the dataset.
        
        Args:
            num_samples: Number of samples to visualize (None = all)
            output_dir: Directory to save visualizations
            save_gifs: Whether to save individual GIFs
            save_grid: Whether to save video grid
            save_frames: Whether to save individual frames for first video
            display: Whether to display videos interactively
        
        Returns:
            Tuple of (videos_list, prompts_list)
        """
        if not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization functions not available. Install required packages.")
            return [], []
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine number of samples
        if num_samples is None:
            num_samples = len(self)
        num_samples = min(num_samples, len(self))
        
        print(f"\nVisualizing {num_samples} videos from dataset...")
        
        # Collect videos and prompts
        videos = []
        prompts = []
        
        for i in range(num_samples):
            sample = self[i]
            # Convert from (T, C, H, W) to (B, T, C, H, W) for visualization functions
            video = sample["video"].unsqueeze(0)  # Add batch dimension
            
            # Convert from [-1, 1] to [0, 1] for visualization
            video_normalized = (video + 1.0) / 2.0
            video_normalized = video_normalized.clamp(0, 1)
            
            videos.append(video_normalized)
            prompts.append(sample["prompt"])
            print(f"  Sample {i}: {sample['prompt']}")
        
        # Save video grid
        if save_grid:
            print(f"\n1. Saving video grid...")
            grid_path = f"{output_dir}/video_grid.png"
            save_video_grid(videos, grid_path, prompts=prompts, ncols=3)
            print(f"   Saved: {grid_path}")
        
        # Create GIFs for each video
        if save_gifs:
            print(f"\n2. Creating GIFs...")
            for i, (video, prompt) in enumerate(zip(videos, prompts)):
                gif_path = f"{output_dir}/video_{i:03d}.gif"
                create_video_gif(video, gif_path, fps=2)
                print(f"   Saved GIF: {gif_path} ({prompt[:50]}...)")
        
        # Save individual frames for first video
        if save_frames and len(videos) > 0:
            print(f"\n3. Saving individual frames for first video...")
            frames_dir = f"{output_dir}/frames_sample_0"
            save_video_frames(videos[0], frames_dir)
            print(f"   Saved frames to: {frames_dir}")
        
        # Display first video (if in interactive environment)
        if display and len(videos) > 0:
            print(f"\n4. Displaying first video...")
            print(f"   Prompt: {prompts[0]}")
            try:
                display_video(videos[0], title=prompts[0])
            except Exception as e:
                print(f"   Note: Interactive display not available ({e})")
                print(f"   Check the saved GIFs and frames instead!")
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
        return videos, prompts
    
    def visualize_single(
        self,
        idx: int = 0,
        output_path: Optional[str] = None,
        display: bool = False
    ):
        """
        Visualize a single video from the dataset.
        
        Args:
            idx: Index of the video in the dataset
            output_path: Path to save the GIF (None = auto-generate)
            display: Whether to display video interactively
        
        Returns:
            Tuple of (video_tensor, prompt)
        """
        if not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization functions not available. Install required packages.")
            return None, None
        
        if idx >= len(self):
            print(f"Error: Index {idx} out of range (dataset has {len(self)} samples)")
            return None, None
        
        # Get sample
        sample = self[idx]
        video = sample["video"].unsqueeze(0)  # Add batch dimension
        prompt = sample["prompt"]
        
        # Convert from [-1, 1] to [0, 1] for visualization
        video_normalized = (video + 1.0) / 2.0
        video_normalized = video_normalized.clamp(0, 1)
        
        print(f"\nVisualizing video {idx}:")
        print(f"  Prompt: {prompt}")
        print(f"  Shape: {video_normalized.shape}")
        
        # Create GIF if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            create_video_gif(video_normalized, output_path, fps=2)
            print(f"  Saved GIF to: {output_path}")
        
        # Display
        if display:
            try:
                display_video(video_normalized, title=prompt)
            except Exception as e:
                print(f"  Note: Interactive display not available ({e})")
        
        return video_normalized, prompt


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Toy Dataset Generator and Visualizer")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to generate")
    parser.add_argument("--width", type=int, default=64,
                       help="Video width")
    parser.add_argument("--height", type=int, default=64,
                       help="Video height")
    parser.add_argument("--num_frames", type=int, default=9,
                       help="Number of frames per video")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize videos")
    parser.add_argument("--num_viz", type=int, default=6,
                       help="Number of videos to visualize")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/toy_dataset_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--single", type=int, default=None,
                       help="Visualize a single video by index")
    parser.add_argument("--save_prompts", type=str, default=None,
                       help="Path to save prompts file")
    parser.add_argument("--save_metadata", type=str, default=None,
                       help="Path to save metadata JSON file")
    
    args = parser.parse_args()
    
    # Create dataset
    print("=" * 70)
    print("Toy Dataset Generator")
    print("=" * 70)
    dataset = ToyDataset(
        num_samples=args.num_samples,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed
    )
    
    print(f"\nGenerated {len(dataset)} videos")
    sample = dataset[0]
    print(f"Sample 0 prompt: {sample['prompt']}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Video value range: [{sample['video'].min():.2f}, {sample['video'].max():.2f}]")
    
    # Save prompts if requested
    if args.save_prompts:
        dataset.save_prompts(args.save_prompts)
    
    # Save metadata if requested
    if args.save_metadata:
        dataset.save_metadata(args.save_metadata)
    
    # Visualize if requested
    if args.visualize:
        if args.single is not None:
            # Visualize single video
            output_path = f"{args.output_dir}/single_video_{args.single}.gif"
            dataset.visualize_single(
                idx=args.single,
                output_path=output_path,
                display=False
            )
        else:
            # Visualize multiple videos
            dataset.visualize(
                num_samples=args.num_viz,
                output_dir=args.output_dir,
                save_gifs=True,
                save_grid=True,
                save_frames=True,
                display=False
            )
