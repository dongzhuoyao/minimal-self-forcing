"""
Toy Dataset Generator for Self-Forcing Tutorial

This module generates synthetic video data for quick experimentation and learning.
No external video data is required - everything is generated programmatically.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import json
import os


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
        prompt = f"A {color_name} {shape} moving {direction}ly"
        
        return video, prompt
    
    def generate_rotating_shape(
        self,
        shape: str = "square",
        color: Tuple[int, int, int] = (0, 0, 255),
        rotation_direction: str = "clockwise"
    ) -> Tuple[np.ndarray, str]:
        """Generate a video of a rotating shape."""
        video = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        
        center_x, center_y = self.width // 2, self.height // 2
        angles = np.linspace(0, 360, self.num_frames)
        if rotation_direction == "counterclockwise":
            angles = -angles
        
        for frame_idx in range(self.num_frames):
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            angle = np.radians(angles[frame_idx])
            size = 40
            
            # Create rotated shape
            if shape == "square":
                corners = [
                    (-size, -size), (size, -size),
                    (size, size), (-size, size)
                ]
            else:  # circle
                corners = [(size * np.cos(a), size * np.sin(a)) 
                          for a in np.linspace(0, 2*np.pi, 8)]
            
            # Rotate and translate
            rotated_corners = []
            for cx, cy in corners:
                rx = cx * np.cos(angle) - cy * np.sin(angle) + center_x
                ry = cx * np.sin(angle) + cy * np.cos(angle) + center_y
                rotated_corners.append((rx, ry))
            
            if shape == "square":
                draw.polygon(rotated_corners, fill=color)
            else:
                draw.ellipse(
                    [center_x - size, center_y - size,
                     center_x + size, center_y + size],
                    fill=color
                )
            
            video[frame_idx] = np.array(img)
        
        color_name = self._color_to_name(color)
        prompt = f"A {color_name} {shape} rotating {rotation_direction}"
        
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
        prompt = f"Color gradient transitioning from {start_name} to {end_name}"
        
        return video, prompt
    
    def generate_bouncing_ball(
        self,
        color: Tuple[int, int, int] = (255, 255, 0)
    ) -> Tuple[np.ndarray, str]:
        """Generate a video of a bouncing ball."""
        video = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        
        radius = 20
        ground_y = self.height - 30
        
        for frame_idx in range(self.num_frames):
            t = frame_idx / (self.num_frames - 1)
            # Simple bouncing motion
            x = int(self.width * t)
            # Parabolic bounce
            bounce_height = 100 * np.sin(np.pi * t)
            y = int(ground_y - bounce_height - radius)
            
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw ground
            draw.line([(0, ground_y), (self.width, ground_y)], fill=(100, 100, 100), width=2)
            
            # Draw ball
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color
            )
            
            video[frame_idx] = np.array(img)
        
        color_name = self._color_to_name(color)
        prompt = f"A {color_name} ball bouncing"
        
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
                "rotating_shape",
                "color_transition",
                "bouncing_ball"
            ])
            
            if anim_type == "moving_shape":
                shape = np.random.choice(["circle", "square", "triangle"])
                color = self._random_color()
                direction = np.random.choice(["horizontal", "vertical", "diagonal"])
                video, prompt = self.generator.generate_moving_shape(
                    shape, color, direction
                )
            
            elif anim_type == "rotating_shape":
                shape = np.random.choice(["circle", "square"])
                color = self._random_color()
                direction = np.random.choice(["clockwise", "counterclockwise"])
                video, prompt = self.generator.generate_rotating_shape(
                    shape, color, direction
                )
            
            elif anim_type == "color_transition":
                start_color = self._random_color()
                end_color = self._random_color()
                video, prompt = self.generator.generate_color_transition(
                    start_color, end_color
                )
            
            else:  # bouncing_ball
                color = self._random_color()
                video, prompt = self.generator.generate_bouncing_ball(color)
            
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


if __name__ == "__main__":
    # Example usage
    dataset = ToyDataset(num_samples=10, width=256, height=256, num_frames=16)
    
    # Save prompts
    dataset.save_prompts("tutorial/data/prompts/toy_prompts.txt")
    dataset.save_metadata("tutorial/data/toy_metadata.json")
    
    # Display first sample
    sample = dataset[0]
    print(f"Sample 0 prompt: {sample['prompt']}")
    print(f"Video shape: {sample['video'].shape}")
