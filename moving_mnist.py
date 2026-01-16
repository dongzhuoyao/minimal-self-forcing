"""
Moving MNIST Dataset Generator for Self-Forcing Tutorial

This module generates Moving MNIST videos - MNIST digits moving around in a video frame.
No external video data is required - everything is generated programmatically.
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import json
import os
import shutil
from pathlib import Path

# Visualization imports (optional, will fail gracefully if not available)
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import imageio
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    animation = None
    imageio = None


# Visualization functions
def create_video_gif(video, output_path, fps=8, loop=0):
    """Create a GIF from a video tensor."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError("Visualization packages (matplotlib, imageio) not available")
    
    if isinstance(video, torch.Tensor):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = np.array(video)
    
    if len(video_np.shape) == 5:
        if video_np.shape[2] == 3 or video_np.shape[2] == 1:
            video_np = video_np[0]
            video_np = np.transpose(video_np, (0, 2, 3, 1))
        else:
            video_np = video_np[0]
    elif len(video_np.shape) == 4:
        if video_np.shape[1] == 3 or video_np.shape[1] == 1:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
    
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    if video_np.shape[-1] == 1:
        video_np = np.repeat(video_np, 3, axis=-1)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, video_np, fps=fps, loop=loop)


def save_video_grid(videos, output_path, prompts=None, ncols=3, figsize=None):
    """Save a grid of video frames as a single image."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError("Visualization packages (matplotlib, imageio) not available")
    
    num_videos = len(videos)
    nrows = (num_videos + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 2, nrows * 2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_videos == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, video in enumerate(videos):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        if isinstance(video, torch.Tensor):
            video_np = video.detach().cpu().numpy()
        else:
            video_np = np.array(video)
        
        if len(video_np.shape) == 5:
            frame = video_np[0, 0]
            if frame.shape[0] == 3 or frame.shape[0] == 1:
                frame = np.transpose(frame, (1, 2, 0))
        elif len(video_np.shape) == 4:
            if video_np.shape[1] == 3 or video_np.shape[1] == 1:
                frame = np.transpose(video_np[0], (1, 2, 0))
            else:
                frame = video_np[0]
        else:
            frame = video_np
        
        if frame.max() > 1.0:
            frame = frame / 255.0
        frame = np.clip(frame, 0, 1)
        
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            ax.imshow(frame.squeeze(), cmap='gray')
        else:
            ax.imshow(frame)
        
        ax.axis('off')
        if prompts and idx < len(prompts):
            title = prompts[idx]
            if len(title) > 40:
                title = title[:37] + "..."
            ax.set_title(title, fontsize=8)
    
    for idx in range(num_videos, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_video_frames(video, output_dir, prefix="frame"):
    """Save individual frames from a video."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError("Visualization packages (matplotlib, imageio) not available")
    
    if isinstance(video, torch.Tensor):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = np.array(video)
    
    if len(video_np.shape) == 5:
        video_np = video_np[0]
        video_np = np.transpose(video_np, (0, 2, 3, 1))
    elif len(video_np.shape) == 4:
        if video_np.shape[1] == 3 or video_np.shape[1] == 1:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
    
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    if video_np.shape[-1] == 1:
        video_np = np.repeat(video_np, 3, axis=-1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(video_np):
        frame_path = output_path / f"{prefix}_{i:03d}.png"
        imageio.imwrite(str(frame_path), frame)


def display_video(video, title="Video", fps=8):
    """Display a video interactively (for Jupyter notebooks or interactive environments)."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError("Visualization packages (matplotlib, imageio) not available")
    
    if isinstance(video, torch.Tensor):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = np.array(video)
    
    if len(video_np.shape) == 5:
        video_np = video_np[0]
        video_np = np.transpose(video_np, (0, 2, 3, 1))
    elif len(video_np.shape) == 4:
        if video_np.shape[1] == 3 or video_np.shape[1] == 1:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
    
    if video_np.max() > 1.0:
        video_np = video_np / 255.0
    video_np = np.clip(video_np, 0, 1)
    
    if video_np.shape[-1] == 1:
        video_np = np.repeat(video_np, 3, axis=-1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.axis('off')
    im = ax.imshow(video_np[0])
    
    def animate(frame_idx):
        im.set_array(video_np[frame_idx])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(video_np), interval=1000/fps, blit=True, repeat=True)
    plt.show()
    return anim


class MovingMNISTGenerator:
    """Generate Moving MNIST videos - digits moving and bouncing around."""
    
    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        num_frames: int = 16,
        fps: int = 8,
        digit_size: int = 64,
        num_digits: int = 1,
        max_velocity: float = 2.0
    ):
        """
        Args:
            width: Video width in pixels
            height: Video height in pixels
            num_frames: Number of frames per video
            fps: Frames per second
            digit_size: Size of MNIST digit (default 64x64)
            num_digits: Number of digits per video (1 or 2)
            max_velocity: Maximum velocity for digit movement
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        # Ensure digit size is smaller than frame to allow movement
        # Use at most 80% of the smaller dimension
        max_digit_size = min(width, height) * 0.8
        self.digit_size = min(int(digit_size), int(max_digit_size))
        if digit_size > max_digit_size:
            print(f"Warning: digit_size {digit_size} is too large for {width}x{height} frame.")
            print(f"         Using {self.digit_size} instead to allow movement.")
        self.num_digits = num_digits
        self.max_velocity = max_velocity
        
        # Load MNIST dataset (only training set, we'll use it for generating videos)
        # Check for existing MNIST files in multiple locations
        data_root = './data'
        mnist_raw_dir = Path(data_root) / 'MNIST' / 'raw'
        mnist_master_dir = Path(data_root) / 'mnist-master'
        
        # Required MNIST files
        required_files_gz = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        
        # Check if files exist in expected location
        files_exist = all((mnist_raw_dir / f).exists() for f in required_files_gz)
        
        # If not found, check if they exist in mnist-master directory
        if not files_exist and mnist_master_dir.exists():
            print(f"Found MNIST files in {mnist_master_dir}")
            print("Copying files to expected location...")
            
            # Create target directory
            mnist_raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files from mnist-master to MNIST/raw
            for filename in required_files_gz:
                src = mnist_master_dir / filename
                dst = mnist_raw_dir / filename
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  Copied {filename}")
            
            # Verify files were copied
            files_exist = all((mnist_raw_dir / f).exists() for f in required_files_gz)
            if files_exist:
                print("✓ MNIST files ready!")
            else:
                print("⚠ Some files may be missing")
        
        if not files_exist:
            print("=" * 70)
            print("MNIST Dataset Download")
            print("=" * 70)
            print("MNIST files not found. Options:")
            print("\nOption 1: Place files in ./data/mnist-master/ (will be auto-detected)")
            print("Option 2: Place files in ./data/MNIST/raw/")
            print("Option 3: Let torchvision download (may be slow)")
            print("=" * 70)
            print("Starting download...")
        
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,  # torchvision will skip if files exist
            transform=transforms.ToTensor()
        )
        print(f"✓ Loaded MNIST dataset with {len(self.mnist_dataset)} digits")
    
    def _get_digit_image(self, digit_label: int) -> np.ndarray:
        """Get a random MNIST digit image for the given label."""
        # Find all images with this label
        indices = [i for i, (_, label) in enumerate(self.mnist_dataset) if label == digit_label]
        if not indices:
            # Fallback: use any digit
            idx = np.random.randint(len(self.mnist_dataset))
        else:
            idx = np.random.choice(indices)
        
        # Get digit image (already normalized to [0, 1])
        digit_tensor, _ = self.mnist_dataset[idx]
        # Convert to numpy and scale to [0, 255]
        digit_img = (digit_tensor.squeeze().numpy() * 255).astype(np.uint8)
        return digit_img
    
    def generate_moving_digit(
        self,
        digit_label: int,
        start_x: Optional[float] = None,
        start_y: Optional[float] = None,
        velocity_x: Optional[float] = None,
        velocity_y: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate a video of a single MNIST digit moving and bouncing around.
        
        Args:
            digit_label: MNIST digit label (0-9)
            start_x: Starting x position (None = random)
            start_y: Starting y position (None = random)
            velocity_x: Initial x velocity (None = random)
            velocity_y: Initial y velocity (None = random)
        
        Returns:
            video: numpy array of shape (num_frames, height, width, 3)
            label: digit label (0-9)
        """
        video = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        
        # Get digit image
        digit_img = self._get_digit_image(digit_label)
        
        # Initialize position and velocity
        if start_x is None:
            start_x = np.random.uniform(self.digit_size // 2, self.width - self.digit_size // 2)
        if start_y is None:
            start_y = np.random.uniform(self.digit_size // 2, self.height - self.digit_size // 2)
        
        if velocity_x is None:
            # Ensure velocity is never zero - use at least 0.5 pixels per frame
            velocity_x = np.random.uniform(-self.max_velocity, self.max_velocity)
            if abs(velocity_x) < 0.5:
                velocity_x = 0.5 if velocity_x >= 0 else -0.5
        if velocity_y is None:
            velocity_y = np.random.uniform(-self.max_velocity, self.max_velocity)
            if abs(velocity_y) < 0.5:
                velocity_y = 0.5 if velocity_y >= 0 else -0.5
        
        x, y = start_x, start_y
        vx, vy = velocity_x, velocity_y
        
        # Generate frames
        for frame_idx in range(self.num_frames):
            # Update position
            x += vx
            y += vy
            
            # Bounce off walls
            if x <= self.digit_size // 2 or x >= self.width - self.digit_size // 2:
                vx = -vx
                x = np.clip(x, self.digit_size // 2, self.width - self.digit_size // 2)
            if y <= self.digit_size // 2 or y >= self.height - self.digit_size // 2:
                vy = -vy
                y = np.clip(y, self.digit_size // 2, self.height - self.digit_size // 2)
            
            # Create frame
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            
            # Resize digit to desired size
            digit_pil = Image.fromarray(digit_img, mode='L')
            digit_pil = digit_pil.resize((self.digit_size, self.digit_size), Image.Resampling.LANCZOS)
            
            # Paste digit at current position
            x_pos = int(x - self.digit_size // 2)
            y_pos = int(y - self.digit_size // 2)
            img.paste(digit_pil, (x_pos, y_pos))
            
            # Convert to RGB (grayscale to 3-channel)
            video[frame_idx] = np.array(img.convert('RGB'))
        
        return video, digit_label
    
    def generate_two_digits(
        self,
        digit1_label: int,
        digit2_label: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Generate a video with two MNIST digits moving independently.
        
        Args:
            digit1_label: First digit label (0-9)
            digit2_label: Second digit label (0-9)
        
        Returns:
            video: numpy array of shape (num_frames, height, width, 3)
            labels: tuple of (digit1_label, digit2_label)
        """
        video = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        
        # Get digit images
        digit1_img = self._get_digit_image(digit1_label)
        digit2_img = self._get_digit_image(digit2_label)
        
        # Initialize positions and velocities for both digits
        x1 = np.random.uniform(self.digit_size // 2, self.width - self.digit_size // 2)
        y1 = np.random.uniform(self.digit_size // 2, self.height - self.digit_size // 2)
        vx1 = np.random.uniform(-self.max_velocity, self.max_velocity)
        if abs(vx1) < 0.5:
            vx1 = 0.5 if vx1 >= 0 else -0.5
        vy1 = np.random.uniform(-self.max_velocity, self.max_velocity)
        if abs(vy1) < 0.5:
            vy1 = 0.5 if vy1 >= 0 else -0.5
        
        x2 = np.random.uniform(self.digit_size // 2, self.width - self.digit_size // 2)
        y2 = np.random.uniform(self.digit_size // 2, self.height - self.digit_size // 2)
        vx2 = np.random.uniform(-self.max_velocity, self.max_velocity)
        if abs(vx2) < 0.5:
            vx2 = 0.5 if vx2 >= 0 else -0.5
        vy2 = np.random.uniform(-self.max_velocity, self.max_velocity)
        if abs(vy2) < 0.5:
            vy2 = 0.5 if vy2 >= 0 else -0.5
        
        # Generate frames
        for frame_idx in range(self.num_frames):
            # Update positions
            x1 += vx1
            y1 += vy1
            x2 += vx2
            y2 += vy2
            
            # Bounce off walls for digit 1
            if x1 <= self.digit_size // 2 or x1 >= self.width - self.digit_size // 2:
                vx1 = -vx1
                x1 = np.clip(x1, self.digit_size // 2, self.width - self.digit_size // 2)
            if y1 <= self.digit_size // 2 or y1 >= self.height - self.digit_size // 2:
                vy1 = -vy1
                y1 = np.clip(y1, self.digit_size // 2, self.height - self.digit_size // 2)
            
            # Bounce off walls for digit 2
            if x2 <= self.digit_size // 2 or x2 >= self.width - self.digit_size // 2:
                vx2 = -vx2
                x2 = np.clip(x2, self.digit_size // 2, self.width - self.digit_size // 2)
            if y2 <= self.digit_size // 2 or y2 >= self.height - self.digit_size // 2:
                vy2 = -vy2
                y2 = np.clip(y2, self.digit_size // 2, self.height - self.digit_size // 2)
            
            # Create frame
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            
            # Resize and paste digits
            digit1_pil = Image.fromarray(digit1_img, mode='L').resize(
                (self.digit_size, self.digit_size), Image.Resampling.LANCZOS
            )
            digit2_pil = Image.fromarray(digit2_img, mode='L').resize(
                (self.digit_size, self.digit_size), Image.Resampling.LANCZOS
            )
            
            x1_pos = int(x1 - self.digit_size // 2)
            y1_pos = int(y1 - self.digit_size // 2)
            x2_pos = int(x2 - self.digit_size // 2)
            y2_pos = int(y2 - self.digit_size // 2)
            
            img.paste(digit1_pil, (x1_pos, y1_pos))
            img.paste(digit2_pil, (x2_pos, y2_pos))
            
            video[frame_idx] = np.array(img.convert('RGB'))
        
        return video, (digit1_label, digit2_label)


class MovingMNISTDataset:
    """Dataset class for Moving MNIST videos."""
    
    def __init__(
        self,
        num_samples: int = 100,
        width: int = 64,
        height: int = 64,
        num_frames: int = 16,
        seed: int = 42,
        num_digits: int = 1,
        digit_size: int = 28,
        max_velocity: float = 2.0
    ):
        """
        Args:
            num_samples: Number of video samples to generate
            width: Video width
            height: Video height
            num_frames: Number of frames per video
            seed: Random seed for reproducibility
            num_digits: Number of digits per video (1 or 2)
            digit_size: Size of MNIST digit
            max_velocity: Maximum velocity for digit movement
        """
        self.num_samples = num_samples
        self.generator = MovingMNISTGenerator(
            width=width,
            height=height,
            num_frames=num_frames,
            num_digits=num_digits,
            digit_size=digit_size,
            max_velocity=max_velocity
        )
        self.videos = []
        self.labels = []
        
        np.random.seed(seed)
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate all video samples."""
        print(f"Generating {self.num_samples} Moving MNIST videos...")
        
        for i in range(self.num_samples):
            if self.generator.num_digits == 1:
                # Single digit - generate videos for digits 0-9 consecutively
                digit_label = i % 10
                video, label = self.generator.generate_moving_digit(digit_label)
            else:
                # Two digits
                digit1_label = np.random.randint(0, 10)
                digit2_label = np.random.randint(0, 10)
                video, labels = self.generator.generate_two_digits(digit1_label, digit2_label)
                label = labels  # Store tuple for two digits
            
            self.videos.append(video)
            self.labels.append(label)
        
        print(f"Generated {len(self.videos)} videos")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        """Get a video sample."""
        video = self.videos[idx]
        label = self.labels[idx]
        
        # Convert to torch tensor and normalize to [-1, 1]
        video_tensor = torch.from_numpy(video).float()
        video_tensor = video_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        return {
            "video": video_tensor,
            "label": label,
            "idx": idx
        }
    
    def save_labels(self, output_path: str):
        """Save labels to a text file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for label in self.labels:
                if isinstance(label, tuple):
                    f.write(f"{label[0]},{label[1]}\n")
                else:
                    f.write(f"{label}\n")
        print(f"Saved labels to {output_path}")
    
    def save_metadata(self, output_path: str):
        """Save dataset metadata to JSON."""
        metadata = {
            "num_samples": self.num_samples,
            "width": self.generator.width,
            "height": self.generator.height,
            "num_frames": self.generator.num_frames,
            "fps": self.generator.fps,
            "num_digits": self.generator.num_digits,
            "digit_size": self.generator.digit_size,
            "labels": [list(label) if isinstance(label, tuple) else label for label in self.labels]
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {output_path}")
    
    def visualize(
        self,
        num_samples: Optional[int] = None,
        output_dir: str = "outputs/moving_mnist_visualization",
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
            Tuple of (videos_list, labels_list)
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
        
        # Collect videos and labels
        videos = []
        labels = []
        
        for i in range(num_samples):
            sample = self[i]
            # Convert from (T, C, H, W) to (B, T, C, H, W) for visualization functions
            video = sample["video"].unsqueeze(0)  # Add batch dimension
            
            # Convert from [-1, 1] to [0, 1] for visualization
            video_normalized = (video + 1.0) / 2.0
            video_normalized = video_normalized.clamp(0, 1)
            
            videos.append(video_normalized)
            label = sample["label"]
            labels.append(label)
            label_str = f"Digit {label}" if isinstance(label, int) else f"Digits {label[0]},{label[1]}"
            print(f"  Sample {i}: {label_str}")
        
        # Create label strings for visualization
        label_strings = [f"Digit {l}" if isinstance(l, int) else f"Digits {l[0]},{l[1]}" for l in labels]
        
        # Save video grid
        if save_grid:
            print(f"\n1. Saving video grid...")
            grid_path = f"{output_dir}/video_grid.png"
            save_video_grid(videos, grid_path, prompts=label_strings, ncols=3)
            print(f"   Saved: {grid_path}")
        
        # Create GIFs for each video
        if save_gifs:
            print(f"\n2. Creating GIFs...")
            for i, (video, label) in enumerate(zip(videos, labels)):
                gif_path = f"{output_dir}/video_{i:03d}.gif"
                create_video_gif(video, gif_path, fps=2)
                label_str = f"Digit {label}" if isinstance(label, int) else f"Digits {label[0]},{label[1]}"
                print(f"   Saved GIF: {gif_path} ({label_str})")
        
        # Save individual frames for first video
        if save_frames and len(videos) > 0:
            print(f"\n3. Saving individual frames for first video...")
            frames_dir = f"{output_dir}/frames_sample_0"
            save_video_frames(videos[0], frames_dir)
            print(f"   Saved frames to: {frames_dir}")
        
        # Display first video (if in interactive environment)
        if display and len(videos) > 0:
            print(f"\n4. Displaying first video...")
            label_str = f"Digit {labels[0]}" if isinstance(labels[0], int) else f"Digits {labels[0][0]},{labels[0][1]}"
            print(f"   Label: {label_str}")
            try:
                display_video(videos[0], title=label_str)
            except Exception as e:
                print(f"   Note: Interactive display not available ({e})")
                print(f"   Check the saved GIFs and frames instead!")
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
        return videos, labels
    
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
            Tuple of (video_tensor, label)
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
        label = sample["label"]
        
        # Convert from [-1, 1] to [0, 1] for visualization
        video_normalized = (video + 1.0) / 2.0
        video_normalized = video_normalized.clamp(0, 1)
        
        print(f"\nVisualizing video {idx}:")
        label_str = f"Digit {label}" if isinstance(label, int) else f"Digits {label[0]},{label[1]}"
        print(f"  Label: {label_str}")
        print(f"  Shape: {video_normalized.shape}")
        
        # Create GIF if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            create_video_gif(video_normalized, output_path, fps=2)
            print(f"  Saved GIF to: {output_path}")
        
        # Display
        if display:
            try:
                display_video(video_normalized, title=label_str)
            except Exception as e:
                print(f"  Note: Interactive display not available ({e})")
        
        return video_normalized, label


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Moving MNIST Dataset Generator and Visualizer")
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
    parser.add_argument("--num_digits", type=int, default=1, choices=[1, 2],
                       help="Number of digits per video (1 or 2)")
    parser.add_argument("--digit_size", type=int, default=64,
                       help="Size of MNIST digit")
    parser.add_argument("--max_velocity", type=float, default=2.0,
                       help="Maximum velocity for digit movement")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize videos")
    parser.add_argument("--num_viz", type=int, default=6,
                       help="Number of videos to visualize")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/moving_mnist_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--single", type=int, default=None,
                       help="Visualize a single video by index")
    parser.add_argument("--save_labels", type=str, default=None,
                       help="Path to save labels file")
    parser.add_argument("--save_metadata", type=str, default=None,
                       help="Path to save metadata JSON file")
    
    args = parser.parse_args()
    
    # Create dataset
    print("=" * 70)
    print("Moving MNIST Dataset Generator")
    print("=" * 70)
    dataset = MovingMNISTDataset(
        num_samples=args.num_samples,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed,
        num_digits=args.num_digits,
        digit_size=args.digit_size,
        max_velocity=args.max_velocity
    )
    
    print(f"\nGenerated {len(dataset)} videos")
    sample = dataset[0]
    label_str = f"Digit {sample['label']}" if isinstance(sample['label'], int) else f"Digits {sample['label'][0]},{sample['label'][1]}"
    print(f"Sample 0 label: {label_str}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Video value range: [{sample['video'].min():.2f}, {sample['video'].max():.2f}]")
    
    # Save labels if requested
    if args.save_labels:
        dataset.save_labels(args.save_labels)
    
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
