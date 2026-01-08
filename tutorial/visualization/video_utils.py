"""
Video Visualization Utilities

Tools for visualizing and comparing generated videos.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple
from PIL import Image
import os


def tensor_to_numpy(video: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor to numpy array.
    
    Args:
        video: Tensor of shape (T, C, H, W) or (B, T, C, H, W), values in [-1, 1]
    
    Returns:
        numpy array of shape (T, H, W, C), values in [0, 255]
    """
    if len(video.shape) == 5:
        video = video[0]  # Take first batch
    
    # Normalize from [-1, 1] to [0, 1]
    video_norm = (video + 1.0) / 2.0
    video_norm = torch.clamp(video_norm, 0, 1)
    
    # Convert to numpy and change format
    video_np = video_norm.permute(0, 2, 3, 1).cpu().numpy()
    
    # Convert to uint8
    video_np = (video_np * 255).astype(np.uint8)
    
    return video_np


def save_video_grid(
    videos: List[torch.Tensor],
    output_path: str,
    prompts: Optional[List[str]] = None,
    ncols: int = 3,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Save a grid of video frames.
    
    Args:
        videos: List of video tensors
        output_path: Path to save the grid image
        prompts: Optional list of prompts to display
        ncols: Number of columns in the grid
        figsize: Figure size
    """
    num_videos = len(videos)
    nrows = (num_videos + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, video in enumerate(videos):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Get middle frame
        video_np = tensor_to_numpy(video)
        mid_frame = video_np[len(video_np) // 2]
        
        ax.imshow(mid_frame)
        ax.axis('off')
        
        if prompts and idx < len(prompts):
            title = prompts[idx][:50] + "..." if len(prompts[idx]) > 50 else prompts[idx]
            ax.set_title(title, fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_videos, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved video grid to {output_path}")


def create_video_gif(
    video: torch.Tensor,
    output_path: str,
    fps: int = 8,
    duration: Optional[float] = None
):
    """
    Create a GIF from a video tensor.
    
    Args:
        video: Video tensor of shape (T, C, H, W)
        output_path: Path to save the GIF
        fps: Frames per second
        duration: Duration per frame in seconds (overrides fps if set)
    """
    video_np = tensor_to_numpy(video)
    
    frames = []
    for frame in video_np:
        frames.append(Image.fromarray(frame))
    
    duration_per_frame = duration if duration is not None else (1.0 / fps) * 1000  # Convert to ms
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame,
        loop=0
    )
    print(f"Saved GIF to {output_path}")


def create_video_comparison(
    video1: torch.Tensor,
    video2: torch.Tensor,
    output_path: str,
    label1: str = "Video 1",
    label2: str = "Video 2",
    fps: int = 8
):
    """
    Create a side-by-side comparison of two videos.
    
    Args:
        video1: First video tensor
        video2: Second video tensor
        output_path: Path to save the comparison GIF
        label1: Label for first video
        label2: Label for second video
        fps: Frames per second
    """
    video1_np = tensor_to_numpy(video1)
    video2_np = tensor_to_numpy(video2)
    
    # Ensure same number of frames
    min_frames = min(len(video1_np), len(video2_np))
    video1_np = video1_np[:min_frames]
    video2_np = video2_np[:min_frames]
    
    # Ensure same height
    h1, w1 = video1_np[0].shape[:2]
    h2, w2 = video2_np[0].shape[:2]
    h = max(h1, h2)
    
    # Resize if needed
    if h1 != h or h2 != h:
        video1_np = np.array([
            np.array(Image.fromarray(frame).resize((w1, h))) 
            for frame in video1_np
        ])
        video2_np = np.array([
            np.array(Image.fromarray(frame).resize((w2, h))) 
            for frame in video2_np
        ])
    
    # Concatenate horizontally
    combined_frames = []
    for f1, f2 in zip(video1_np, video2_np):
        combined = np.hstack([f1, f2])
        combined_frames.append(Image.fromarray(combined))
    
    # Add labels
    frames_with_labels = []
    for frame in combined_frames:
        frame_np = np.array(frame)
        h_frame, w_frame = frame_np.shape[:2]
        
        # Create image with text
        img = Image.fromarray(frame_np)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw labels
        draw.text((10, 10), label1, fill=(255, 255, 255), font=font)
        draw.text((w_frame // 2 + 10, 10), label2, fill=(255, 255, 255), font=font)
        
        frames_with_labels.append(img)
    
    frames_with_labels[0].save(
        output_path,
        save_all=True,
        append_images=frames_with_labels[1:],
        duration=(1.0 / fps) * 1000,
        loop=0
    )
    print(f"Saved comparison GIF to {output_path}")


def display_video(
    video: torch.Tensor,
    title: str = "Video",
    save_path: Optional[str] = None
):
    """
    Display a video in a Jupyter notebook or save as GIF.
    
    Args:
        video: Video tensor
        title: Title for the video
        save_path: Optional path to save as GIF
    """
    video_np = tensor_to_numpy(video)
    
    if save_path:
        create_video_gif(video, save_path)
    
    # Create matplotlib animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.axis('off')
    
    im = ax.imshow(video_np[0])
    
    def animate(frame):
        im.set_array(video_np[frame])
        return [im]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(video_np),
        interval=1000 / 8, blit=True, repeat=True
    )
    
    plt.tight_layout()
    return anim


def save_video_frames(
    video: torch.Tensor,
    output_dir: str,
    prefix: str = "frame"
):
    """
    Save individual frames of a video.
    
    Args:
        video: Video tensor
        output_dir: Directory to save frames
        prefix: Prefix for frame filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_np = tensor_to_numpy(video)
    
    for idx, frame in enumerate(video_np):
        frame_img = Image.fromarray(frame)
        frame_path = os.path.join(output_dir, f"{prefix}_{idx:04d}.png")
        frame_img.save(frame_path)
    
    print(f"Saved {len(video_np)} frames to {output_dir}")
