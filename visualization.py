"""
Visualization utilities for video generation and display.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Optional
import imageio

def create_video_gif(video, output_path, fps=8, loop=0):
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
