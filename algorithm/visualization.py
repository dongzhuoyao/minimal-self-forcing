"""
Visualization tools for understanding the Self-Forcing algorithm.

These tools help visualize how the algorithm works step-by-step.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional
import numpy as np


def visualize_autoregressive_generation(
    video_blocks: List[torch.Tensor],
    save_path: Optional[str] = None,
    block_labels: Optional[List[str]] = None
):
    """
    Visualize how video is generated block by block.
    
    Args:
        video_blocks: List of video blocks, each of shape (B, T, C, H, W)
        save_path: Optional path to save the visualization
        block_labels: Optional labels for each block
    """
    num_blocks = len(video_blocks)
    fig, axes = plt.subplots(1, num_blocks, figsize=(4 * num_blocks, 4))
    
    if num_blocks == 1:
        axes = [axes]
    
    for idx, (block, ax) in enumerate(zip(video_blocks, axes)):
        # Get middle frame of block
        block_np = block[0].permute(1, 2, 0).cpu().numpy()
        mid_frame_idx = block_np.shape[0] // 2
        frame = block_np[mid_frame_idx]
        
        # Normalize to [0, 1]
        if frame.min() < 0:
            frame = (frame + 1) / 2
        frame = np.clip(frame, 0, 1)
        
        ax.imshow(frame)
        ax.set_title(f"Block {idx + 1}" + (f": {block_labels[idx]}" if block_labels else ""))
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_kv_cache_growth(
    cache_sizes: List[int],
    save_path: Optional[str] = None
):
    """
    Visualize how KV cache grows during autoregressive generation.
    
    Args:
        cache_sizes: List of cache sizes after each block
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cache_sizes, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Block Index", fontsize=12)
    plt.ylabel("KV Cache Size", fontsize=12)
    plt.title("KV Cache Growth During Autoregressive Generation", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_denoising_process(
    noisy_frames: List[torch.Tensor],
    denoised_frames: List[torch.Tensor],
    timesteps: List[int],
    save_path: Optional[str] = None
):
    """
    Visualize the denoising process for a single block.
    
    Args:
        noisy_frames: List of noisy frames at each timestep
        denoised_frames: List of denoised predictions
        timesteps: List of timestep values
        save_path: Optional path to save the visualization
    """
    num_steps = len(timesteps)
    fig, axes = plt.subplots(2, num_steps, figsize=(3 * num_steps, 6))
    
    if num_steps == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (noisy, denoised, t) in enumerate(zip(noisy_frames, denoised_frames, timesteps)):
        # Noisy frame
        noisy_np = noisy[0, 0].permute(1, 2, 0).cpu().numpy()
        if noisy_np.min() < 0:
            noisy_np = (noisy_np + 1) / 2
        noisy_np = np.clip(noisy_np, 0, 1)
        
        axes[0, idx].imshow(noisy_np)
        axes[0, idx].set_title(f"Timestep {t}\n(Noisy Input)")
        axes[0, idx].axis('off')
        
        # Denoised frame
        denoised_np = denoised[0, 0].permute(1, 2, 0).cpu().numpy()
        if denoised_np.min() < 0:
            denoised_np = (denoised_np + 1) / 2
        denoised_np = np.clip(denoised_np, 0, 1)
        
        axes[1, idx].imshow(denoised_np)
        axes[1, idx].set_title(f"Denoised Output")
        axes[1, idx].axis('off')
    
    plt.suptitle("Denoising Process for One Block", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def create_algorithm_diagram(save_path: Optional[str] = None):
    """
    Create a diagram explaining the Self-Forcing algorithm flow.
    
    Args:
        save_path: Optional path to save the diagram
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, "Self-Forcing Algorithm Flow", 
            ha='center', va='top', fontsize=16, weight='bold', transform=ax.transAxes)
    
    # Steps
    steps = [
        ("1. Initialize", "Sample noise\nfor entire video"),
        ("2. Block Loop", "For each block:"),
        ("  a. Denoise", "Denoise block\nwith KV cache"),
        ("  b. Update Cache", "Update KV cache\nwith denoised frames"),
        ("  c. Next Block", "Move to\nnext block"),
        ("3. Compute Loss", "Distribution\nmatching loss"),
        ("4. Backprop", "Backpropagate\nthrough process")
    ]
    
    y_positions = np.linspace(0.8, 0.1, len(steps))
    box_width = 0.15
    box_height = 0.08
    
    for idx, ((step_num, step_desc), y_pos) in enumerate(zip(steps, y_positions)):
        # Draw box
        box = plt.Rectangle(
            (0.4 - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            fill=True, edgecolor='black', linewidth=2,
            facecolor='lightblue' if idx < 2 else 'lightgreen'
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(0.4, y_pos, f"{step_num}\n{step_desc}",
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        # Draw arrow
        if idx < len(steps) - 1:
            ax.arrow(0.4, y_pos - box_height/2 - 0.02, 0, -0.05,
                    head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add side notes
    ax.text(0.1, 0.5, "Key Insight:\nSimulate inference\nduring training",
            ha='center', va='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            transform=ax.transAxes)
    
    ax.text(0.9, 0.5, "Benefits:\n• Matches inference\n• KV cache consistency\n• Distribution alignment",
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagram to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Create example visualizations
    print("Creating algorithm visualizations...")
    
    # Algorithm diagram
    create_algorithm_diagram("tutorial/outputs/algorithm_diagram.png")
    
    # Example KV cache growth
    cache_sizes = [0, 3, 6, 9, 12, 15, 18, 21]
    visualize_kv_cache_growth(cache_sizes, "tutorial/outputs/kv_cache_growth.png")
    
    print("Visualizations created!")
