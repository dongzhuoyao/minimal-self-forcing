"""
Training Progress Visualization

Tools for visualizing training metrics and progress.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import json
import os


class TrainingPlotter:
    """Plot training metrics over time."""
    
    def __init__(self, save_dir: str = "tutorial/logs/plots"):
        """
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metrics_history = {}
    
    def log_metric(self, name: str, value: float, step: int):
        """Log a metric value."""
        if name not in self.metrics_history:
            self.metrics_history[name] = {"steps": [], "values": []}
        
        self.metrics_history[name]["steps"].append(step)
        self.metrics_history[name]["values"].append(value)
    
    def plot_metric(
        self,
        metric_name: str,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot a single metric over time.
        
        Args:
            metric_name: Name of the metric to plot
            title: Optional plot title
            ylabel: Optional y-axis label
            save_path: Optional path to save the plot
        """
        if metric_name not in self.metrics_history:
            print(f"Warning: Metric '{metric_name}' not found")
            return
        
        steps = self.metrics_history[metric_name]["steps"]
        values = self.metrics_history[metric_name]["values"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(ylabel or metric_name, fontsize=12)
        plt.title(title or f"{metric_name} over Training", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, f"{metric_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_multiple_metrics(
        self,
        metric_names: List[str],
        title: str = "Training Metrics",
        save_path: Optional[str] = None
    ):
        """
        Plot multiple metrics on the same figure.
        
        Args:
            metric_names: List of metric names to plot
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name in metric_names:
            if metric_name in self.metrics_history:
                steps = self.metrics_history[metric_name]["steps"]
                values = self.metrics_history[metric_name]["values"]
                plt.plot(steps, values, label=metric_name, linewidth=2)
        
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, "all_metrics.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_loss_components(
        self,
        loss_components: Dict[str, List[float]],
        steps: List[int],
        save_path: Optional[str] = None
    ):
        """
        Plot different components of the loss.
        
        Args:
            loss_components: Dictionary mapping component names to values
            steps: List of step numbers
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for component_name, values in loss_components.items():
            plt.plot(steps, values, label=component_name, linewidth=2)
        
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Loss Components", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for loss
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, "loss_components.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def save_history(self, filepath: str):
        """Save metric history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Saved metric history to {filepath}")
    
    def load_history(self, filepath: str):
        """Load metric history from JSON."""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        print(f"Loaded metric history from {filepath}")


def plot_evaluation_results(
    results: Dict[str, float],
    save_path: str = "tutorial/logs/plots/evaluation_results.png"
):
    """
    Plot evaluation results as a bar chart.
    
    Args:
        results: Dictionary of metric names and values
        save_path: Path to save the plot
    """
    metrics = list(results.keys())
    values = list(results.values())
    
    # Filter out None values
    valid_pairs = [(m, v) for m, v in zip(metrics, values) if v is not None]
    if not valid_pairs:
        print("No valid metrics to plot")
        return
    
    metrics, values = zip(*valid_pairs)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color='steelblue', alpha=0.7)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Evaluation Results", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plot to {save_path}")
