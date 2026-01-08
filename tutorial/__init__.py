"""
Self-Forcing Tutorial Package

A simplified, educational version of the Self-Forcing codebase.
"""

__version__ = "0.1.0"

from tutorial.data import ToyDataset, ToyVideoGenerator
from tutorial.evaluation import (
    FrameConsistencyMetric,
    CLIPScoreMetric,
    compute_all_metrics
)
from tutorial.visualization import (
    save_video_grid,
    create_video_gif,
    TrainingPlotter
)

__all__ = [
    'ToyDataset',
    'ToyVideoGenerator',
    'FrameConsistencyMetric',
    'CLIPScoreMetric',
    'compute_all_metrics',
    'save_video_grid',
    'create_video_gif',
    'TrainingPlotter',
]
