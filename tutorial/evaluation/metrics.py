"""
Simple Evaluation Metrics for Video Generation

This module provides easy-to-use metrics for evaluating generated videos.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import torch.nn.functional as F


class FrameConsistencyMetric:
    """Measure temporal consistency between consecutive frames."""
    
    def __init__(self):
        self.name = "frame_consistency"
    
    def compute(self, video: torch.Tensor) -> float:
        """
        Compute frame consistency score.
        
        Args:
            video: Tensor of shape (T, C, H, W) or (B, T, C, H, W)
        
        Returns:
            Consistency score (higher is better, range [0, 1])
        """
        if len(video.shape) == 4:
            video = video.unsqueeze(0)  # Add batch dimension
        
        batch_size, num_frames, channels, height, width = video.shape
        
        # Compute differences between consecutive frames
        frame_diffs = []
        for t in range(num_frames - 1):
            frame_t = video[:, t]
            frame_t1 = video[:, t + 1]
            
            # L2 distance normalized by frame size
            diff = torch.mean((frame_t - frame_t1) ** 2, dim=(1, 2, 3))
            frame_diffs.append(diff)
        
        frame_diffs = torch.stack(frame_diffs, dim=1)  # (B, T-1)
        
        # Convert to consistency score (lower diff = higher consistency)
        # Normalize to [0, 1] range (assuming normalized inputs in [-1, 1])
        consistency = 1.0 - torch.clamp(frame_diffs.mean(), 0, 1.0)
        
        return consistency.item()
    
    def compute_batch(self, videos: List[torch.Tensor]) -> List[float]:
        """Compute consistency for a batch of videos."""
        return [self.compute(video) for video in videos]


class CLIPScoreMetric:
    """Compute CLIP score for text-video alignment."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        try:
            import clip
            self.device = device
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            self.model.eval()
            self.available = True
        except ImportError:
            print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            self.available = False
    
    def compute(
        self,
        video: torch.Tensor,
        text: str,
        frame_sample_rate: int = 4
    ) -> float:
        """
        Compute CLIP score between video and text.
        
        Args:
            video: Tensor of shape (T, C, H, W) or (B, T, C, H, W), values in [-1, 1]
            text: Text prompt
            frame_sample_rate: Sample every Nth frame for efficiency
        
        Returns:
            CLIP score (higher is better)
        """
        if not self.available:
            return 0.0
        
        if len(video.shape) == 4:
            video = video.unsqueeze(0)
        
        batch_size, num_frames, channels, height, width = video.shape
        
        # Sample frames
        sampled_frames = video[:, ::frame_sample_rate]
        num_sampled = sampled_frames.shape[1]
        
        # Normalize video from [-1, 1] to [0, 1] and convert to PIL images
        video_norm = (sampled_frames + 1.0) / 2.0
        video_norm = torch.clamp(video_norm, 0, 1)
        
        # Resize to CLIP input size (224x224)
        video_norm = F.interpolate(
            video_norm.reshape(-1, channels, height, width),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Process frames
        frame_features = []
        for i in range(num_sampled):
            frame = video_norm[i].cpu()
            frame_pil = Image.fromarray(
                (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            frame_tensor = self.preprocess(frame_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                frame_feat = self.model.encode_image(frame_tensor)
                frame_features.append(frame_feat)
        
        # Average frame features
        video_feature = torch.stack(frame_features).mean(dim=0)
        
        # Encode text
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_tokens)
        
        # Compute cosine similarity
        video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
        score = (video_feature * text_feature).sum().item()
        
        return score
    
    def compute_batch(
        self,
        videos: List[torch.Tensor],
        texts: List[str],
        frame_sample_rate: int = 4
    ) -> List[float]:
        """Compute CLIP scores for a batch."""
        return [
            self.compute(video, text, frame_sample_rate)
            for video, text in zip(videos, texts)
        ]


class VisualQualityMetric:
    """Simple visual quality metrics."""
    
    def __init__(self):
        self.name = "visual_quality"
    
    def compute_psnr(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor
    ) -> float:
        """
        Compute PSNR between two videos (requires ground truth).
        
        Args:
            video1: Generated video, shape (T, C, H, W)
            video2: Ground truth video, shape (T, C, H, W)
        
        Returns:
            PSNR in dB (higher is better)
        """
        mse = torch.mean((video1 - video2) ** 2)
        if mse == 0:
            return float('inf')
        
        max_val = 2.0  # Assuming values in [-1, 1]
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        
        return psnr.item()
    
    def compute_ssim(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        window_size: int = 11
    ) -> float:
        """
        Compute SSIM between two videos (simplified version).
        
        Args:
            video1: Generated video, shape (T, C, H, W)
            video2: Ground truth video, shape (T, C, H, W)
            window_size: Size of the Gaussian window
        
        Returns:
            SSIM score (higher is better, range [0, 1])
        """
        # Simplified SSIM - average over frames
        ssim_scores = []
        
        for t in range(video1.shape[0]):
            frame1 = video1[t]
            frame2 = video2[t]
            
            # Convert to grayscale for simplicity
            if frame1.shape[0] == 3:
                frame1_gray = 0.299 * frame1[0] + 0.587 * frame1[1] + 0.114 * frame1[2]
                frame2_gray = 0.299 * frame2[0] + 0.587 * frame2[1] + 0.114 * frame2[2]
            else:
                frame1_gray = frame1[0]
                frame2_gray = frame2[0]
            
            # Simple SSIM approximation
            mu1 = frame1_gray.mean()
            mu2 = frame2_gray.mean()
            
            sigma1_sq = ((frame1_gray - mu1) ** 2).mean()
            sigma2_sq = ((frame2_gray - mu2) ** 2).mean()
            sigma12 = ((frame1_gray - mu1) * (frame2_gray - mu2)).mean()
            
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            ssim_scores.append(ssim.item())
        
        return np.mean(ssim_scores)


class GenerationSpeedMetric:
    """Measure generation speed."""
    
    def __init__(self):
        self.name = "generation_speed"
    
    def compute_fps(
        self,
        num_frames: int,
        generation_time: float
    ) -> float:
        """
        Compute frames per second.
        
        Args:
            num_frames: Number of generated frames
            generation_time: Time taken in seconds
        
        Returns:
            FPS (higher is better)
        """
        if generation_time == 0:
            return 0.0
        return num_frames / generation_time


def compute_all_metrics(
    videos: List[torch.Tensor],
    prompts: List[str],
    ground_truth_videos: Optional[List[torch.Tensor]] = None,
    generation_times: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        videos: List of generated videos
        prompts: List of text prompts
        ground_truth_videos: Optional ground truth videos for PSNR/SSIM
        generation_times: Optional generation times for FPS calculation
    
    Returns:
        Dictionary of metric names and average scores
    """
    results = {}
    
    # Frame consistency
    consistency_metric = FrameConsistencyMetric()
    consistency_scores = consistency_metric.compute_batch(videos)
    results["frame_consistency"] = np.mean(consistency_scores)
    results["frame_consistency_std"] = np.std(consistency_scores)
    
    # CLIP score
    clip_metric = CLIPScoreMetric()
    if clip_metric.available:
        clip_scores = clip_metric.compute_batch(videos, prompts)
        results["clip_score"] = np.mean(clip_scores)
        results["clip_score_std"] = np.std(clip_scores)
    else:
        results["clip_score"] = None
    
    # Visual quality (if ground truth available)
    if ground_truth_videos is not None:
        quality_metric = VisualQualityMetric()
        psnr_scores = []
        ssim_scores = []
        for vid_gen, vid_gt in zip(videos, ground_truth_videos):
            psnr_scores.append(quality_metric.compute_psnr(vid_gen, vid_gt))
            ssim_scores.append(quality_metric.compute_ssim(vid_gen, vid_gt))
        results["psnr"] = np.mean(psnr_scores)
        results["ssim"] = np.mean(ssim_scores)
    
    # Generation speed
    if generation_times is not None:
        speed_metric = GenerationSpeedMetric()
        fps_scores = []
        for vid, time in zip(videos, generation_times):
            fps_scores.append(speed_metric.compute_fps(vid.shape[0], time))
        results["fps"] = np.mean(fps_scores)
        results["fps_std"] = np.std(fps_scores)
    
    return results
