"""
Test script to verify tutorial components work correctly.

Run this after installing dependencies:
    pip install -r tutorial/requirements.txt
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from tutorial.data import ToyDataset, ToyVideoGenerator
        print("  ✓ tutorial.data")
        
        from tutorial.evaluation import (
            FrameConsistencyMetric,
            CLIPScoreMetric,
            compute_all_metrics
        )
        print("  ✓ tutorial.evaluation")
        
        from tutorial.visualization import (
            save_video_grid,
            create_video_gif,
            TrainingPlotter
        )
        print("  ✓ tutorial.visualization")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_toy_dataset():
    """Test toy dataset generation."""
    print("\nTesting toy dataset...")
    try:
        from tutorial.data import ToyDataset
        
        dataset = ToyDataset(num_samples=5, width=128, height=128, num_frames=8)
        assert len(dataset) == 5, "Dataset length mismatch"
        
        sample = dataset[0]
        assert "video" in sample, "Missing 'video' key"
        assert "prompt" in sample, "Missing 'prompt' key"
        assert sample["video"].shape[0] == 8, "Wrong number of frames"
        
        print(f"  ✓ Generated {len(dataset)} videos")
        print(f"  ✓ Sample shape: {sample['video'].shape}")
        print(f"  ✓ Sample prompt: {sample['prompt'][:50]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    try:
        from tutorial.data import ToyDataset
        from tutorial.evaluation import FrameConsistencyMetric, compute_all_metrics
        
        dataset = ToyDataset(num_samples=3, width=128, height=128, num_frames=8)
        videos = [dataset[i]["video"] for i in range(3)]
        prompts = [dataset[i]["prompt"] for i in range(3)]
        
        # Test frame consistency
        consistency_metric = FrameConsistencyMetric()
        score = consistency_metric.compute(videos[0])
        assert 0 <= score <= 1, "Consistency score out of range"
        print(f"  ✓ Frame consistency: {score:.4f}")
        
        # Test batch evaluation
        results = compute_all_metrics(videos, prompts)
        assert "frame_consistency" in results, "Missing frame_consistency in results"
        print(f"  ✓ Batch evaluation: {len(results)} metrics computed")
        
        return True
    except Exception as e:
        print(f"  ✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization tools."""
    print("\nTesting visualization...")
    try:
        from tutorial.data import ToyDataset
        from tutorial.visualization import tensor_to_numpy, save_video_grid, create_video_gif
        
        dataset = ToyDataset(num_samples=4, width=128, height=128, num_frames=8)
        videos = [dataset[i]["video"] for i in range(4)]
        prompts = [dataset[i]["prompt"] for i in range(4)]
        
        # Test tensor conversion
        video_np = tensor_to_numpy(videos[0])
        assert video_np.shape[0] == 8, "Wrong number of frames in numpy array"
        assert video_np.shape[3] == 3, "Wrong number of channels"
        print(f"  ✓ Tensor conversion: {video_np.shape}")
        
        # Test video grid (save to temp location)
        output_dir = Path("tutorial/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_video_grid(videos, str(output_dir / "test_grid.png"), prompts=prompts)
        print(f"  ✓ Video grid saved")
        
        # Test GIF creation
        create_video_gif(videos[0], str(output_dir / "test_video.gif"), fps=4)
        print(f"  ✓ GIF created")
        
        return True
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Tutorial Codebase Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Toy Dataset", test_toy_dataset),
        ("Evaluation", test_evaluation),
        ("Visualization", test_visualization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
