#!/usr/bin/env python3
"""
Test script for the skeletal constraint loss implementation.
"""

import torch
import numpy as np
from deeplabcut.pose_estimation_pytorch.runners.train import compute_skeletal_constraint_loss


def test_skeletal_loss_basic():
    """Test basic skeletal loss computation"""
    print("Testing basic skeletal loss computation...")
    
    device = torch.device('cpu')
    
    # Create mock bodyparts list
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    # Create mock predicted keypoints (batch_size=2, num_animals=1, num_joints=26, coords=3)
    batch_size = 2
    num_joints = len(bodyparts)
    predicted_keypoints = torch.zeros(batch_size, 1, num_joints, 3)
    
    # Set up keypoints for first sample
    # snout at (100, 100), tail1 at (200, 100) -> SVL = 100
    predicted_keypoints[0, 0, 0, :] = torch.tensor([100.0, 100.0, 1.0])  # snout
    predicted_keypoints[0, 0, 12, :] = torch.tensor([200.0, 100.0, 1.0])  # tail1
    # head: base_of_head at (110, 100) -> head length = 10
    predicted_keypoints[0, 0, 1, :] = torch.tensor([110.0, 100.0, 1.0])  # base_of_head
    
    # Set up keypoints for second sample
    # snout at (50, 50), tail1 at (150, 50) -> SVL = 100
    predicted_keypoints[1, 0, 0, :] = torch.tensor([50.0, 50.0, 1.0])  # snout
    predicted_keypoints[1, 0, 12, :] = torch.tensor([150.0, 50.0, 1.0])  # tail1
    # head: base_of_head at (65, 50) -> head length = 15
    predicted_keypoints[1, 0, 1, :] = torch.tensor([65.0, 50.0, 1.0])  # base_of_head
    
    # Create mock skeletal data
    skeletal_data = {
        'links': [
            [(0, 12), (0, 1)],  # SVL (snout-tail1), head.length (snout-base_of_head)
            [(0, 12), (0, 1)]   # Same for second sample
        ],
        'link_lengths': [
            [100.0, 15.0],  # Expected SVL=100, head.length=15
            [100.0, 12.0]   # Expected SVL=100, head.length=12
        ]
    }
    
    # Compute loss
    loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        loss_weight=1.0
    )
    
    print(f"Computed loss: {loss.item()}")
    
    # Expected behavior:
    # Sample 1: predicted head = 10, expected head = 15, expected SVL = 100
    #   normalized_predicted = 10/100 = 0.1
    #   normalized_expected = 15/100 = 0.15
    #   diff = 0.1 - 0.15 = -0.05 < 0, so loss = 0
    # Sample 2: predicted head = 15, expected head = 12, expected SVL = 100
    #   normalized_predicted = 15/100 = 0.15
    #   normalized_expected = 12/100 = 0.12
    #   diff = 0.15 - 0.12 = 0.03 > 0, so loss = 0.03^2 = 0.0009
    # Average loss over 2 links per sample: (0 + 0) / 2 = 0 for sample 1, (0 + 0.0009) / 2 = 0.00045 for sample 2
    # Average loss over 2 samples: (0 + 0.00045) / 2 = 0.000225

    expected_loss = 0.000225
    assert abs(loss.item() - expected_loss) < 1e-6, f"Expected loss ~{expected_loss}, got {loss.item()}"
    
    print("✓ Basic skeletal loss test passed")


def test_skeletal_loss_missing_data():
    """Test skeletal loss with missing data"""
    print("Testing skeletal loss with missing data...")
    
    device = torch.device('cpu')
    bodyparts = ['snout', 'base_of_head', 'tail1']
    
    # Create keypoints with missing visibility
    predicted_keypoints = torch.zeros(1, 1, 3, 3)
    predicted_keypoints[0, 0, 0, :] = torch.tensor([100.0, 100.0, 0.0])  # snout invisible
    predicted_keypoints[0, 0, 1, :] = torch.tensor([110.0, 100.0, 1.0])  # base_of_head visible
    predicted_keypoints[0, 0, 2, :] = torch.tensor([200.0, 100.0, 1.0])  # tail1 visible
    
    skeletal_data = {
        'links': [[(0, 2), (0, 1)]],  # SVL, head.length
        'link_lengths': [[100.0, 15.0]]
    }
    
    loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device
    )
    
    # Should be 0 because snout is not visible, so normalization fails
    assert loss.item() == 0.0, f"Expected 0 loss for missing snout, got {loss.item()}"
    
    print("✓ Missing data test passed")


def test_skeletal_loss_no_data():
    """Test skeletal loss with no skeletal data"""
    print("Testing skeletal loss with no skeletal data...")
    
    device = torch.device('cpu')
    bodyparts = ['snout', 'base_of_head', 'tail1']
    
    predicted_keypoints = torch.zeros(1, 1, 3, 3)
    skeletal_data = {'links': [], 'link_lengths': []}
    
    loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device
    )
    
    assert loss.item() == 0.0, f"Expected 0 loss for no data, got {loss.item()}"
    
    print("✓ No data test passed")


def main():
    """Run all tests"""
    print("Running skeletal constraint loss tests...")
    
    try:
        test_skeletal_loss_basic()
        test_skeletal_loss_missing_data()
        test_skeletal_loss_no_data()
        print("\n✅ All skeletal loss tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
