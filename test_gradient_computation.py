#!/usr/bin/env python3
"""
Test that the skeletal constraint loss can compute gradients correctly without in-place operation errors.
"""

import torch
from deeplabcut.pose_estimation_pytorch.runners.train import compute_skeletal_constraint_loss


def test_gradient_computation():
    """Test that skeletal constraint loss computes gradients correctly"""
    print("Testing gradient computation for skeletal constraint loss...")
    
    # Create test data
    device = torch.device('cpu')
    batch_size = 2
    num_joints = 26
    
    # Create predicted keypoints that require gradients
    predicted_keypoints = torch.zeros(batch_size, 1, num_joints, 3, requires_grad=True)
    
    # Set up keypoints for testing
    # Sample 1: SVL=100, head=20 (ratio=0.2)
    predicted_keypoints.data[0, 0, 0, :] = torch.tensor([0.0, 0.0, 1.0])    # snout
    predicted_keypoints.data[0, 0, 12, :] = torch.tensor([100.0, 0.0, 1.0]) # tail1
    predicted_keypoints.data[0, 0, 1, :] = torch.tensor([25.0, 0.0, 1.0])   # base_of_head (too long)
    
    # Sample 2: SVL=200, head=50 (ratio=0.25)
    predicted_keypoints.data[1, 0, 0, :] = torch.tensor([0.0, 0.0, 1.0])    # snout
    predicted_keypoints.data[1, 0, 12, :] = torch.tensor([200.0, 0.0, 1.0]) # tail1
    predicted_keypoints.data[1, 0, 1, :] = torch.tensor([60.0, 0.0, 1.0])   # base_of_head (too long)
    
    # Create skeletal data
    skeletal_data = {
        'links': [
            [(0, 12), (0, 1)],  # Sample 1: SVL, head.length
            [(0, 12), (0, 1)]   # Sample 2: SVL, head.length
        ],
        'link_lengths': [
            [100.0, 15.0],  # Sample 1: Expected SVL=100, head=15 (ratio=0.15)
            [200.0, 40.0]   # Sample 2: Expected SVL=200, head=40 (ratio=0.20)
        ]
    }
    
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    # Compute skeletal constraint loss
    loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        loss_weight=1.0
    )
    
    print(f"Computed loss: {loss.item():.6f}")
    
    # Test that the loss requires gradients
    assert loss.requires_grad, "Loss should require gradients"
    
    # Test that we can compute gradients without errors
    try:
        loss.backward()
        print("✓ Gradient computation successful")
        
        # Check that gradients were computed
        assert predicted_keypoints.grad is not None, "Gradients should be computed for predicted_keypoints"
        
        # Check that gradients are not all zero (indicating the loss is connected to the inputs)
        grad_norm = torch.norm(predicted_keypoints.grad)
        print(f"Gradient norm: {grad_norm.item():.6f}")
        assert grad_norm.item() > 0, "Gradients should be non-zero"
        
        print("✓ Gradients computed correctly")
        
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"❌ In-place operation error: {e}")
            raise
        else:
            print(f"❌ Other gradient computation error: {e}")
            raise
    
    print("✓ Gradient computation test passed")


def test_gradient_with_mixed_data():
    """Test gradient computation with mixed data (some samples with data, some without)"""
    print("Testing gradient computation with mixed skeletal data...")
    
    device = torch.device('cpu')
    batch_size = 3
    num_joints = 26
    
    # Create predicted keypoints that require gradients
    predicted_keypoints = torch.zeros(batch_size, 1, num_joints, 3, requires_grad=True)
    
    # Set up keypoints for all samples
    for i in range(batch_size):
        predicted_keypoints.data[i, 0, 0, :] = torch.tensor([0.0, 0.0, 1.0])      # snout
        predicted_keypoints.data[i, 0, 12, :] = torch.tensor([100.0, 0.0, 1.0])   # tail1
        predicted_keypoints.data[i, 0, 1, :] = torch.tensor([20.0, 0.0, 1.0])     # base_of_head
    
    # Create skeletal data with mixed availability
    skeletal_data = {
        'links': [
            [],                 # Sample 1: No skeletal data
            [(0, 12), (0, 1)],  # Sample 2: Has skeletal data
            [(0, 12), (0, 1)]   # Sample 3: Has skeletal data
        ],
        'link_lengths': [
            [],                 # Sample 1: No skeletal data
            [100.0, 15.0],      # Sample 2: Expected SVL=100, head=15
            [100.0, 18.0]       # Sample 3: Expected SVL=100, head=18
        ]
    }
    
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    # Compute skeletal constraint loss
    loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        loss_weight=1.0
    )
    
    print(f"Computed loss with mixed data: {loss.item():.6f}")
    
    # Test gradient computation
    try:
        loss.backward()
        print("✓ Gradient computation with mixed data successful")
        
        # Check that gradients were computed
        assert predicted_keypoints.grad is not None, "Gradients should be computed"
        
        print("✓ Mixed data gradient test passed")
        
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"❌ In-place operation error with mixed data: {e}")
            raise
        else:
            print(f"❌ Other gradient computation error with mixed data: {e}")
            raise


def main():
    """Run gradient computation tests"""
    print("Running gradient computation tests...")
    print("=" * 50)
    
    try:
        test_gradient_computation()
        test_gradient_with_mixed_data()
        print("\n✅ All gradient computation tests passed!")
        print("\nThe skeletal constraint loss can now compute gradients correctly!")
        print("No more in-place operation errors during training.")
    except Exception as e:
        print(f"\n❌ Gradient computation test failed: {e}")
        raise


if __name__ == "__main__":
    main()
