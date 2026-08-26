#!/usr/bin/env python3
"""
Test skeletal-aware target masking functionality.
"""

import torch
import numpy as np
from deeplabcut.pose_estimation_pytorch.runners.train import apply_skeletal_target_masking


def test_skeletal_masking():
    """Test that skeletal masking applies correctly to target heatmaps"""
    print("Testing skeletal-aware target masking...")
    
    device = torch.device('cpu')
    batch_size = 2
    height, width = 112, 112  # Typical heatmap size (448/4 = 112)
    num_joints = 26
    
    # Create mock target structure
    target = {
        "bodypart": {
            "heatmap": {
                "target": torch.ones(batch_size, height, width, num_joints)  # All ones initially
            }
        }
    }
    
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    # Create mock batch annotations
    batch_annotations = {
        'keypoints': []
    }
    
    # Add keypoints for each sample in batch
    for batch_idx in range(batch_size):
        # Create keypoints for one animal
        animal_keypoints = torch.zeros(num_joints, 3)  # (num_joints, 3) - x, y, visibility
        
        # Set up some visible keypoints for testing
        # Left arm: shoulder -> elbow -> wrist
        animal_keypoints[2, :] = torch.tensor([50.0, 50.0, 1.0])   # left_shoulder
        animal_keypoints[18, :] = torch.tensor([80.0, 80.0, 1.0])  # left_elbow  
        animal_keypoints[19, :] = torch.tensor([100.0, 100.0, 1.0]) # left_wrist
        
        # Right arm: shoulder -> elbow -> wrist
        animal_keypoints[3, :] = torch.tensor([350.0, 50.0, 1.0])   # right_shoulder
        animal_keypoints[20, :] = torch.tensor([380.0, 80.0, 1.0])  # right_elbow
        animal_keypoints[21, :] = torch.tensor([400.0, 100.0, 1.0]) # right_wrist
        
        batch_annotations['keypoints'].append([animal_keypoints])  # List of animals (just one)
    
    # Create skeletal data
    skeletal_data = {
        'links': [
            [(2, 18), (18, 19), (3, 20), (20, 21)],  # Sample 1: left_shoulder->left_elbow, left_elbow->left_wrist, etc.
            [(2, 18), (18, 19)]                      # Sample 2: fewer links
        ],
        'link_lengths': [
            [30.0, 25.0, 30.0, 25.0],  # Sample 1: upper.forelimb, lower.forelimb, upper.forelimb, lower.forelimb
            [30.0, 25.0]               # Sample 2: fewer lengths
        ]
    }
    
    print("Before masking:")
    print(f"Left elbow heatmap sum (sample 0): {target['bodypart']['heatmap']['target'][0, :, :, 18].sum().item()}")
    print(f"Left wrist heatmap sum (sample 0): {target['bodypart']['heatmap']['target'][0, :, :, 19].sum().item()}")
    
    # Apply skeletal masking
    masked_target = apply_skeletal_target_masking(
        target=target,
        batch_annotations=batch_annotations,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        stride=4.0,  # Test with stride=4
        skeletal_radius_multiplier=1.0,
        union_intersect_adjacent_skeletal_mask_alpha=0.0  # Default union mode
    )
    
    print("\nAfter masking:")
    print(f"Left elbow heatmap sum (sample 0): {masked_target['bodypart']['heatmap']['target'][0, :, :, 18].sum().item()}")
    print(f"Left wrist heatmap sum (sample 0): {masked_target['bodypart']['heatmap']['target'][0, :, :, 19].sum().item()}")
    
    # Check that masking was applied (sum should be less than original)
    original_sum = height * width  # All ones
    elbow_sum = masked_target['bodypart']['heatmap']['target'][0, :, :, 18].sum().item()
    wrist_sum = masked_target['bodypart']['heatmap']['target'][0, :, :, 19].sum().item()
    
    if elbow_sum < original_sum:
        print("✓ Left elbow masking applied successfully")
    else:
        print("❌ Left elbow masking not applied")
    
    if wrist_sum < original_sum:
        print("✓ Left wrist masking applied successfully")
    else:
        print("❌ Left wrist masking not applied")
    
    # Test with no skeletal data
    print("\nTesting with no skeletal data...")
    target_no_skeletal = {
        "bodypart": {
            "heatmap": {
                "target": torch.ones(batch_size, height, width, num_joints)
            }
        }
    }
    
    masked_target_no_skeletal = apply_skeletal_target_masking(
        target=target_no_skeletal,
        batch_annotations=batch_annotations,
        skeletal_data={},  # Empty skeletal data
        bodyparts=bodyparts,
        device=device,
        stride=4.0,
        skeletal_radius_multiplier=1.0,
        union_intersect_adjacent_skeletal_mask_alpha=0.0
    )
    
    # Should remain unchanged
    if torch.equal(target_no_skeletal['bodypart']['heatmap']['target'], 
                   masked_target_no_skeletal['bodypart']['heatmap']['target']):
        print("✓ No masking applied when no skeletal data (correct)")
    else:
        print("❌ Unexpected masking applied when no skeletal data")
    
    # Test with invalid target structure
    print("\nTesting with invalid target structure...")
    invalid_target = {"invalid": "structure"}
    
    masked_invalid = apply_skeletal_target_masking(
        target=invalid_target,
        batch_annotations=batch_annotations,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        stride=4.0,
        skeletal_radius_multiplier=1.0,
        union_intersect_adjacent_skeletal_mask_alpha=0.0
    )
    
    if masked_invalid == invalid_target:
        print("✓ Invalid target structure handled correctly")
    else:
        print("❌ Invalid target structure not handled correctly")


def main():
    """Run skeletal masking tests"""
    print("Running skeletal-aware target masking tests...")
    print("=" * 60)
    
    try:
        test_skeletal_masking()
        print("\n✅ All skeletal masking tests completed!")
        print("The skeletal masking function constrains target heatmaps based on limb lengths.")
    except Exception as e:
        print(f"\n❌ Skeletal masking test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
