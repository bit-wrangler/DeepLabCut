#!/usr/bin/env python3
"""
Test the complete skeletal constraint loss integration.
This tests the full pipeline from dataset creation to loss computation.
"""

import tempfile
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Test imports
try:
    from deeplabcut.pose_estimation_pytorch.data.dataset import (
        create_skeleton_dictionary,
        SkeletalPoseDataset,
        PoseDatasetParameters
    )
    from deeplabcut.pose_estimation_pytorch.runners.train import compute_skeletal_constraint_loss
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def test_complete_pipeline():
    """Test the complete skeletal constraint pipeline"""
    print("Testing complete skeletal constraint pipeline...")
    
    # Create temporary skeletal data CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        # Create test skeletal data
        skeletal_data = {
            'lizard_id': [1, 2, 42],
            'alpha_tag': ['yellow.yellow.yellow', 'a57', 'b23'],
            'species': ['equestris', 'sagrei', 'sagrei'],
            'sex': ['male', 'male', 'female'],
            'mass_g': [47.88, 5.84, 6.12],
            'sprint_vertical_Tb': [28.8, 31.1, 29.5],
            'svl': [np.nan, 57.81, 62.15],  # Subject 1 has missing data
            'head.length': [np.nan, 15.01, 16.23],
            'upper.forelimb': [np.nan, 11.48, 12.15],
            'lower.forelimb': [np.nan, 8.31, 9.02],
            'upper.hindlimb': [np.nan, 14.46, 15.12],
            'lower.hindlimb': [np.nan, 13.64, 14.23]
        }
        df = pd.DataFrame(skeletal_data)
        df.to_csv(csv_path, index=False)
        
        # Create test config
        config = {
            'bodyparts': [
                'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
                'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
                'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
                'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
                'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
            ],
            'lizard_skeletal_data_path': csv_path
        }
        
        # Test skeleton dictionary creation
        skeleton_dict = create_skeleton_dictionary(config, csv_path)
        
        print(f"Created skeleton dictionary with {len(skeleton_dict)} subjects")
        assert len(skeleton_dict) == 3, f"Expected 3 subjects, got {len(skeleton_dict)}"
        
        # Subject 0001 should have no data (all NaN)
        assert len(skeleton_dict['0001']['links']) == 0, "Subject 0001 should have no links"
        
        # Subject 0002 should have data
        assert len(skeleton_dict['0002']['links']) > 0, "Subject 0002 should have links"
        
        # Test dataset creation with skeletal data
        params = PoseDatasetParameters(
            bodyparts=config['bodyparts'],
            unique_bpts=[],
            individuals=['individual1'],
            with_center_keypoints=False,
            color_mode='RGB',
            top_down_crop_size=None,
            top_down_crop_margin=None,
            top_down_crop_with_context=True
        )
        
        # Create mock images and annotations
        images = [
            {'id': 1, 'file_name': 'labeled-data/0002_1_notes/img001.jpg', 'width': 640, 'height': 480},
            {'id': 2, 'file_name': 'labeled-data/0042_2_test/img002.jpg', 'width': 640, 'height': 480}
        ]
        
        # Create mock keypoints for all bodyparts
        num_bodyparts = len(config['bodyparts'])
        keypoints_1 = []
        keypoints_2 = []
        for i in range(num_bodyparts):
            keypoints_1.extend([100 + i*2, 100 + i*2, 2])  # x, y, visibility
            keypoints_2.extend([200 + i*3, 200 + i*3, 2])  # x, y, visibility
        
        annotations = [
            {
                'id': 1, 'image_id': 1, 'category_id': 1, 'iscrowd': 0,
                'keypoints': keypoints_1,
                'bbox': [90, 90, 80, 80], 'area': 6400
            },
            {
                'id': 2, 'image_id': 2, 'category_id': 1, 'iscrowd': 0,
                'keypoints': keypoints_2,
                'bbox': [190, 190, 80, 80], 'area': 6400
            }
        ]
        
        # Create skeletal dataset
        dataset = SkeletalPoseDataset(
            skeleton_dict=skeleton_dict,
            images=images,
            annotations=annotations,
            parameters=params,
            mode='train'
        )
        
        print(f"Created dataset with {len(dataset)} samples")

        # Test the skeletal loss function directly
        print("Testing skeletal loss computation...")
        
        # Create mock predicted keypoints
        batch_size = 2
        predicted_keypoints = torch.zeros(batch_size, 1, num_bodyparts, 3)
        
        # Set up realistic keypoints for testing
        # Sample 1: subject 0002
        predicted_keypoints[0, 0, 0, :] = torch.tensor([100.0, 100.0, 1.0])   # snout
        predicted_keypoints[0, 0, 12, :] = torch.tensor([200.0, 100.0, 1.0])  # tail1 (SVL=100)
        predicted_keypoints[0, 0, 1, :] = torch.tensor([115.0, 100.0, 1.0])   # base_of_head (head=15)
        
        # Sample 2: subject 0042
        predicted_keypoints[1, 0, 0, :] = torch.tensor([50.0, 50.0, 1.0])     # snout
        predicted_keypoints[1, 0, 12, :] = torch.tensor([150.0, 50.0, 1.0])   # tail1 (SVL=100)
        predicted_keypoints[1, 0, 1, :] = torch.tensor([66.0, 50.0, 1.0])     # base_of_head (head=16)
        
        # Create skeletal data for loss computation
        skeletal_data_for_loss = {
            'links': [
                skeleton_dict['0002']['links'],
                skeleton_dict['0042']['links']
            ],
            'link_lengths': [
                skeleton_dict['0002']['link_lengths'],
                skeleton_dict['0042']['link_lengths']
            ]
        }
        
        # Compute skeletal loss
        device = torch.device('cpu')
        loss = compute_skeletal_constraint_loss(
            predicted_keypoints=predicted_keypoints,
            skeletal_data=skeletal_data_for_loss,
            bodyparts=config['bodyparts'],
            device=device,
            loss_weight=1.0
        )
        
        print(f"Computed skeletal loss: {loss.item():.6f}")
        
        # The loss should be finite and non-negative
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        print("✓ Complete pipeline test passed")
        
    finally:
        # Clean up
        os.unlink(csv_path)


def test_configuration_integration():
    """Test that configuration options work correctly"""
    print("Testing configuration integration...")
    
    # Test different loss weights
    device = torch.device('cpu')
    
    # Create simple test data
    predicted_keypoints = torch.zeros(1, 1, 26, 3)
    predicted_keypoints[0, 0, 0, :] = torch.tensor([0.0, 0.0, 1.0])    # snout
    predicted_keypoints[0, 0, 12, :] = torch.tensor([100.0, 0.0, 1.0]) # tail1
    predicted_keypoints[0, 0, 1, :] = torch.tensor([20.0, 0.0, 1.0])   # base_of_head (too long)
    
    skeletal_data = {
        'links': [[(0, 12), (0, 1)]],  # SVL, head.length
        'link_lengths': [[100.0, 15.0]]  # Expected SVL=100, head=15
    }
    
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    # Test different weights
    loss_weight_1 = compute_skeletal_constraint_loss(
        predicted_keypoints, skeletal_data, bodyparts, device, loss_weight=1.0
    )
    
    loss_weight_2 = compute_skeletal_constraint_loss(
        predicted_keypoints, skeletal_data, bodyparts, device, loss_weight=2.0
    )
    
    # Loss should scale with weight
    assert abs(loss_weight_2.item() - 2 * loss_weight_1.item()) < 1e-6, \
        "Loss should scale linearly with weight"
    
    print("✓ Configuration integration test passed")


def main():
    """Run all integration tests"""
    print("Running complete skeletal constraint integration tests...")
    print("=" * 60)
    
    try:
        test_complete_pipeline()
        test_configuration_integration()
        print("\n✅ All integration tests passed!")
        print("\nThe skeletal constraint loss is ready to use!")
        print("Add 'lizard_skeletal_data_path: /path/to/your/data.csv' to your config.yaml")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    main()
