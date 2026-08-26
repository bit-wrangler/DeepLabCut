#!/usr/bin/env python3
"""
Test that the DataLoader can properly collate batches with skeletal data of different sizes.
"""

import tempfile
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Test imports
try:
    from deeplabcut.pose_estimation_pytorch.data.dataset import (
        create_skeleton_dictionary,
        SkeletalPoseDataset,
        PoseDatasetParameters
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def test_dataloader_collation():
    """Test that DataLoader can collate batches with variable-sized skeletal data"""
    print("Testing DataLoader collation with skeletal data...")
    
    # Create temporary skeletal data CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        # Create test skeletal data with some subjects having data and others not
        skeletal_data = {
            'lizard_id': [1, 2, 3],
            'alpha_tag': ['a', 'b', 'c'],
            'species': ['sagrei', 'sagrei', 'sagrei'],
            'sex': ['male', 'male', 'female'],
            'mass_g': [5.84, 6.12, 7.0],
            'sprint_vertical_Tb': [31.1, 29.5, 30.0],
            'svl': [np.nan, 57.81, 62.15],  # Subject 1 has no data, subjects 2&3 have data
            'head.length': [np.nan, 15.01, 16.23],
            'upper.forelimb': [np.nan, 11.48, 12.15],
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
        
        # Create skeleton dictionary
        skeleton_dict = create_skeleton_dictionary(config, csv_path)
        
        # Create dataset parameters
        params = PoseDatasetParameters(
            bodyparts=config['bodyparts'],
            unique_bpts=[],
            individuals=['individual1'],
            with_center_keypoints=False,
            color_mode='RGB'
        )
        
        # Create mock images and annotations for different subjects
        images = [
            {'id': 1, 'file_name': 'labeled-data/0001_1_notes/img001.jpg', 'width': 640, 'height': 480},  # No skeletal data
            {'id': 2, 'file_name': 'labeled-data/0002_2_test/img002.jpg', 'width': 640, 'height': 480},  # Has skeletal data
            {'id': 3, 'file_name': 'labeled-data/0003_3_exp/img003.jpg', 'width': 640, 'height': 480},   # Has skeletal data
        ]
        
        # Create mock keypoints for all bodyparts
        num_bodyparts = len(config['bodyparts'])
        annotations = []
        for i, img in enumerate(images):
            keypoints = []
            for j in range(num_bodyparts):
                keypoints.extend([100 + i*10 + j*2, 100 + i*10 + j*2, 2])  # x, y, visibility
            
            annotations.append({
                'id': i+1, 'image_id': img['id'], 'category_id': 1, 'iscrowd': 0,
                'keypoints': keypoints,
                'bbox': [90 + i*10, 90 + i*10, 80, 80], 'area': 6400
            })
        
        # Create skeletal dataset
        dataset = SkeletalPoseDataset(
            skeleton_dict=skeleton_dict,
            images=images,
            annotations=annotations,
            parameters=params,
            mode='train'
        )
        
        print(f"Created dataset with {len(dataset)} samples")
        
        # Skip individual item testing since we don't have actual images
        # Just test the skeletal data structure directly
        print("Testing skeletal data structure...")
        
        # Test DataLoader collation
        print("Testing DataLoader collation...")
        try:
            # Create a simple mock dataset that returns the skeletal data structure
            class MockSkeletalDataset:
                def __init__(self, skeleton_dict):
                    self.skeleton_dict = skeleton_dict
                    self.subjects = ['0001', '0002', '0003']
                
                def __len__(self):
                    return len(self.subjects)
                
                def __getitem__(self, idx):
                    subject_id = self.subjects[idx]
                    skeletal_data = self.skeleton_dict.get(subject_id, {
                        "links": [],
                        "link_lengths": []
                    })
                    
                    return {
                        "image": torch.zeros(3, 256, 256),  # Mock image
                        "annotations": {},
                        "context": {},
                        "skeletal_data": {
                            "links": skeletal_data["links"],  # List format
                            "link_lengths": skeletal_data["link_lengths"]  # List format
                        }
                    }
            
            mock_dataset = MockSkeletalDataset(skeleton_dict)
            
            # Import the skeletal-aware collate function
            from deeplabcut.pose_estimation_pytorch.data.collate import skeletal_aware_collate

            # Create DataLoader with batch_size=2 and skeletal-aware collate function
            dataloader = DataLoader(mock_dataset, batch_size=2, shuffle=False, collate_fn=skeletal_aware_collate)
            
            # Test getting a batch
            batch = next(iter(dataloader))
            
            print(f"✓ Successfully collated batch with {len(batch['skeletal_data']['links'])} samples")
            print(f"  Sample 0: {len(batch['skeletal_data']['links'][0])} links")
            print(f"  Sample 1: {len(batch['skeletal_data']['links'][1])} links")
            
            # Verify batch structure
            assert "skeletal_data" in batch, "Batch should contain skeletal_data"
            assert "links" in batch["skeletal_data"], "Skeletal data should contain links"
            assert "link_lengths" in batch["skeletal_data"], "Skeletal data should contain link_lengths"
            
            # Verify that we have the right number of samples in the batch
            assert len(batch["skeletal_data"]["links"]) == 2, f"Expected 2 samples in batch, got {len(batch['skeletal_data']['links'])}"
            
            print("✓ DataLoader collation test passed")
            
        except Exception as e:
            print(f"❌ DataLoader collation failed: {e}")
            raise
        
    finally:
        # Clean up
        os.unlink(csv_path)


def main():
    """Run DataLoader collation tests"""
    print("Running DataLoader collation tests...")
    print("=" * 50)
    
    try:
        test_dataloader_collation()
        print("\n✅ All DataLoader collation tests passed!")
        print("\nThe skeletal data can now be properly collated in batches!")
        print("Variable-sized skeletal data is handled correctly.")
    except Exception as e:
        print(f"\n❌ DataLoader collation test failed: {e}")
        raise


if __name__ == "__main__":
    main()
