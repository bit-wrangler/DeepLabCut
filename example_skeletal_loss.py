#!/usr/bin/env python3
"""
Example demonstrating the skeletal constraint loss functionality in DeepLabCut.

This shows how the skeletal constraint loss is automatically integrated into the training
process when skeletal data is available.
"""

import torch
import numpy as np
from deeplabcut.pose_estimation_pytorch.runners.train import compute_skeletal_constraint_loss


def demonstrate_skeletal_loss():
    """Demonstrate how the skeletal constraint loss works"""
    
    print("DeepLabCut Skeletal Constraint Loss Demo")
    print("=" * 50)
    
    # Example bodyparts from your config
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    print(f"Number of bodyparts: {len(bodyparts)}")
    print(f"Snout index: {bodyparts.index('snout')}")
    print(f"Tail1 index: {bodyparts.index('tail1')}")
    print()
    
    # Create example predicted keypoints for 2 samples
    batch_size = 2
    num_joints = len(bodyparts)
    predicted_keypoints = torch.zeros(batch_size, 1, num_joints, 3)  # (batch, animals, joints, [x,y,vis])
    
    # Sample 1: Lizard with SVL=200 pixels
    predicted_keypoints[0, 0, 0, :] = torch.tensor([100.0, 100.0, 1.0])   # snout
    predicted_keypoints[0, 0, 12, :] = torch.tensor([300.0, 100.0, 1.0])  # tail1 (SVL=200)
    predicted_keypoints[0, 0, 1, :] = torch.tensor([130.0, 100.0, 1.0])   # base_of_head (head=30)
    predicted_keypoints[0, 0, 2, :] = torch.tensor([150.0, 80.0, 1.0])    # left_shoulder
    predicted_keypoints[0, 0, 18, :] = torch.tensor([170.0, 60.0, 1.0])   # left_elbow (upper_forelimb≈28)
    
    # Sample 2: Smaller lizard with SVL=100 pixels
    predicted_keypoints[1, 0, 0, :] = torch.tensor([50.0, 50.0, 1.0])     # snout
    predicted_keypoints[1, 0, 12, :] = torch.tensor([150.0, 50.0, 1.0])   # tail1 (SVL=100)
    predicted_keypoints[1, 0, 1, :] = torch.tensor([65.0, 50.0, 1.0])     # base_of_head (head=15)
    predicted_keypoints[1, 0, 2, :] = torch.tensor([75.0, 40.0, 1.0])     # left_shoulder
    predicted_keypoints[1, 0, 18, :] = torch.tensor([85.0, 30.0, 1.0])    # left_elbow (upper_forelimb≈14)
    
    # Example skeletal data (from CSV)
    skeletal_data = {
        'links': [
            # Sample 1 links: SVL, head.length, upper.forelimb
            [(0, 12), (0, 1), (2, 18)],
            # Sample 2 links: SVL, head.length, upper.forelimb  
            [(0, 12), (0, 1), (2, 18)]
        ],
        'link_lengths': [
            # Sample 1 expected lengths (absolute measurements from CSV)
            [57.81, 15.01, 11.48],  # SVL=57.81mm, head=15.01mm, upper_forelimb=11.48mm
            # Sample 2 expected lengths
            [62.15, 16.23, 12.15]   # SVL=62.15mm, head=16.23mm, upper_forelimb=12.15mm
        ]
    }
    
    print("Predicted keypoint positions:")
    print("Sample 1:")
    print(f"  Snout: {predicted_keypoints[0, 0, 0, :2]}")
    print(f"  Tail1: {predicted_keypoints[0, 0, 12, :2]}")
    print(f"  Base_of_head: {predicted_keypoints[0, 0, 1, :2]}")
    print(f"  Left_shoulder: {predicted_keypoints[0, 0, 2, :2]}")
    print(f"  Left_elbow: {predicted_keypoints[0, 0, 18, :2]}")
    
    print("Sample 2:")
    print(f"  Snout: {predicted_keypoints[1, 0, 0, :2]}")
    print(f"  Tail1: {predicted_keypoints[1, 0, 12, :2]}")
    print(f"  Base_of_head: {predicted_keypoints[1, 0, 1, :2]}")
    print(f"  Left_shoulder: {predicted_keypoints[1, 0, 2, :2]}")
    print(f"  Left_elbow: {predicted_keypoints[1, 0, 18, :2]}")
    print()
    
    # Compute distances
    print("Predicted distances:")
    svl1 = torch.norm(predicted_keypoints[0, 0, 0, :2] - predicted_keypoints[0, 0, 12, :2])
    head1 = torch.norm(predicted_keypoints[0, 0, 0, :2] - predicted_keypoints[0, 0, 1, :2])
    forelimb1 = torch.norm(predicted_keypoints[0, 0, 2, :2] - predicted_keypoints[0, 0, 18, :2])
    
    svl2 = torch.norm(predicted_keypoints[1, 0, 0, :2] - predicted_keypoints[1, 0, 12, :2])
    head2 = torch.norm(predicted_keypoints[1, 0, 0, :2] - predicted_keypoints[1, 0, 1, :2])
    forelimb2 = torch.norm(predicted_keypoints[1, 0, 2, :2] - predicted_keypoints[1, 0, 18, :2])
    
    print(f"Sample 1: SVL={svl1:.1f}, head={head1:.1f}, forelimb={forelimb1:.1f}")
    print(f"Sample 2: SVL={svl2:.1f}, head={head2:.1f}, forelimb={forelimb2:.1f}")
    print()
    
    print("Expected measurements (from CSV):")
    print(f"Sample 1: SVL={skeletal_data['link_lengths'][0][0]:.2f}, head={skeletal_data['link_lengths'][0][1]:.2f}, forelimb={skeletal_data['link_lengths'][0][2]:.2f}")
    print(f"Sample 2: SVL={skeletal_data['link_lengths'][1][0]:.2f}, head={skeletal_data['link_lengths'][1][1]:.2f}, forelimb={skeletal_data['link_lengths'][1][2]:.2f}")
    print()
    
    # Compute skeletal constraint loss
    device = torch.device('cpu')
    loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        loss_weight=1.0
    )
    
    print(f"Skeletal constraint loss: {loss.item():.6f}")
    print()
    
    # Explain the loss calculation
    print("Loss calculation explanation:")
    print("The loss normalizes all measurements by the predicted SVL to make them scale-invariant.")
    print("For each limb, it computes: max(0, (predicted_ratio - expected_ratio)²)")
    print("where ratio = limb_length / SVL")
    print()
    
    # Calculate expected ratios
    print("Normalized ratios (limb_length / SVL):")
    for i, sample_name in enumerate(["Sample 1", "Sample 2"]):
        expected_svl = skeletal_data['link_lengths'][i][0]
        expected_head = skeletal_data['link_lengths'][i][1]
        expected_forelimb = skeletal_data['link_lengths'][i][2]
        
        predicted_svl = svl1 if i == 0 else svl2
        predicted_head = head1 if i == 0 else head2
        predicted_forelimb = forelimb1 if i == 0 else forelimb2
        
        pred_head_ratio = predicted_head / predicted_svl
        exp_head_ratio = expected_head / expected_svl
        pred_forelimb_ratio = predicted_forelimb / predicted_svl
        exp_forelimb_ratio = expected_forelimb / expected_svl
        
        print(f"{sample_name}:")
        print(f"  Head ratio - predicted: {pred_head_ratio:.4f}, expected: {exp_head_ratio:.4f}")
        print(f"  Forelimb ratio - predicted: {pred_forelimb_ratio:.4f}, expected: {exp_forelimb_ratio:.4f}")
        
        head_diff = pred_head_ratio - exp_head_ratio
        forelimb_diff = pred_forelimb_ratio - exp_forelimb_ratio
        
        head_loss = max(0, head_diff) ** 2
        forelimb_loss = max(0, forelimb_diff) ** 2
        
        print(f"  Head loss contribution: {head_loss:.6f}")
        print(f"  Forelimb loss contribution: {forelimb_loss:.6f}")
        print()


def show_integration_info():
    """Show how this integrates with training"""
    
    print("Integration with Training Pipeline")
    print("=" * 40)
    
    print("""
The skeletal constraint loss is automatically added to the training process when:

1. Your config.yaml contains: lizard_skeletal_data_path: /path/to/skeletal_data.csv

2. The CSV contains measurements for the bodyparts in your model

3. During training, the loss is computed as:
   
   total_loss = pose_loss + skeletal_constraint_loss
   
   where skeletal_constraint_loss = Σ max(0, (pred_ratio - exp_ratio)²)

4. The loss encourages the model to predict limb proportions that match
   the known anatomical constraints from your skeletal measurements.

5. Edge cases handled:
   - Missing skeletal data for a subject → loss = 0
   - Missing SVL keypoints → loss = 0 (can't normalize)
   - Missing other keypoints → skip those constraints

6. Loss weight can be adjusted by setting skeletal_loss_weight in your model config.

7. The SVL landmark pair is configurable via svl_landmarks in your model config:

     svl_landmarks: [snout, spine6]

   Default (key absent) is [snout, tail1], which reproduces all earlier runs.
   This single pair is used for BOTH the mm->pixel scale that THT converts
   X-ray limb lengths with AND the distance LLL normalizes limb proportions by,
   and it is the pair create_skeleton_dictionary attaches the CSV 'svl' trait
   to. The data side and the runner read the same key through the same resolver
   (deeplabcut/pose_estimation_pytorch/skeletal_config.py), so they cannot
   disagree about which pair carries the reference length.

   Why you might change it: the CSV 'svl' trait is a snout-to-VENT measurement,
   but 'tail1' is the first TAIL landmark and sits behind the vent. Measured
   over 1,623 labelled frames from 76 specimens, the ratio
   ||snout-tail1|| / ||snout-spine6|| has median 1.17, and 'spine6' (the pelvic
   hub) is within 1.2% of SVL of the hip midpoint. So the default makes the
   scale ~17% too large, which loosens LLL's threshold and inflates THT's radii.
""")


if __name__ == "__main__":
    demonstrate_skeletal_loss()
    show_integration_info()
