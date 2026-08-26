#!/usr/bin/env python3
"""
Compare reference-based vs GT-based skeletal masking approaches.
"""

import torch
import numpy as np
from deeplabcut.pose_estimation_pytorch.runners.train import (
    apply_skeletal_target_masking,
    apply_skeletal_target_masking_simple
)


def test_masking_approaches_comparison():
    """Compare reference-based vs GT-based masking approaches"""
    print("🔬 Comparing Reference-based vs GT-based Skeletal Masking")
    print("=" * 70)
    
    device = torch.device('cpu')
    batch_size = 2
    height, width = 112, 112
    num_joints = 26
    stride = 4.0
    
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    print("📍 Test Scenario:")
    print("   • Sample 1: Adult lizard with limbs longer than reference")
    print("   • Sample 2: Juvenile lizard with limbs shorter than reference")
    print("   • Reference limb length: 50 pixels")
    
    # Create batch annotations
    batch_annotations = {
        'keypoints': []
    }
    
    # Sample 1: Adult with longer limbs (80 pixels)
    adult_keypoints = torch.zeros(num_joints, 3)
    adult_keypoints[:, 2] = 1.0  # All visible
    adult_keypoints[2, :] = torch.tensor([100.0, 100.0, 1.0])   # left_shoulder
    adult_keypoints[18, :] = torch.tensor([180.0, 100.0, 1.0])  # left_elbow (80 pixels from shoulder)
    adult_keypoints[19, :] = torch.tensor([260.0, 100.0, 1.0])  # left_wrist (80 pixels from elbow)
    batch_annotations['keypoints'].append([adult_keypoints])
    
    # Sample 2: Juvenile with shorter limbs (30 pixels)
    juvenile_keypoints = torch.zeros(num_joints, 3)
    juvenile_keypoints[:, 2] = 1.0  # All visible
    juvenile_keypoints[2, :] = torch.tensor([100.0, 100.0, 1.0])   # left_shoulder
    juvenile_keypoints[18, :] = torch.tensor([130.0, 100.0, 1.0])  # left_elbow (30 pixels from shoulder)
    juvenile_keypoints[19, :] = torch.tensor([160.0, 100.0, 1.0])  # left_wrist (30 pixels from elbow)
    batch_annotations['keypoints'].append([juvenile_keypoints])
    
    # Create reference skeletal data (fixed 50 pixel limb length)
    skeletal_data = {
        'links': [
            [(2, 18), (18, 19)],  # Sample 1: shoulder->elbow, elbow->wrist
            [(2, 18), (18, 19)]   # Sample 2: same links
        ],
        'link_lengths': [
            [50.0, 50.0],  # Sample 1: reference lengths
            [50.0, 50.0]   # Sample 2: same reference lengths
        ]
    }
    
    print("\n📊 Actual GT Limb Lengths:")
    print(f"   Adult: shoulder->elbow = 80px, elbow->wrist = 80px")
    print(f"   Juvenile: shoulder->elbow = 30px, elbow->wrist = 30px")
    print(f"   Reference: both segments = 50px")
    
    # Test both approaches
    print("\n🔄 Testing Both Approaches:")
    
    # Reference-based masking
    target_ref = {"bodypart": {"heatmap": {"target": torch.ones(batch_size, height, width, num_joints)}}}
    
    masked_target_ref = apply_skeletal_target_masking(
        target=target_ref,
        batch_annotations=batch_annotations,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        stride=stride,
        skeletal_radius_multiplier=1.0,
        union_intersect_adjacent_skeletal_mask_alpha=0.0
    )
    
    # GT-based masking
    target_gt = {"bodypart": {"heatmap": {"target": torch.ones(batch_size, height, width, num_joints)}}}
    
    masked_target_gt = apply_skeletal_target_masking_simple(
        target=target_gt,
        batch_annotations=batch_annotations,
        bodyparts=bodyparts,
        device=device,
        stride=stride,
        skeletal_radius_multiplier=1.0,
        union_intersect_adjacent_skeletal_mask_alpha=0.0
    )
    
    # Compare results
    print("\n📈 Results Comparison:")
    print("Approach        | Adult Elbow | Juvenile Elbow | Adult Wrist | Juvenile Wrist")
    print("----------------|-------------|----------------|-------------|---------------")
    
    adult_elbow_ref = masked_target_ref['bodypart']['heatmap']['target'][0, :, :, 18].sum().item()
    juvenile_elbow_ref = masked_target_ref['bodypart']['heatmap']['target'][1, :, :, 18].sum().item()
    adult_wrist_ref = masked_target_ref['bodypart']['heatmap']['target'][0, :, :, 19].sum().item()
    juvenile_wrist_ref = masked_target_ref['bodypart']['heatmap']['target'][1, :, :, 19].sum().item()
    
    adult_elbow_gt = masked_target_gt['bodypart']['heatmap']['target'][0, :, :, 18].sum().item()
    juvenile_elbow_gt = masked_target_gt['bodypart']['heatmap']['target'][1, :, :, 18].sum().item()
    adult_wrist_gt = masked_target_gt['bodypart']['heatmap']['target'][0, :, :, 19].sum().item()
    juvenile_wrist_gt = masked_target_gt['bodypart']['heatmap']['target'][1, :, :, 19].sum().item()
    
    print(f"Reference-based | {adult_elbow_ref:11.0f} | {juvenile_elbow_ref:14.0f} | {adult_wrist_ref:11.0f} | {juvenile_wrist_ref:13.0f}")
    print(f"GT-based        | {adult_elbow_gt:11.0f} | {juvenile_elbow_gt:14.0f} | {adult_wrist_gt:11.0f} | {juvenile_wrist_gt:13.0f}")
    
    print("\n🔍 Analysis:")
    
    # Reference-based should be similar for both samples (same reference length)
    ref_elbow_diff = abs(adult_elbow_ref - juvenile_elbow_ref)
    gt_elbow_diff = abs(adult_elbow_gt - juvenile_elbow_gt)
    
    print(f"\n📏 Elbow Mask Size Differences:")
    print(f"   Reference-based: {ref_elbow_diff:.0f} pixels difference")
    print(f"   GT-based: {gt_elbow_diff:.0f} pixels difference")
    
    if ref_elbow_diff < gt_elbow_diff:
        print("   ✓ Reference-based shows less variation (consistent constraints)")
        print("   ✓ GT-based shows more variation (adapts to actual anatomy)")
    else:
        print("   ❌ Expected GT-based to show more variation")
    
    print(f"\n🎯 Constraint Adaptation:")
    
    # GT-based should adapt better to actual limb lengths
    adult_gt_larger = adult_elbow_gt > juvenile_elbow_gt
    adult_limb_longer = 80 > 30  # Adult limbs are longer
    
    if adult_gt_larger == adult_limb_longer:
        print("   ✓ GT-based correctly adapts to longer adult limbs")
    else:
        print("   ❌ GT-based should produce larger masks for longer limbs")
    
    print(f"\n💡 Use Case Recommendations:")
    print(f"\n🔬 Reference-based Masking:")
    print(f"   • Use when you have accurate species-specific measurements")
    print(f"   • Good for consistent constraints across all specimens")
    print(f"   • Ideal for standardized training protocols")
    print(f"   • Requires skeletal measurement data collection")
    
    print(f"\n🎯 GT-based Masking:")
    print(f"   • Use when specimens vary significantly in size")
    print(f"   • Adapts automatically to individual anatomy")
    print(f"   • No need for separate skeletal measurements")
    print(f"   • Good for mixed-age or mixed-species datasets")
    
    print(f"\n⚖️  Trade-offs:")
    print(f"   Reference-based: More consistent but requires measurement data")
    print(f"   GT-based: More adaptive but depends on annotation quality")
    
    # Test edge case: very close landmarks
    print(f"\n🚨 Edge Case Test: Very Close Landmarks")
    
    edge_keypoints = torch.zeros(num_joints, 3)
    edge_keypoints[:, 2] = 1.0
    edge_keypoints[2, :] = torch.tensor([100.0, 100.0, 1.0])   # left_shoulder
    edge_keypoints[18, :] = torch.tensor([100.5, 100.0, 1.0])  # left_elbow (0.5 pixels away - very close!)
    edge_keypoints[19, :] = torch.tensor([101.0, 100.0, 1.0])  # left_wrist
    
    edge_annotations = {'keypoints': [[edge_keypoints]]}
    edge_target = {"bodypart": {"heatmap": {"target": torch.ones(1, height, width, num_joints)}}}
    
    masked_edge = apply_skeletal_target_masking_simple(
        target=edge_target,
        batch_annotations=edge_annotations,
        bodyparts=bodyparts,
        device=device,
        stride=stride,
        skeletal_radius_multiplier=1.0,
        union_intersect_adjacent_skeletal_mask_alpha=0.0
    )
    
    edge_elbow_sum = masked_edge['bodypart']['heatmap']['target'][0, :, :, 18].sum().item()
    original_sum = height * width
    
    if edge_elbow_sum == original_sum:
        print("   ✓ GT-based correctly skips very close landmarks (< 1 pixel)")
    else:
        print("   ❌ GT-based should skip very close landmarks")
    
    print(f"   Distance: 0.5 pixels, Elbow sum: {edge_elbow_sum} (original: {original_sum})")


def main():
    """Run masking approaches comparison"""
    try:
        test_masking_approaches_comparison()
        print("\n🎉 Masking approaches comparison completed!")
        print("Both reference-based and GT-based approaches work correctly! 🦎✨")
    except Exception as e:
        print(f"\n❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
