"""
Test that demonstrates the fix for the truncation equivalence issue.

This test verifies that with a very large radius multiplier, truncation
should produce the same results as no truncation at all.
"""
import torch
import sys
sys.path.insert(0, '/home/alek/projects/dlc-dev')

from deeplabcut.pose_estimation_pytorch.runners.train import apply_skeletal_target_masking

device = torch.device('cpu')
batch_size = 2
height, width = 64, 64
num_joints = 30
stride = 4.0

bodyparts = [
    'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'tail1', 'tail2', 'tail3', 'tail4',
    'extra1', 'extra2', 'extra3', 'extra4', 'extra5', 'extra6',
    'extra7', 'extra8', 'extra9', 'extra10', 'extra11', 'extra12'
]

# Create realistic keypoints for two animals
keypoints = torch.zeros(batch_size, 1, num_joints, 3)

# Animal 1 (larger)
keypoints[0, 0, 0, :] = torch.tensor([100.0, 100.0, 1.0])  # snout
keypoints[0, 0, 14, :] = torch.tensor([200.0, 100.0, 1.0])  # tail1 (SVL=100)
keypoints[0, 0, 2, :] = torch.tensor([120.0, 120.0, 1.0])  # left_shoulder
keypoints[0, 0, 4, :] = torch.tensor([140.0, 140.0, 1.0])  # left_elbow
keypoints[0, 0, 6, :] = torch.tensor([150.0, 160.0, 1.0])  # left_wrist
keypoints[0, 0, 3, :] = torch.tensor([120.0, 80.0, 1.0])   # right_shoulder
keypoints[0, 0, 5, :] = torch.tensor([140.0, 60.0, 1.0])   # right_elbow
keypoints[0, 0, 7, :] = torch.tensor([150.0, 40.0, 1.0])   # right_wrist

# Animal 2 (smaller)
keypoints[1, 0, 0, :] = torch.tensor([50.0, 50.0, 1.0])    # snout
keypoints[1, 0, 14, :] = torch.tensor([100.0, 50.0, 1.0])  # tail1 (SVL=50)
keypoints[1, 0, 2, :] = torch.tensor([60.0, 60.0, 1.0])    # left_shoulder
keypoints[1, 0, 4, :] = torch.tensor([70.0, 70.0, 1.0])    # left_elbow
keypoints[1, 0, 6, :] = torch.tensor([75.0, 80.0, 1.0])    # left_wrist

# Skeletal reference data
skeletal_data = {
    'links': [
        [  # Animal 1
            [0, 14],  # snout to tail1
            [2, 4],   # left_shoulder to left_elbow
            [4, 6],   # left_elbow to left_wrist
            [3, 5],   # right_shoulder to right_elbow
            [5, 7],   # right_elbow to right_wrist
        ],
        [  # Animal 2
            [0, 14],  # snout to tail1
            [2, 4],   # left_shoulder to left_elbow
            [4, 6],   # left_elbow to left_wrist
        ]
    ],
    'link_lengths': [
        [  # Animal 1 (in mm)
            50.0,  # SVL
            10.0,  # upper arm
            10.0,  # lower arm
            10.0,  # upper arm
            10.0,  # lower arm
        ],
        [  # Animal 2 (in mm)
            50.0,  # SVL
            10.0,  # upper arm
            10.0,  # lower arm
        ]
    ]
}

batch_annotations = {'keypoints': keypoints}

print("="*80)
print("TRUNCATION EQUIVALENCE TEST")
print("="*80)
print("This test verifies that with a very large radius multiplier (10.0),")
print("truncation should produce the same results as no truncation.")
print()

# Test 1: No truncation (baseline)
target_no_truncation = {
    "bodypart": {
        "heatmap": {
            "target": torch.ones(batch_size, height, width, num_joints)
        }
    }
}

# Test 2: Truncation with large multiplier
target_with_truncation = {
    "bodypart": {
        "heatmap": {
            "target": torch.ones(batch_size, height, width, num_joints)
        }
    }
}

result_with_truncation = apply_skeletal_target_masking(
    target=target_with_truncation,
    batch_annotations=batch_annotations,
    skeletal_data=skeletal_data,
    bodyparts=bodyparts,
    device=device,
    stride=stride,
    skeletal_radius_multiplier=10.0,
    union_intersect_adjacent_skeletal_mask_alpha=0.0
)

# Compare results
expected = height * width
limbs_to_test = [
    ('left_elbow', 4),
    ('right_elbow', 5),
    ('left_wrist', 6),
    ('right_wrist', 7),
]

print("Animal 1 (larger):")
print("-" * 40)
all_match = True
for limb_name, limb_idx in limbs_to_test:
    no_trunc = target_no_truncation['bodypart']['heatmap']['target'][0, :, :, limb_idx].sum().item()
    with_trunc = result_with_truncation['bodypart']['heatmap']['target'][0, :, :, limb_idx].sum().item()
    diff = abs(no_trunc - with_trunc)
    
    if diff < 1.0:
        status = "✅ MATCH"
    else:
        status = f"❌ DIFFER by {diff:.2f}"
        all_match = False
    
    print(f"  {limb_name:15s}: no_trunc={no_trunc:7.2f}, with_trunc={with_trunc:7.2f} {status}")

print()
print("Animal 2 (smaller):")
print("-" * 40)
for limb_name, limb_idx in limbs_to_test[:3]:  # Only test limbs with data
    no_trunc = target_no_truncation['bodypart']['heatmap']['target'][1, :, :, limb_idx].sum().item()
    with_trunc = result_with_truncation['bodypart']['heatmap']['target'][1, :, :, limb_idx].sum().item()
    diff = abs(no_trunc - with_trunc)
    
    if diff < 1.0:
        status = "✅ MATCH"
    else:
        status = f"❌ DIFFER by {diff:.2f}"
        all_match = False
    
    print(f"  {limb_name:15s}: no_trunc={no_trunc:7.2f}, with_trunc={with_trunc:7.2f} {status}")

print()
print("="*80)
if all_match:
    print("✅ SUCCESS: Large radius multiplier produces no truncation!")
    print("   The fix correctly handles the channels-last layout.")
else:
    print("❌ FAILURE: Large radius multiplier still produces truncation!")
    print("   There may be additional issues to investigate.")
print("="*80)

# Test 3: Verify that small multiplier DOES produce truncation
print()
print("="*80)
print("VERIFICATION: Small multiplier should produce truncation")
print("="*80)

target_small_multiplier = {
    "bodypart": {
        "heatmap": {
            "target": torch.ones(batch_size, height, width, num_joints)
        }
    }
}

result_small_multiplier = apply_skeletal_target_masking(
    target=target_small_multiplier,
    batch_annotations=batch_annotations,
    skeletal_data=skeletal_data,
    bodyparts=bodyparts,
    device=device,
    stride=stride,
    skeletal_radius_multiplier=1.0,
    union_intersect_adjacent_skeletal_mask_alpha=0.0
)

print("Animal 1 with multiplier=1.0:")
print("-" * 40)
truncation_applied = False
for limb_name, limb_idx in limbs_to_test:
    no_trunc = target_no_truncation['bodypart']['heatmap']['target'][0, :, :, limb_idx].sum().item()
    with_trunc = result_small_multiplier['bodypart']['heatmap']['target'][0, :, :, limb_idx].sum().item()
    reduction = no_trunc - with_trunc
    
    if reduction > 1.0:
        status = f"✅ TRUNCATED (reduced by {reduction:.2f})"
        truncation_applied = True
    else:
        status = "❌ NOT TRUNCATED"
    
    print(f"  {limb_name:15s}: {with_trunc:7.2f} / {no_trunc:7.2f} {status}")

print()
if truncation_applied:
    print("✅ Verification passed: Small multiplier produces truncation as expected")
else:
    print("❌ Verification failed: Small multiplier should produce truncation")

