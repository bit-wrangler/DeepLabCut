#!/usr/bin/env python3
"""
Test the interaction between main loss and skeletal loss to identify in-place operation issues.
"""

import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.runners.train import compute_skeletal_constraint_loss


class MockPoseModel(nn.Module):
    """Mock pose model that mimics DeepLabCut's structure"""
    
    def __init__(self, num_joints=26):
        super().__init__()
        self.num_joints = num_joints
        self.backbone = nn.Linear(100, 256)
        self.heatmap_head = nn.Linear(256, num_joints * 64 * 64)  # Mock heatmap output
        self.locref_head = nn.Linear(256, num_joints * 64 * 64 * 2)  # Mock locref output
        
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.heatmap_head(features).view(-1, self.num_joints, 64, 64)
        locrefs = self.locref_head(features).view(-1, self.num_joints, 64, 64, 2)
        
        return {
            "bodypart": {
                "heatmap": heatmaps,
                "locref": locrefs
            }
        }
    
    def get_predictions(self, outputs):
        """Mock prediction generation that creates keypoints"""
        heatmaps = outputs["bodypart"]["heatmap"]
        locrefs = outputs["bodypart"]["locref"]
        
        batch_size = heatmaps.shape[0]
        
        # Mock keypoint extraction (simplified)
        poses = torch.zeros(batch_size, 1, self.num_joints, 3, device=heatmaps.device)
        
        for b in range(batch_size):
            for j in range(self.num_joints):
                # Find max location in heatmap
                hm = heatmaps[b, j]
                max_idx = torch.argmax(hm.flatten())
                y, x = max_idx // 64, max_idx % 64
                
                # Get position with locref refinement
                poses[b, 0, j, 0] = x.float() + locrefs[b, j, y, x, 0]  # x
                poses[b, 0, j, 1] = y.float() + locrefs[b, j, y, x, 1]  # y
                poses[b, 0, j, 2] = torch.sigmoid(hm[y, x])  # confidence
        
        return {
            "bodypart": {
                "poses": poses
            }
        }
    
    def get_target(self, outputs, annotations):
        """Mock target generation"""
        batch_size = outputs["bodypart"]["heatmap"].shape[0]
        return {
            "bodypart": {
                "heatmap": torch.randn_like(outputs["bodypart"]["heatmap"]),
                "locref": torch.randn_like(outputs["bodypart"]["locref"])
            }
        }
    
    def get_loss(self, outputs, target):
        """Mock loss computation that might modify tensors"""
        heatmap_loss = torch.mean((outputs["bodypart"]["heatmap"] - target["bodypart"]["heatmap"]) ** 2)
        locref_loss = torch.mean((outputs["bodypart"]["locref"] - target["bodypart"]["locref"]) ** 2)
        
        total_loss = heatmap_loss + 0.05 * locref_loss
        
        return {
            "total_loss": total_loss,
            "heatmap_loss": heatmap_loss,
            "locref_loss": locref_loss
        }


def test_loss_interaction():
    """Test the interaction between main loss and skeletal loss"""
    print("Testing loss interaction...")
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device('cpu')
    batch_size = 8
    num_joints = 26
    
    # Create mock model
    model = MockPoseModel(num_joints=num_joints)
    model.train()
    
    # Create mock input
    inputs = torch.randn(batch_size, 100, requires_grad=True)
    
    # Create skeletal data
    skeletal_data = {
        'links': [
            [(0, 12), (0, 1)] for _ in range(batch_size)
        ],
        'link_lengths': [
            [100.0, 15.0] for _ in range(batch_size)
        ]
    }
    
    bodyparts = [
        'snout', 'base_of_head', 'left_shoulder', 'right_shoulder',
        'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5',
        'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3',
        'tail4', 'tail5', 'left_elbow', 'left_wrist', 'right_elbow',
        'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
    ]
    
    # Mock annotations
    annotations = {}
    
    print("Step 1: Forward pass...")
    outputs = model(inputs)
    print(f"Outputs generated: {list(outputs.keys())}")
    
    print("Step 2: Generate targets...")
    target = model.get_target(outputs, annotations)
    print(f"Targets generated: {list(target.keys())}")
    
    print("Step 3: Compute main losses...")
    losses_dict = model.get_loss(outputs, target)
    print(f"Main losses: {list(losses_dict.keys())}")
    print(f"Total loss: {losses_dict['total_loss'].item():.6f}")
    
    print("Step 4: Generate predictions...")
    predictions = model.get_predictions(outputs)
    predicted_keypoints = predictions["bodypart"]["poses"]
    print(f"Predicted keypoints shape: {predicted_keypoints.shape}")
    print(f"Predicted keypoints requires_grad: {predicted_keypoints.requires_grad}")
    
    # Set up some reasonable keypoint positions for skeletal loss
    with torch.no_grad():
        for i in range(batch_size):
            predicted_keypoints[i, 0, 0, :2] = torch.tensor([0.0, 0.0])    # snout
            predicted_keypoints[i, 0, 0, 2] = 1.0  # visible
            predicted_keypoints[i, 0, 12, :2] = torch.tensor([100.0, 0.0]) # tail1
            predicted_keypoints[i, 0, 12, 2] = 1.0  # visible
            predicted_keypoints[i, 0, 1, :2] = torch.tensor([20.0, 0.0])   # base_of_head (too long)
            predicted_keypoints[i, 0, 1, 2] = 1.0  # visible
    
    print("Step 5: Compute skeletal constraint loss...")
    skeletal_loss = compute_skeletal_constraint_loss(
        predicted_keypoints=predicted_keypoints,
        skeletal_data=skeletal_data,
        bodyparts=bodyparts,
        device=device,
        loss_weight=0.1
    )
    print(f"Skeletal loss: {skeletal_loss.item():.6f}")
    
    print("Step 6: Combine losses...")
    losses_dict["skeletal_loss"] = skeletal_loss
    losses_dict["total_loss"] = losses_dict["total_loss"] + skeletal_loss
    print(f"Combined total loss: {losses_dict['total_loss'].item():.6f}")
    
    print("Step 7: Backward pass...")
    try:
        losses_dict["total_loss"].backward()
        print("✓ Backward pass successful")
        
        # Check gradients
        if inputs.grad is not None:
            grad_norm = torch.norm(inputs.grad)
            print(f"Input gradient norm: {grad_norm.item():.6f}")
        
        print("✓ Loss interaction test passed")
        
    except RuntimeError as e:
        print(f"❌ Error in backward pass: {e}")
        print("\nThis error should help identify the interaction issue.")
        raise


def main():
    """Run loss interaction test"""
    print("Running loss interaction test...")
    print("=" * 50)
    
    try:
        test_loss_interaction()
        print("\n✅ Loss interaction test passed!")
        print("No issues found in the interaction between main loss and skeletal loss.")
    except Exception as e:
        print(f"\n❌ Loss interaction test failed: {e}")
        print("\nThis should help identify the source of the in-place operation error.")


if __name__ == "__main__":
    main()
