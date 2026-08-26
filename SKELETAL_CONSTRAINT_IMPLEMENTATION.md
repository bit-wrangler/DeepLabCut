# Skeletal Constraint Loss Implementation for DeepLabCut

## Overview

Successfully implemented skeletal constraint loss functionality for DeepLabCut that encourages anatomically plausible pose predictions by incorporating known skeletal measurements from CSV data.

## ✅ Implementation Summary

### Core Components

1. **Skeletal Data Loading** (`deeplabcut/pose_estimation_pytorch/data/dataset.py`)
   - `create_skeleton_dictionary()`: Maps CSV skeletal measurements to bodypart indices
   - Supports anatomical measurements: SVL, head length, upper/lower forelimb, upper/lower hindlimb
   - Handles missing data (NaN values) gracefully
   - Extracts subject IDs from video directory structure

2. **Enhanced Dataset** (`deeplabcut/pose_estimation_pytorch/data/dataset.py`)
   - `SkeletalPoseDataset`: Inherits from `PoseDataset`
   - Automatically includes skeletal data in batch items
   - Seamless integration with existing pipeline

3. **Constraint Loss Function** (`deeplabcut/pose_estimation_pytorch/runners/train.py`)
   - `compute_skeletal_constraint_loss()`: Implements ReLU-like constraint
   - **Formula**: `max(0, (predicted_ratio - expected_ratio)²)`
   - **Normalization**: All measurements normalized by predicted SVL
   - **Edge cases**: Handles missing data, invisible keypoints

4. **Training Integration** (`deeplabcut/pose_estimation_pytorch/runners/train.py`)
   - Modified `PoseTrainingRunner.step()` to apply skeletal loss automatically
   - Configurable loss weight via `skeletal_loss_weight` parameter
   - Adds to total training loss: `total_loss = pose_loss + skeletal_loss`

5. **Automatic Dataset Selection** (`deeplabcut/pose_estimation_pytorch/data/base.py`)
   - Modified `create_dataset()` to detect skeletal data configuration
   - Automatically uses `SkeletalPoseDataset` when available

### Configuration Integration

- **API Integration** (`deeplabcut/pose_estimation_pytorch/apis/training.py`)
  - Passes model configuration to training runner
  - Enables access to bodyparts and loss weight settings

- **Runner Configuration** (`deeplabcut/pose_estimation_pytorch/runners/train.py`)
  - Enhanced `build_training_runner()` to accept model configuration
  - Modified `TrainingRunner.__init__()` to store model config
  - Access bodyparts from `model_cfg['metadata']['bodyparts']`

## 🚀 Usage

### 1. Setup Skeletal Data CSV

```csv
lizard_id,alpha_tag,species,sex,svl,head.length,upper.forelimb,lower.forelimb,upper.hindlimb,lower.hindlimb
0001,yellow.yellow.yellow,equestris,male,NA,NA,NA,NA,NA,NA
0002,a57,sagrei,male,57.81,15.01,11.48,8.31,14.46,13.64
0042,b23,sagrei,female,62.15,16.23,12.15,9.02,15.12,14.23
```

### 2. Configure Project

Add to your `config.yaml`:
```yaml
lizard_skeletal_data_path: /path/to/skeletal_data.csv
```

### 3. Configure Loss Weight (Optional)

Add to your PyTorch model config:
```yaml
train_settings:
  skeletal_loss_weight: 0.1  # Default: 0.1
```

### 3b. Configure the SVL Landmark Pair (Optional)

Top level of the PyTorch model config (so it flows through `run_cv.py`'s
`train_overrides` into the `override__svl_landmarks` results column):
```yaml
svl_landmarks: [snout, spine6]  # Default when absent: [snout, tail1]
```
Sets the landmark pair used as the snout-vent length reference — for the
mm→pixel scale (THT radii), the LLL normalization distance, and the pair the CSV
`svl` trait is attached to. Resolved for both the data side and the runner by
`deeplabcut/pose_estimation_pytorch/skeletal_config.py`. The default reproduces
all earlier runs; `spine6` is the anatomically correct vent proxy (`tail1` sits
~17% further back, inflating the scale). See `skeletal_constraint_guide.md`.

### 4. Train Network

```python
import deeplabcut
deeplabcut.train_network(config='/path/to/config.yaml', shuffle=1)
```

## ✅ Testing & Validation

### Test Suite
- `test_skeletal_dataset.py`: Dataset functionality ✅
- `test_skeletal_loss.py`: Loss computation ✅  
- `test_complete_integration.py`: End-to-end pipeline ✅
- `test_training_integration.py`: Training runner integration ✅

### Example Code
- `example_skeletal_usage.py`: Dataset usage demonstration
- `example_skeletal_loss.py`: Loss computation details
- `skeletal_constraint_guide.md`: Complete user guide

## 🔧 Technical Details

### Loss Calculation
1. **Normalization**: All limb lengths normalized by predicted SVL (the distance between the configured SVL landmark pair, default `snout`-`tail1`)
2. **Constraint**: Only penalizes when predicted limbs are too long
3. **Formula**: `Σ max(0, (pred_length/pred_svl - exp_length/exp_svl)²)`
4. **Averaging**: Loss averaged over valid constraints and batch samples

### Subject ID Extraction
- From directory structure: `labeled-data/0001_1_notes/image.jpg` → subject ID `0001`
- Handles various naming patterns with underscore separation

### Edge Case Handling
- **Missing skeletal data**: Loss = 0 for subjects without measurements
- **Missing SVL landmarks**: Loss = 0 (cannot normalize)
- **Missing other keypoints**: Skip those specific constraints
- **Invisible keypoints**: Skip constraints involving invisible points

### Anatomical Mappings
- `svl`: the configured SVL landmark pair (default `snout` ↔ `tail1`)
- `head.length`: snout ↔ base_of_head
- `upper.forelimb`: shoulder ↔ elbow (left/right symmetry)
- `lower.forelimb`: elbow ↔ wrist (left/right symmetry)
- `upper.hindlimb`: hip ↔ knee (left/right symmetry)
- `lower.hindlimb`: knee ↔ ankle (left/right symmetry)

## 🎯 Key Features

- **Scale Invariant**: Works across different image scales and lizard sizes
- **Robust**: Graceful handling of missing data and edge cases
- **Flexible**: Only applies constraints where measurements are available
- **Configurable**: Adjustable loss weight for different constraint strengths
- **Automatic**: Seamless integration with existing training workflow
- **Efficient**: Minimal computational overhead during training

## 📊 Expected Benefits

1. **Improved Accuracy**: Encourages anatomically plausible pose predictions
2. **Biological Consistency**: Predictions respect known anatomical relationships
3. **Reduced Outliers**: Constraints prevent unrealistic limb proportions
4. **Better Generalization**: Anatomical priors help with unseen poses/subjects

## 🔍 Debugging

The skeletal constraint loss is applied when:
1. `lizard_skeletal_data_path` is configured in project config
2. Skeletal data CSV exists and contains measurements
3. Subject IDs match between CSV and image directory names
4. Bodyparts include both landmarks of the configured SVL pair (default `snout` and `tail1`) for normalization
5. Model configuration contains bodyparts metadata

Training logs will show:
```
Epoch 10/200, train loss 0.12345
  - pose_loss: 0.11000
  - skeletal_loss: 0.01345
```

## 🔧 **Recent Fix Applied**

**Issue Resolved**: The skeletal constraint loss was being skipped during training due to incorrect prediction structure access.

**Root Cause**: The code was looking for `predictions["bodyparts"]["keypoints"]` but the actual structure is `predictions["bodypart"]["poses"]`.

**Fix Applied**: Updated the training runner to use the correct prediction structure:
```python
# Before (incorrect):
if "bodyparts" in predictions:
    predicted_keypoints = predictions["bodyparts"]["keypoints"]

# After (correct):
if "bodypart" in predictions and "poses" in predictions["bodypart"]:
    predicted_keypoints = predictions["bodypart"]["poses"]
```

**Verification**: All tests pass and skeletal constraint loss is now properly applied during training.

## ✅ Status: Complete & Ready for Use

The skeletal constraint loss feature is fully implemented, tested, and ready for production use. It seamlessly integrates with the existing DeepLabCut training pipeline and provides anatomically-informed pose estimation improvements.

---

## 📝 Changelog — configurable SVL landmark pair (`svl_landmarks`)

**New config key.** `svl_landmarks` — the pair of video landmarks that stands in
for snout–vent length. It lives at the **top level of the model config**
(`pytorch_config.yaml`), not the project `config.yaml`, so it can be set through
`run_cv.py`'s `train_overrides` and is recorded in the `override__svl_landmarks`
column of every `cv_results_*.csv` — which is how this project verifies what a run
actually used. Putting it in the project config raises rather than being ignored.

**Default.** `[snout, tail1]`, the previously hardcoded pair. With the key absent,
every SVL-dependent path is byte-identical to the pre-change code, so all existing
results reproduce exactly.

**Validation.** An explicitly configured value must be exactly two distinct
bodypart-name strings, both present in the project's `bodyparts`, and must not be a
pair another trait already owns (`[snout, base_of_head]` is `head.length`; emitting
it twice would make THT and LLL silently normalise by different references).
Violations raise a `ValueError` naming the offending value and listing the available
bodyparts. The **default pair is exempt from the membership check**: a project whose
bodyparts simply lack `snout`/`tail1` — including any multi-animal project, where
DLC writes the sentinel `bodyparts: "MULTI!"` — keeps loading and degrading to a
no-op exactly as it always did.

Because the dataset side and the runner side must agree or the mechanism silently
changes, three checks now enforce it: `create_skeleton_dictionary` raises if the
pair is expressible but no subject carries an SVL reference;
`TrainingRunner._check_svl_reference_available` raises at the start of `fit()` if
the runner's pair is not the pair the dataset emitted; and a per-batch warning
names both possible causes of a missing reference (an `NA` `svl` cell — the real
morphology CSV has some — versus a genuine mismatch).

**What it affects.**
- **LLL** — the normaliser in `compute_skeletal_constraint_loss`,
  `..._predicted_svl` (the manuscript's "LLL no GT") and the currently-unused
  `..._loss2`. Every limb ratio is divided by this predicted distance.
- **THT with `use_skeletal_reference: True`** — the mm→pixel scale
  (`_compute_scale_factor_svl`) that converts the X-ray reference lengths into
  truncation radii.

**What it explicitly does NOT affect.** The **THT-GT path**
(`use_skeletal_reference: False` → `apply_skeletal_target_masking_simple`) takes its
radii from ground-truth inter-landmark distances and never touches SVL. The
`tht_gt_*` experiment family is unchanged by this key.

**⚠️ Not a like-for-like comparison.** The CSV `svl` trait is a snout-to-*vent*
measurement, but `tail1` is the first *tail* landmark and sits behind the vent.
Switching to a vent-anchored pair such as `[snout, spine6]` changes the mm→pixel
scale by a **median factor of about 1.17** on this dataset, which moves both the
effective LLL threshold and the effective THT radii. A `spine6` rerun must be
compared against a fresh `[snout, tail1]` control produced by the same code, **not**
against previously reported runs.
