# Research: Body Length Error vs Confidence Analysis

## Overview
This document identifies and explains the relevant code sections for implementing body length error vs confidence analysis using frame-level aggregate results from cross-validation experiments.

## Relevant Files and Code Sections

### 1. Frame-Level Aggregate CSV Structure

#### File: `run_cv.py`
**Lines: 158-176** - Frame-level CSV aggregation logic

```python
if SAVE_FRAME_LEVEL_RESULTS:
    # Find frame-level CSV file
    frame_level_csv_files = list(evaluation_folder.glob('*-frame-level-results.csv'))

    if frame_level_csv_files:
        # Read the frame-level CSV
        df_frame_level = pd.read_csv(frame_level_csv_files[0])

        # Add metadata columns at the beginning
        df_frame_level.insert(0, 'shuffle_num', shuffle_num)
        df_frame_level.insert(0, 'landmark_set_name', landmark_set_name)
        df_frame_level.insert(0, 'experiment_id', experiment_id)
        df_frame_level.insert(0, 'seed', seed_idx)
        df_frame_level.insert(0, 'fold', fold_idx)

        # Append to landmark-set-specific aggregate file
        aggregate_file = f'cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv'
        save_frame_level_results_incrementally(df_frame_level, aggregate_file)
```

**How it works:**
- After each `deeplabcut.evaluate_network()` call, the script searches for the generated frame-level CSV
- Metadata columns (fold, seed, experiment_id, landmark_set_name, shuffle_num) are prepended to each row
- Results are appended to a landmark-set-specific aggregate file

**Aggregate CSV Structure:**
- **Filename pattern**: `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`
- **Location**: Same directory as `cv_results_{experiment_id}.csv` (project root)
- **Columns**: 
  - Metadata: `fold`, `seed`, `experiment_id`, `landmark_set_name`, `shuffle_num`
  - Frame info: `frame_index`, `image_path`
  - Per-bodypart data: `gt_{bodypart}_x`, `gt_{bodypart}_y`, `pred_{bodypart}_x`, `pred_{bodypart}_y`, `conf_{bodypart}`

**Example columns for a project with bodyparts [snout, base_of_head, ..., tail1]:**
```
fold, seed, experiment_id, landmark_set_name, shuffle_num, frame_index, image_path,
gt_snout_x, gt_snout_y, pred_snout_x, pred_snout_y, conf_snout,
gt_base_of_head_x, gt_base_of_head_y, pred_base_of_head_x, pred_base_of_head_y, conf_base_of_head,
...,
gt_tail1_x, gt_tail1_y, pred_tail1_x, pred_tail1_y, conf_tail1
```

### 2. Frame-Level CSV Creation Functions

#### File: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`
**Lines: 46-113** - TensorFlow backend frame-level CSV creation

```python
def create_frame_level_results_tf(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    test_indices: List[int],
    scorer_gt: str,
    scorer_pred: str,
) -> pd.DataFrame:
    """Creates a frame-level results DataFrame for test set evaluation."""
    
    # Get bodyparts list from ground truth DataFrame
    bodyparts = (
        ground_truth[scorer_gt].columns.get_level_values("bodyparts").unique().tolist()
    )

    # Build column names
    columns = ["frame_index", "image_path"]
    for bp in bodyparts:
        columns.extend(
            [f"gt_{bp}_x", f"gt_{bp}_y", f"pred_{bp}_x", f"pred_{bp}_y", f"conf_{bp}"]
        )

    # Build rows - iterate through test indices
    rows = []
    for frame_idx, data_idx in enumerate(test_indices):
        image_path = ground_truth.index[data_idx]
        row = [frame_idx, image_path_str]

        for bp in bodyparts:
            # Ground truth coordinates
            gt_x = ground_truth.iloc[data_idx][(scorer_gt, bp, "x")]
            gt_y = ground_truth.iloc[data_idx][(scorer_gt, bp, "y")]

            # Predicted coordinates and confidence
            pred_x = predictions.iloc[data_idx][(scorer_pred, bp, "x")]
            pred_y = predictions.iloc[data_idx][(scorer_pred, bp, "y")]
            conf = predictions.iloc[data_idx][(scorer_pred, bp, "likelihood")]

            row.extend([gt_x, gt_y, pred_x, pred_y, conf])

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
```

**How it works:**
- Extracts bodypart names from the ground truth DataFrame's MultiIndex columns
- For each test frame, extracts GT positions (x, y), predicted positions (x, y), and confidence scores
- Confidence scores come from the "likelihood" column in the predictions DataFrame
- Returns a flat DataFrame with one row per test frame

#### File: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`
**Lines: 108-164** - PyTorch backend frame-level CSV creation

```python
def create_frame_level_results_pytorch(
    ground_truth: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    bodyparts: list[str],
    image_paths: list[str],
) -> pd.DataFrame:
    """Creates a frame-level results DataFrame for test set evaluation."""
    
    # Build column names
    columns = ["frame_index", "image_path"]
    for bp in bodyparts:
        columns.extend(
            [f"gt_{bp}_x", f"gt_{bp}_y", f"pred_{bp}_x", f"pred_{bp}_y", f"conf_{bp}"]
        )

    # Build rows
    rows = []
    for idx, image_path in enumerate(image_paths):
        row = [idx, image_path]

        # Get arrays for this image (single animal: use index 0)
        gt_array = ground_truth.get(image_path, np.full((1, len(bodyparts), 3), np.nan))
        pred_array = predictions.get(image_path, np.full((1, len(bodyparts), 3), np.nan))

        for bp_idx, bp in enumerate(bodyparts):
            # Ground truth (x, y) - for single animal, use individual index 0
            gt_x = gt_array[0, bp_idx, 0]
            gt_y = gt_array[0, bp_idx, 1]

            # Predictions (x, y, confidence) - for single animal, use individual index 0
            pred_x = pred_array[0, bp_idx, 0]
            pred_y = pred_array[0, bp_idx, 1]
            conf = pred_array[0, bp_idx, 2]

            row.extend([gt_x, gt_y, pred_x, pred_y, conf])

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
```

**How it works:**
- Takes dictionaries mapping image paths to numpy arrays
- Arrays have shape `[n_individuals, n_bodyparts, 3]` where last dimension is `[x, y, confidence]`
- For single-animal projects, uses index 0 for the individual dimension
- Handles missing data by filling with NaN values
- Returns the same flat DataFrame structure as the TensorFlow version

### 3. Body Length Calculation in Existing Code

#### File: `deeplabcut/pose_estimation_pytorch/runners/train.py`
**Lines: 70-97** - SVL (body length) calculation for scaling

```python
def _compute_scale_factor_svl(animal_keypoints, bodyparts, skeletal_links_lengths):
    """Return mm->pixel scale using SVL if available; else None."""
    try:
        snout_idx = bodyparts.index('snout')
        tail1_idx = bodyparts.index('tail1')
    except ValueError:
        return None

    # GT SVL in pixels (requires both visible)
    if (animal_keypoints[snout_idx, 2] > 0.5) and (animal_keypoints[tail1_idx, 2] > 0.5):
        gt_svl_pix = float(torch.norm(
            torch.tensor(animal_keypoints[snout_idx, :2]) -
            torch.tensor(animal_keypoints[tail1_idx, :2])
        ).item())
    else:
        return None

    # Expected SVL (mm) from reference
    expected_svl_mm = None
    if ('snout', 'tail1') in skeletal_links_lengths:
        expected_svl_mm = float(skeletal_links_lengths[('snout', 'tail1')])
    elif ('tail1', 'snout') in skeletal_links_lengths:
        expected_svl_mm = float(skeletal_links_lengths[('tail1', 'snout')])

    if expected_svl_mm is None or expected_svl_mm <= 0:
        return None

    return gt_svl_pix / expected_svl_mm  # pixels per mm
```

**How it works:**
- SVL (Snout-Vent Length) is the body length metric used in this codebase
- Calculated as Euclidean distance between 'snout' and 'tail1' bodyparts
- Uses `torch.norm()` to compute the distance: `sqrt((x1-x2)^2 + (y1-y2)^2)`
- Requires both bodyparts to be visible (confidence > 0.5)
- Returns a scale factor (pixels per mm) by dividing pixel distance by expected mm distance

**Key insight for our analysis:**
- Body length = Euclidean distance between snout and tail1
- Formula: `sqrt((snout_x - tail1_x)^2 + (snout_y - tail1_y)^2)`
- This applies to both ground truth and predicted positions

### 4. Skeletal Data and Subject ID Extraction

#### File: `deeplabcut/pose_estimation_pytorch/data/dataset.py`
**Lines: 549-593** - Loading skeletal measurements per specimen

```python
def create_skeleton_dictionary(cfg, skeletal_csv_path):
    """Loads skeletal data from CSV and maps it to body part indices from the DLC config."""
    bodyparts = cfg['bodyparts']

    # Mapping from CSV columns to body part pairs
    link_mapping = {
        'svl': [('snout', 'tail1')],
        'head.length': [('snout', 'base_of_head')],
        'upper.forelimb': [('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')],
        'lower.forelimb': [('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist')],
        'upper.hindlimb': [('left_hip', 'left_knee'), ('right_hip', 'right_knee')],
        'lower.hindlimb': [('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')],
    }

    df = pd.read_csv(skeletal_csv_path)

    skeleton_dict = {}
    for _, row in df.iterrows():
        subject_id = str(row['lizard_id']).zfill(4)  # e.g., "0001"

        subject_links = []
        subject_lengths = []

        for col, pairs in link_mapping.items():
            if col in row and pd.notna(row[col]):
                link_length = row[col]  # Length in mm
                for bp1_name, bp2_name in pairs:
                    if bp1_name in bodyparts and bp2_name in bodyparts:
                        bp1_idx = bodyparts.index(bp1_name)
                        bp2_idx = bodyparts.index(bp2_name)
                        subject_links.append((bp1_idx, bp2_idx))
                        subject_lengths.append(link_length)

        skeleton_dict[subject_id] = {
            "links": subject_links,
            "link_lengths": subject_lengths,
        }

    return skeleton_dict
```

**How it works:**
- Loads a CSV file containing skeletal measurements for each specimen (lizard)
- CSV has columns: `lizard_id`, `svl`, `head.length`, `upper.forelimb`, etc.
- Each row represents one specimen with their reference body measurements in millimeters
- The 'svl' column contains the reference body length (snout to tail1 distance in mm)
- Creates a dictionary mapping subject_id (e.g., "0001") to their skeletal measurements

**Lines: 634-650** - Extracting subject ID from image path

```python
def _extract_subject_id(self, image_path: str) -> str:
    """
    Extract subject ID from image path.
    The subject ID is in the video directory name (first part when splitting on "_").
    For example: "0001_1_notes/image.jpg" -> "0001"
    """
    path_parts = Path(image_path).parts

    # Look for the video directory (should contain the subject ID)
    for part in reversed(path_parts):
        if part and '_' in part:
            # Split on underscore and take the first part as subject ID
            subject_id = part.split('_')[0]
            # Ensure it's a 4-digit zero-padded number
            if subject_id.isdigit():
                return subject_id.zfill(4)

    # Fallback: look for any 4-digit number in the path
    import re
    for part in reversed(path_parts):
        match = re.search(r'\b(\d{4})\b', part)
        if match:
            return match.group(1)

    return "0000"  # Default fallback
```

**How it works:**
- Parses the image path to extract the specimen/subject ID
- Assumes the video directory name starts with the subject ID (e.g., "0001_1_notes")
- Returns a 4-digit zero-padded string (e.g., "0001")
- This ID is used to look up the reference skeletal measurements for that specimen

**Relevance to our analysis:**
- The frame-level CSV contains `image_path` column
- We can extract subject_id from image_path to get reference body length measurements
- However, for this user story, we're calculating body length directly from positions, not using reference measurements

### 5. Confidence Score Usage in Existing Code

#### File: `deeplabcut/pose_estimation_pytorch/runners/train.py`
**Lines: 79, 200** - Confidence threshold checks

```python
# Checking if bodyparts are visible (confidence > 0.5)
if (animal_keypoints[snout_idx, 2] > 0.5) and (animal_keypoints[tail1_idx, 2] > 0.5):
    # Calculate body length
    ...

# Require adjacent bodypart to be visible
if float(animal_kpts[adj_idx, 2]) < 0.5:
    continue
```

**How it works:**
- Confidence scores are stored in the 3rd dimension of keypoint arrays (index 2)
- A threshold of 0.5 is commonly used to determine if a bodypart is "visible" or reliable
- If confidence < 0.5, the bodypart is often skipped in calculations

**Relevance to our analysis:**
- We should consider filtering frames where confidence scores are too low
- The 0.5 threshold is a reasonable starting point
- Our analysis will help determine better thresholds for body length calculations

### 6. Skeletal Constraint Loss Functions

#### File: `deeplabcut/pose_estimation_pytorch/runners/train.py`
**Lines: 406-573** - Skeletal constraint loss computation

```python
def compute_skeletal_constraint_loss(
    predicted_keypoints: torch.Tensor,
    skeletal_data: dict,
    bodyparts: list[str],
    device: torch.device,
    loss_weight: float = 1.0,
    radius_multiplier: float = 1.0,
) -> torch.Tensor:
    """Compute skeletal constraint loss based on expected limb lengths."""

    # Get indices for snout and tail1 for normalization
    try:
        snout_idx = bodyparts.index('snout')
        tail1_idx = bodyparts.index('tail1')
    except ValueError:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ... (processing each sample in batch)

    # Compute normalization factor (snout to tail1 distance)
    snout_pos = kpts[snout_idx, :2]  # [x, y]
    tail1_pos = kpts[tail1_idx, :2]  # [x, y]
    svl_distance = torch.norm(snout_pos - tail1_pos)

    if svl_distance < 1e-6:  # Avoid division by zero
        continue

    # ... (compute loss for each skeletal link)

    # Normalize distances by predicted SVL to make them scale-invariant
    normalized_predicted = predicted_distance / svl_distance
```

**How it works:**
- Uses predicted body length (SVL) to normalize limb length predictions
- Computes loss by comparing normalized predicted limb lengths to expected proportions
- Requires snout and tail1 to be visible to calculate SVL
- This is the loss function that will eventually use predicted body length in semi-supervised training

**Relevance to our analysis:**
- This loss function depends on accurate body length prediction
- Our analysis will determine when predicted body length is reliable enough to use this loss
- The confidence threshold we determine will gate whether this loss is applied

## Data Flow Summary

1. **Training & Evaluation**:
   - `deeplabcut.evaluate_network()` generates predictions for test set
   - `create_frame_level_results_tf/pytorch()` creates CSV with GT, predictions, and confidence scores
   - CSV saved as `{DLCscorer}-frame-level-results.csv` in evaluation folder

2. **Cross-Validation Aggregation** (User Story 02):
   - `run_cv.py` calls `evaluate_network()` for each fold/seed/landmark_set
   - Reads each frame-level CSV and adds metadata (fold, seed, experiment_id, etc.)
   - Appends to aggregate file: `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`

3. **Our Analysis** (User Story 03):
   - Load aggregate CSV file
   - Extract columns: `gt_snout_x`, `gt_snout_y`, `gt_tail1_x`, `gt_tail1_y`, `pred_snout_x`, `pred_snout_y`, `pred_tail1_x`, `pred_tail1_y`, `conf_snout`, `conf_tail1`
   - Calculate true body length: `sqrt((gt_snout_x - gt_tail1_x)^2 + (gt_snout_y - gt_tail1_y)^2)`
   - Calculate predicted body length: `sqrt((pred_snout_x - pred_tail1_x)^2 + (pred_snout_y - pred_tail1_y)^2)`
   - Calculate relative error: `abs(pred - true) / true`
   - Calculate mean confidence: `(conf_snout + conf_tail1) / 2`
   - Plot: relative_error vs mean_confidence

## Script Configuration Pattern

### Reference: `analyze_cv_results.py`
The implementation should follow the same configuration pattern used in `analyze_cv_results.py`, which uses constants at the top of the file for easy configuration.

**Example from `analyze_cv_results.py` (Lines 4-9):**
```python
# Configuration
EXPERIMENT_ID = 'll_0d2'
LANDMARK_SET_NAMES = ['all', 'truncated', 'non_truncated']

# Metric names to analyze
METRIC_NAMES = ['test rmse', 'test rmse_pcutoff', 'test mAP', 'test mAR']
```

**Key Benefits:**
- Easy to modify without changing code logic
- Clear separation of configuration from implementation
- Consistent with existing codebase patterns
- User can quickly change experiment ID or landmark set without searching through code

### Recommended Configuration Constants for User Story 03

```python
# Configuration
EXPERIMENT_ID = 'll_0d025'  # Experiment to analyze
LANDMARK_SET_NAME = 'all'   # Which landmark set to analyze ('all', 'truncated', etc.)

# Body length calculation settings
BODYPART_1 = 'snout'  # First bodypart for body length
BODYPART_2 = 'tail1'  # Second bodypart for body length

# Filtering thresholds
MIN_BODY_LENGTH_PIXELS = 10.0  # Minimum valid body length (avoid division by zero)
CONFIDENCE_THRESHOLD = 0.0      # Minimum confidence to include (0.0 = include all)

# Visualization settings
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 8)
PLOT_ALPHA = 0.5
MAX_RELATIVE_ERROR_DISPLAY = 1.0  # Cap at 100% for visualization

# Output settings
SAVE_COMPUTED_METRICS = True  # Whether to save computed metrics to CSV
OUTPUT_DIR = '.'              # Directory for output files
```

**Usage Pattern:**
```python
def main():
    # Construct input file path from configuration
    input_file = f'cv_frame_level_results_{EXPERIMENT_ID}_{LANDMARK_SET_NAME}.csv'

    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Frame-level results file not found: {input_file}\n"
            f"Please check that EXPERIMENT_ID='{EXPERIMENT_ID}' and "
            f"LANDMARK_SET_NAME='{LANDMARK_SET_NAME}' are correct."
        )

    # Load data
    print(f"Loading frame-level results from: {input_file}")
    df = pd.read_csv(input_file)

    # Build column names from configuration
    required_cols = [
        f'gt_{BODYPART_1}_x', f'gt_{BODYPART_1}_y',
        f'gt_{BODYPART_2}_x', f'gt_{BODYPART_2}_y',
        f'pred_{BODYPART_1}_x', f'pred_{BODYPART_1}_y',
        f'pred_{BODYPART_2}_x', f'pred_{BODYPART_2}_y',
        f'conf_{BODYPART_1}', f'conf_{BODYPART_2}',
    ]

    # Validate columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ... rest of implementation
```

## Implementation Approach

### Required Columns from Aggregate CSV
- `gt_snout_x`, `gt_snout_y`
- `gt_tail1_x`, `gt_tail1_y`
- `pred_snout_x`, `pred_snout_y`
- `pred_tail1_x`, `pred_tail1_y`
- `conf_snout`
- `conf_tail1`
- Optional metadata: `fold`, `seed`, `experiment_id`, `landmark_set_name`

### Calculation Steps (Vectorized with Pandas/NumPy)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('cv_frame_level_results_ll_0d025_all.csv')

# Calculate true body length (SVL)
df['true_body_length'] = np.sqrt(
    (df['gt_snout_x'] - df['gt_tail1_x'])**2 +
    (df['gt_snout_y'] - df['gt_tail1_y'])**2
)

# Calculate predicted body length
df['pred_body_length'] = np.sqrt(
    (df['pred_snout_x'] - df['pred_tail1_x'])**2 +
    (df['pred_snout_y'] - df['pred_tail1_y'])**2
)

# Calculate errors
df['absolute_error'] = np.abs(df['pred_body_length'] - df['true_body_length'])
df['relative_error'] = df['absolute_error'] / df['true_body_length']

# Calculate confidence metrics
df['mean_confidence'] = (df['conf_snout'] + df['conf_tail1']) / 2
df['min_confidence'] = df[['conf_snout', 'conf_tail1']].min(axis=1)

# Filter out invalid data
df_valid = df[df['true_body_length'] > 0].dropna(subset=['relative_error', 'mean_confidence'])

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(df_valid['mean_confidence'], df_valid['relative_error'] * 100, alpha=0.5)
plt.xlabel('Mean Confidence Score')
plt.ylabel('Relative Error (%)')
plt.title('Body Length Prediction Error vs Confidence')
plt.grid(True, alpha=0.3)
plt.savefig('body_length_error_vs_confidence.png', dpi=300)
```

## Edge Cases and Considerations

### 1. Missing or Invalid Data
- **Issue**: Some frames may have NaN values for ground truth or predictions
- **Handling**: Filter out rows where any required column is NaN
- **Code**: `df.dropna(subset=['gt_snout_x', 'gt_snout_y', 'gt_tail1_x', 'gt_tail1_y', ...])`

### 2. Zero or Very Small Body Length
- **Issue**: Division by zero when calculating relative error if true_body_length ≈ 0
- **Handling**: Filter out frames where `true_body_length < threshold` (e.g., 10 pixels)
- **Code**: `df[df['true_body_length'] > 10]`

### 3. Outliers
- **Issue**: Some frames may have extreme errors due to prediction failures
- **Handling**: Consider capping relative error at a maximum value (e.g., 100%) for visualization
- **Code**: `df['relative_error_capped'] = df['relative_error'].clip(upper=1.0)`

### 4. Different Landmark Sets
- **Issue**: Aggregate CSV may contain data from multiple landmark sets (e.g., 'all', 'truncated')
- **Handling**: Filter by `landmark_set_name` column if analyzing a specific set
- **Code**: `df[df['landmark_set_name'] == 'all']`

### 5. Bodypart Name Variations
- **Issue**: Column names depend on bodypart names in the config
- **Handling**: Validate that required columns exist before processing
- **Code**:
```python
required_cols = ['gt_snout_x', 'gt_snout_y', 'gt_tail1_x', 'gt_tail1_y',
                 'pred_snout_x', 'pred_snout_y', 'pred_tail1_x', 'pred_tail1_y',
                 'conf_snout', 'conf_tail1']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
```

## Impact on Existing Code

### No Direct Impact
This user story is purely analytical and does not modify any existing code. It:
- Reads existing CSV files generated by User Story 02
- Performs calculations and generates plots
- Does not affect training, evaluation, or cross-validation workflows

### Future Integration Points
The insights from this analysis will inform:
1. **Confidence thresholds** for semi-supervised training (future user story)
2. **Filtering criteria** for unlabeled data selection (future user story)
3. **Loss function gating** in skeletal constraint loss (future user story)

### Potential Extensions
- Add this analysis as an optional step in `run_cv.py` (controlled by a flag)
- Create a standalone script/notebook for ad-hoc analysis
- Generate analysis report automatically after CV completes


