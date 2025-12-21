# User Story: Frame-Level Validation Output

## Overview
Enhance the `deeplabcut.evaluate_network()` function to generate detailed frame-level validation data in CSV format.

## User Intent
When running validation via `deeplabcut.evaluate_network()`, I want to create an additional CSV file that provides detailed, frame-by-frame comparison data for the test set.

## Requirements

### Primary Goal
Generate a CSV file during network evaluation that contains frame-level data for each image in the test set.

### CSV File Contents
For each frame in the test set, the CSV should include:

1. **Ground Truth Positions**: X and Y coordinates for all bodyparts
   - Columns: `gt_<bodypart>_x`, `gt_<bodypart>_y` for each bodypart
   
2. **Predicted Positions**: X and Y coordinates for all bodyparts
   - Columns: `pred_<bodypart>_x`, `pred_<bodypart>_y` for each bodypart
   
3. **Confidence Scores**: Likelihood/confidence values for all bodyparts
   - Columns: `conf_<bodypart>` for each bodypart

### Additional Metadata
The CSV should also include:
- Frame identifier (image path/name)
- Frame index in the dataset
- Whether the frame is in train or test set (focus on test set initially)

## Expected Output Format

```csv
frame_index,image_path,gt_nose_x,gt_nose_y,pred_nose_x,pred_nose_y,conf_nose,gt_left_ear_x,gt_left_ear_y,pred_left_ear_x,pred_left_ear_y,conf_left_ear,...
0,path/to/image1.png,100.5,200.3,101.2,199.8,0.95,120.1,195.4,121.0,194.9,0.92,...
1,path/to/image2.png,105.2,198.7,104.8,199.1,0.88,125.3,193.2,124.9,193.5,0.91,...
...
```

## Use Cases
- Detailed error analysis at the frame level
- Identifying specific frames with high/low prediction accuracy
- Debugging model performance on individual images
- Creating custom visualizations and analyses
- Statistical analysis of prediction quality across the test set

## Scope
- **In Scope**: 
  - Test set frames
  - Single-animal projects (initial implementation)
  - Both TensorFlow and PyTorch backends
  - CSV output format
  
- **Out of Scope** (for initial implementation):
  - Training set frames (can be added later)
  - Multi-animal projects (requires different data structure)
  - Other output formats (JSON, HDF5, etc.)

## Acceptance Criteria

### AC1: CSV File Generation for TensorFlow Backend
**Given** a DeepLabCut project with a trained TensorFlow model and labeled test data
**When** I call `deeplabcut.evaluate_network(config, save_frame_level_results=True)`
**Then** a CSV file named `{DLCscorer}-frame-level-results.csv` is created in the evaluation-results folder
**And** the CSV contains one row for each frame in the test set only (not training frames)

### AC2: CSV File Generation for PyTorch Backend
**Given** a DeepLabCut project with a trained PyTorch model and labeled test data
**When** I call `deeplabcut.evaluate_network(config, save_frame_level_results=True)`
**Then** a CSV file named `{DLCscorer}-frame-level-results.csv` is created in the evaluation-results-pytorch folder
**And** the CSV contains one row for each frame in the test set only (not training frames)

### AC3: CSV Contains Required Columns
**Given** a project with bodyparts: `['nose', 'left_ear', 'right_ear']`
**When** the frame-level CSV is generated
**Then** the CSV contains the following columns:
- `frame_index` (integer, 0-based index within test set)
- `image_path` (string, relative path from project root)
- For each bodypart: `gt_{bodypart}_x`, `gt_{bodypart}_y`, `pred_{bodypart}_x`, `pred_{bodypart}_y`, `conf_{bodypart}`

**Example columns**: `frame_index`, `image_path`, `gt_nose_x`, `gt_nose_y`, `pred_nose_x`, `pred_nose_y`, `conf_nose`, `gt_left_ear_x`, `gt_left_ear_y`, `pred_left_ear_x`, `pred_left_ear_y`, `conf_left_ear`, ...

### AC4: Ground Truth Data Accuracy
**Given** a test frame with ground truth annotation for 'nose' at coordinates (100.5, 200.3)
**When** the frame-level CSV is generated
**Then** the row for that frame contains `gt_nose_x=100.5` and `gt_nose_y=200.3`
**And** the ground truth values match the values in the original CollectedData HDF5 file

### AC5: Prediction Data Accuracy
**Given** a test frame where the model predicts 'nose' at coordinates (101.2, 199.8) with confidence 0.95
**When** the frame-level CSV is generated
**Then** the row for that frame contains `pred_nose_x=101.2`, `pred_nose_y=199.8`, and `conf_nose=0.95`
**And** the prediction values match the values in the evaluation HDF5 predictions file

### AC6: Test Set Filtering
**Given** a project with 100 labeled frames split into 80% training (80 frames) and 20% test (20 frames)
**When** the frame-level CSV is generated
**Then** the CSV contains exactly 20 rows (one per test frame)
**And** the CSV does NOT contain any training frames
**And** the `frame_index` column ranges from 0 to 19

### AC7: Handling Missing/NaN Values
**Given** a test frame where ground truth for 'tail' is missing (NaN)
**When** the frame-level CSV is generated
**Then** the row for that frame contains NaN (or empty) values for `gt_tail_x` and `gt_tail_y`
**And** the CSV file is still valid and can be loaded by pandas

### AC8: Backward Compatibility - Default Behavior
**Given** existing code that calls `deeplabcut.evaluate_network(config)` without the new parameter
**When** the evaluation runs
**Then** the frame-level CSV is generated by default (parameter defaults to `True`)
**And** all existing evaluation outputs (summary CSV, plots, HDF5) are still generated
**And** no errors or warnings are raised

### AC9: Backward Compatibility - Opt-Out
**Given** a user who does not want the frame-level CSV
**When** I call `deeplabcut.evaluate_network(config, save_frame_level_results=False)`
**Then** the frame-level CSV is NOT generated
**And** all other evaluation outputs work normally

### AC10: File Location and Naming - TensorFlow
**Given** a TensorFlow project with model name `DLC_resnet50_taskJan30shuffle1_50000`
**When** evaluation completes
**Then** the CSV is saved at:
`{project_path}/evaluation-results/iteration-{train_frac}-shuffle-{shuffle}/DLC_resnet50_taskJan30shuffle1_50000-frame-level-results.csv`
**And** the file is in the same directory as `DLC_resnet50_taskJan30shuffle1_50000.h5` and `DLC_resnet50_taskJan30shuffle1_50000-results.csv`

### AC11: File Location and Naming - PyTorch
**Given** a PyTorch project with model name `DLC_dlcrnet_ms5_taskJan30shuffle1_50000`
**When** evaluation completes
**Then** the CSV is saved at:
`{project_path}/evaluation-results-pytorch/iteration-{train_frac}-shuffle-{shuffle}/DLC_dlcrnet_ms5_taskJan30shuffle1_50000-frame-level-results.csv`
**And** the file is in the same directory as other PyTorch evaluation outputs

### AC12: Console Output Confirmation
**Given** frame-level results are being saved
**When** the CSV export completes
**Then** a confirmation message is printed to console:
`"Frame-level results saved to: {filename}"`
**And** the message includes the filename (not full path)

### AC13: Multi-Shuffle Support
**Given** a project evaluated with multiple shuffles `[1, 2, 3]`
**When** `deeplabcut.evaluate_network(config, Shuffles=[1, 2, 3])` is called
**Then** a separate frame-level CSV is created for each shuffle
**And** each CSV is in its respective shuffle folder

### AC14: Multi-Training Fraction Support
**Given** a project evaluated with multiple training fractions `[0.5, 0.8, 0.95]`
**When** `deeplabcut.evaluate_network(config, TrainingFractions=[0.5, 0.8, 0.95])` is called
**Then** a separate frame-level CSV is created for each training fraction
**And** each CSV contains only the test frames for that specific train/test split

### AC15: Empty Test Set Handling
**Given** a project where the test set is empty (all frames used for training)
**When** the frame-level CSV generation is attempted
**Then** an empty CSV file is created with headers only
**And** no errors are raised

### AC16: Integration with per_keypoint_evaluation
**Given** a user calls `deeplabcut.evaluate_network(config, per_keypoint_evaluation=True, save_frame_level_results=True)`
**When** evaluation completes
**Then** both the keypoint-results CSV and frame-level-results CSV are generated
**And** both files are saved in the same evaluation folder

## Technical Considerations
- File should be saved in the same evaluation-results folder as other evaluation outputs
- Naming convention: `{DLCscorer}-frame-level-results.csv` (follows existing pattern)
- Should handle missing/NaN values appropriately
- Should respect the existing test/train split indices
- Should work with the existing data structures (DataMachine, Data for TF; predictions dict for PyTorch)
- CSV should use `index=False` when saving (no row index column)
- Image paths should be relative to project root for portability

