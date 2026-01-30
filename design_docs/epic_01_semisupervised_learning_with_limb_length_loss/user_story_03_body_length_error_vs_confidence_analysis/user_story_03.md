# User Story: Body Length Error vs Confidence Analysis

## Overview
Create analysis tools to evaluate the relationship between body length prediction error and bodypart confidence scores using the frame-level aggregate results from cross-validation experiments.

## User Intent
When I have completed a cross-validation experiment with frame-level results aggregated (from User Story 02), I want to:
1. Load the frame-level aggregate CSV file
2. Calculate true body length (from ground truth positions)
3. Calculate predicted body length (from predicted positions)
4. Calculate relative error between true and predicted body length
5. Calculate mean confidence scores for the bodyparts used in body length calculation
6. Generate a plot showing the relationship between relative error and mean confidence score

This analysis will help determine confidence thresholds for when to trust predicted body length in semi-supervised training.

## Background
- **User Story 01** introduced frame-level validation output with ground truth positions, predicted positions, and confidence scores
- **User Story 02** aggregated these frame-level results across all CV folds/seeds into landmark-set-specific CSV files
- The epic goal requires determining when to trust predicted body length based on confidence scores
- Body length is calculated as the Euclidean distance between 'snout' and 'tail1' bodyparts (SVL - Snout-Vent Length)

## Requirements

### Primary Goal
Create a script/notebook that:
1. Reads the aggregated frame-level CSV file (e.g., `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`)
2. Computes body length metrics for each frame
3. Generates visualization of error vs confidence relationship

### Functional Requirements

#### 1. Data Loading
- Load the frame-level aggregate CSV file
- Filter data by experiment_id and landmark_set_name if needed
- Handle missing/NaN values appropriately

#### 2. Body Length Calculation
For each frame in the dataset:
- **True Body Length**: Calculate Euclidean distance between ground truth 'snout' and 'tail1' positions
  - Formula: `sqrt((gt_snout_x - gt_tail1_x)^2 + (gt_snout_y - gt_tail1_y)^2)`
- **Predicted Body Length**: Calculate Euclidean distance between predicted 'snout' and 'tail1' positions
  - Formula: `sqrt((pred_snout_x - pred_tail1_x)^2 + (pred_snout_y - pred_tail1_y)^2)`

#### 3. Error Metrics
- **Absolute Error**: `abs(predicted_body_length - true_body_length)`
- **Relative Error**: `abs(predicted_body_length - true_body_length) / true_body_length`
  - Express as percentage: `relative_error * 100`

#### 4. Confidence Score Aggregation
- Extract confidence scores for 'snout' and 'tail1' bodyparts
- Calculate mean confidence: `(conf_snout + conf_tail1) / 2`
- Alternative: minimum confidence: `min(conf_snout, conf_tail1)`

#### 5. Visualization
Generate a scatter plot with:
- **X-axis**: Mean confidence score (0 to 1)
- **Y-axis**: Relative error (percentage)
- **Points**: Each frame as a data point
- **Optional enhancements**:
  - Color-code by fold or seed
  - Add trend line or moving average
  - Add horizontal lines for error thresholds (e.g., 5%, 10%, 20%)
  - Add vertical lines for confidence thresholds (e.g., 0.5, 0.7, 0.9)

### Output Format

#### CSV Output (Optional)
Save computed metrics to a new CSV file:
- Filename: `body_length_analysis_{experiment_id}_{landmark_set_name}.csv`
- Columns:
  - All original columns from frame-level CSV
  - `true_body_length`: Ground truth SVL in pixels
  - `pred_body_length`: Predicted SVL in pixels
  - `absolute_error`: Absolute error in pixels
  - `relative_error`: Relative error as decimal (0.05 = 5%)
  - `mean_confidence`: Mean of snout and tail1 confidence scores
  - `min_confidence`: Minimum of snout and tail1 confidence scores

#### Plot Output
- Filename: `body_length_error_vs_confidence_{experiment_id}_{landmark_set_name}.png`
- Format: PNG with high DPI (300)
- Size: 10x8 inches or similar

## Use Cases
- Determine minimum confidence threshold for semi-supervised training
- Identify frames with high confidence but high error (potential labeling issues)
- Identify frames with low confidence but low error (model is uncertain but correct)
- Understand the relationship between prediction confidence and accuracy
- Guide hyperparameter selection for confidence-based filtering

## Scope

### In Scope
- Analysis of existing frame-level aggregate CSV files
- Body length calculation using 'snout' and 'tail1' bodyparts
- Scatter plot visualization
- Single-animal projects
- Both 'all' and 'truncated' landmark sets

### Out of Scope (for initial implementation)
- Multi-animal projects
- Other body length definitions (e.g., total length including tail)
- Interactive visualizations (Plotly, Bokeh)
- Statistical modeling (regression, confidence intervals)
- Automated threshold selection algorithms
- Integration into run_cv.py (this is a separate analysis step)

## Acceptance Criteria

### AC1: Load Frame-Level Aggregate CSV
**Given** a frame-level aggregate CSV file exists at `cv_frame_level_results_ll_0d025_all.csv`
**When** I run the analysis script
**Then** the CSV is loaded into a pandas DataFrame
**And** the DataFrame contains the expected columns (fold, seed, experiment_id, landmark_set_name, shuffle_num, frame_index, image_path, gt_snout_x, gt_snout_y, pred_snout_x, pred_snout_y, conf_snout, gt_tail1_x, gt_tail1_y, pred_tail1_x, pred_tail1_y, conf_tail1, ...)

### AC2: Calculate True Body Length
**Given** a frame with gt_snout at (100, 200) and gt_tail1 at (100, 300)
**When** true body length is calculated
**Then** the result is 100.0 pixels (Euclidean distance)

### AC3: Calculate Predicted Body Length
**Given** a frame with pred_snout at (101, 199) and pred_tail1 at (101, 301)
**When** predicted body length is calculated
**Then** the result is approximately 102.0 pixels

### AC4: Calculate Relative Error
**Given** true_body_length = 100.0 and pred_body_length = 105.0
**When** relative error is calculated
**Then** the result is 0.05 (5%)

### AC5: Calculate Mean Confidence
**Given** conf_snout = 0.95 and conf_tail1 = 0.85
**When** mean confidence is calculated
**Then** the result is 0.90

### AC6: Handle Missing Values
**Given** a frame where gt_snout_x is NaN
**When** body length is calculated
**Then** the true_body_length is NaN
**And** the frame is excluded from the plot (or handled appropriately)

### AC7: Generate Scatter Plot
**Given** a DataFrame with computed metrics
**When** the plot is generated
**Then** a scatter plot is created with mean_confidence on x-axis and relative_error on y-axis
**And** the plot is saved to a PNG file

### AC8: Plot Contains All Frames
**Given** 1000 frames in the aggregate CSV (5 folds × 2 seeds × 100 test frames)
**When** the plot is generated
**Then** the plot contains up to 1000 data points (excluding frames with missing values)

## Technical Considerations

### Script Configuration Pattern
The implementation script should follow the configuration pattern used in `analyze_cv_results.py`:
- **Constants at top of file**: All configuration parameters (experiment ID, landmark set, thresholds, etc.) should be defined as constants at the top of the script
- **Easy modification**: Users should be able to change experiment settings by editing only the constants section
- **Clear separation**: Configuration should be separated from implementation logic
- **Validation**: Script should validate that input files exist and provide helpful error messages

**Example configuration section:**
```python
# Configuration
EXPERIMENT_ID = 'll_0d025'
LANDMARK_SET_NAME = 'all'
BODYPART_1 = 'snout'
BODYPART_2 = 'tail1'
MIN_BODY_LENGTH_PIXELS = 10.0
PLOT_DPI = 300
```

### Dependencies
- pandas: For data loading and manipulation
- numpy: For numerical calculations
- matplotlib or seaborn: For plotting

### Performance
- The aggregate CSV may contain thousands of rows (e.g., 5 folds × 2 seeds × 100 frames = 1000 rows)
- Calculations should be vectorized using pandas/numpy for efficiency

### Data Quality
- Some frames may have missing ground truth (NaN values)
- Some predictions may have very low confidence scores
- Handle edge cases where true_body_length is 0 or very small (avoid division by zero)

### Bodypart Names
- Assumes bodyparts are named 'snout' and 'tail1' (consistent with the codebase)
- Should validate that these columns exist in the CSV before processing

