# Implementation Plan: Frame-Level Validation Output

## Overview
This document outlines the detailed implementation plan for adding frame-level validation CSV output to DeepLabCut's `evaluate_network()` function. The implementation will support both TensorFlow and PyTorch backends for single-animal projects.

## Goals
1. Generate a CSV file containing frame-by-frame ground truth, predictions, and confidence scores **for test set frames only**
2. Maintain backward compatibility - no breaking changes
3. Follow existing patterns (similar to `per_keypoint_evaluation`)
4. Support both TensorFlow and PyTorch backends
5. Focus on single-animal projects initially

**Important**: The CSV output will contain **ONLY test set data**, not training data. This is the standard evaluation practice in DeepLabCut.

---

## Implementation Strategy

### Phase 1: Add Helper Functions
Create utility functions to convert existing data structures to frame-level CSV format.

### Phase 2: Modify TensorFlow Backend
Update `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`

### Phase 3: Modify PyTorch Backend
Update `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`

### Phase 4: Testing & Documentation
Add tests and update user documentation

---

## Detailed Implementation Plan

## 1. Helper Functions

### 1.1 TensorFlow Helper Function
**New Function**: `create_frame_level_results_tf()`
**Location**: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py` (add near other helper functions, around line 45)

**Purpose**: Convert TensorFlow DataFrames to frame-level CSV format

**Signature**:
```python
def create_frame_level_results_tf(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    test_indices: List[int],
    scorer_gt: str,
    scorer_pred: str,
) -> pd.DataFrame:
    """
    Creates a frame-level results DataFrame for test set evaluation.
    
    Args:
        ground_truth: DataFrame with ground truth annotations (MultiIndex columns)
        predictions: DataFrame with model predictions (MultiIndex columns)
        test_indices: List of indices for test set frames
        scorer_gt: Name of the ground truth scorer (human annotator)
        scorer_pred: Name of the prediction scorer (DLC model)
    
    Returns:
        DataFrame with columns: frame_index, image_path, gt_{bodypart}_x, 
        gt_{bodypart}_y, pred_{bodypart}_x, pred_{bodypart}_y, conf_{bodypart}
    """
```

**Implementation Details**:
- **CRITICAL**: Filter data to **test set only** using `test_indices` - do NOT include training data
- Extract bodypart names from DataFrame columns
- Iterate through test frames and build row-by-row data
- Handle NaN values appropriately
- Return DataFrame with flat column structure (no MultiIndex)

**Data Access Pattern**:
```python
# Ground truth: ground_truth[scorer_gt][bodypart]["x"]
# Predictions: predictions[scorer_pred][bodypart]["x"], ["y"], ["likelihood"]
```

---

### 1.2 PyTorch Helper Function
**New Function**: `create_frame_level_results_pytorch()`
**Location**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py` (add near other helper functions, around line 100)

**Purpose**: Convert PyTorch dictionaries to frame-level CSV format

**Signature**:
```python
def create_frame_level_results_pytorch(
    ground_truth: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    bodyparts: list[str],
    image_paths: list[str],
) -> pd.DataFrame:
    """
    Creates a frame-level results DataFrame for test set evaluation.
    
    Args:
        ground_truth: Dict mapping image paths to GT arrays [n_individuals, n_bodyparts, 3]
        predictions: Dict mapping image paths to pred arrays [n_individuals, n_bodyparts, 3]
        bodyparts: List of bodypart names
        image_paths: List of image paths (for ordering)
    
    Returns:
        DataFrame with columns: frame_index, image_path, gt_{bodypart}_x, 
        gt_{bodypart}_y, pred_{bodypart}_x, pred_{bodypart}_y, conf_{bodypart}
    """
```

**Implementation Details**:
- **CRITICAL**: Only process **test set images** - the `ground_truth` and `predictions` dicts passed in should already be filtered to test set only
- Iterate through image_paths to maintain consistent ordering
- For single-animal: extract first individual (index 0)
- Extract x, y coordinates and confidence from arrays
- Handle missing data (NaN values)
- Return DataFrame with flat column structure

**Data Access Pattern**:
```python
# Ground truth: ground_truth[image_path][0, bodypart_idx, 0:2]  # x, y
# Predictions: predictions[image_path][0, bodypart_idx, :]  # x, y, confidence
```

---

## 2. TensorFlow Backend Modifications

### 2.1 Add Parameter to `evaluate_network()`
**File**: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`
**Line**: ~543 (in function signature)

**Change**:
```python
def evaluate_network(
    config,
    Shuffles=[1],
    trainingsetindex=0,
    plotting=False,
    show_errors=True,
    comparisonbodyparts="all",
    gputouse=None,
    rescale=False,
    modelprefix="",
    per_keypoint_evaluation: bool = False,
    snapshots_to_evaluate: List[str] = None,
    save_frame_level_results: bool = True,  # NEW PARAMETER
):
```

**Documentation Addition** (around line 600):
```python
save_frame_level_results: bool, default=True
    Save frame-by-frame ground truth, predictions, and confidence scores to a CSV file
    named {model_name}-frame-level-results.csv in the evaluation-results folder.
    The CSV contains detailed data for each frame in the test set.
```

### 2.2 Insert CSV Export Logic
**File**: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`
**Location**: After line 936 (after `per_keypoint_evaluation` block)

**Code to Insert**:
```python
                        # Save frame-level results CSV
                        if save_frame_level_results:
                            df_frame_level = create_frame_level_results_tf(
                                ground_truth=Data,
                                predictions=DataMachine,
                                test_indices=testIndices,
                                scorer_gt=cfg["scorer"],
                                scorer_pred=DLCscorer,
                            )
                            frame_level_filename = DLCscorer + "-frame-level-results.csv"
                            frame_level_path = Path(evaluationfolder) / frame_level_filename
                            df_frame_level.to_csv(frame_level_path, index=False)
                            print(f"Frame-level results saved to: {frame_level_filename}")
```

**Available Variables at This Point**:
- `Data`: Ground truth DataFrame
- `DataMachine`: Predictions DataFrame  
- `testIndices`: Test set indices
- `trainIndices`: Training set indices
- `evaluationfolder`: Output directory path
- `DLCscorer`: Model name string
- `cfg["scorer"]`: Human annotator name

---

## 3. PyTorch Backend Modifications

### 3.1 Add Parameter to `evaluate_network()`
**File**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`
**Line**: ~685 (in function signature)

**Change**:
```python
def evaluate_network(
    config: str | Path,
    shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    snapshotindex: int | str | None = None,
    device: str | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    transform: A.Compose = None,
    snapshots_to_evaluate: list[str] | None = None,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    modelprefix: str = "",
    detector_snapshot_index: int | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
    save_frame_level_results: bool = True,  # NEW PARAMETER
) -> None:
```

**Documentation Addition** (around line 730):
```python
save_frame_level_results: bool, default=True
    Save frame-by-frame ground truth, predictions, and confidence scores to a CSV file
    named {model_name}-frame-level-results.csv in the evaluation-results folder.
    The CSV contains detailed data for each frame in the test set.
```

### 3.2 Pass Parameter to `evaluate_snapshot()`
**File**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`
**Location**: Around line 820 (in the call to `evaluate_snapshot`)

**Change**: Add `save_frame_level_results` parameter to the function call

### 3.3 Update `evaluate_snapshot()` Signature
**File**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`
**Line**: ~469 (in function signature)

**Change**:
```python
def evaluate_snapshot(
    cfg: dict,
    loader: DLCLoader,
    snapshot: Snapshot,
    scorer: str,
    transform: A.Compose | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    detector_snapshot: Snapshot | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
    save_frame_level_results: bool = True,  # NEW PARAMETER
) -> pd.DataFrame:
```

### 3.4 Insert CSV Export Logic
**File**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`
**Location**: After line 640 (after `per_keypoint_evaluation` block)

**Code to Insert**:
```python
    # Save frame-level results CSV
    if save_frame_level_results:
        # Get test set ground truth and predictions
        gt_test = loader.ground_truth_keypoints("test")
        pred_test = {
            img: pred["bodyparts"]
            for img, pred in predictions["test"].items()
        }

        # Get ordered list of test image paths
        test_image_paths = list(gt_test.keys())

        df_frame_level = create_frame_level_results_pytorch(
            ground_truth=gt_test,
            predictions=pred_test,
            bodyparts=eval_parameters.bodyparts,
            image_paths=test_image_paths,
        )

        frame_level_path = output_filename.with_name(
            output_filename.stem + "-frame-level-results.csv"
        )
        df_frame_level.to_csv(frame_level_path, index=False)
        print(f"Frame-level results saved to: {frame_level_path.name}")
```

**Available Variables at This Point**:
- `predictions`: Dict with "train" and "test" keys, each containing prediction dicts
- `loader`: DLCLoader object with `ground_truth_keypoints()` method
- `eval_parameters`: PoseDatasetParameters with bodyparts list
- `output_filename`: Path object for output files
- `scorer`: Model name string

---

## 4. Output CSV Format

### 4.1 Column Structure
The CSV will have the following columns (example for 3 bodyparts: nose, left_ear, right_ear).

**NOTE**: This CSV contains **ONLY test set frames**, not training frames.

```
frame_index,image_path,gt_nose_x,gt_nose_y,pred_nose_x,pred_nose_y,conf_nose,gt_left_ear_x,gt_left_ear_y,pred_left_ear_x,pred_left_ear_y,conf_left_ear,gt_right_ear_x,gt_right_ear_y,pred_right_ear_x,pred_right_ear_y,conf_right_ear
0,labeled-data/video1/img001.png,100.5,200.3,101.2,199.8,0.95,120.1,195.4,121.0,194.9,0.92,80.3,196.1,79.8,195.5,0.89
1,labeled-data/video1/img005.png,105.2,198.7,104.8,199.1,0.88,125.3,193.2,124.9,193.5,0.91,85.1,194.3,84.7,193.9,0.87
...
```

Each row represents one frame from the **test set only**.

### 4.2 Column Naming Convention
- `frame_index`: Integer index (0-based) within the test set
- `image_path`: Relative path to the image from project root
- `gt_{bodypart}_x`: Ground truth x-coordinate for bodypart
- `gt_{bodypart}_y`: Ground truth y-coordinate for bodypart
- `pred_{bodypart}_x`: Predicted x-coordinate for bodypart
- `pred_{bodypart}_y`: Predicted y-coordinate for bodypart
- `conf_{bodypart}`: Confidence/likelihood score for bodypart (0.0 to 1.0)

### 4.3 Handling Missing Data
- NaN values in ground truth: Keep as NaN in CSV
- NaN values in predictions: Keep as NaN in CSV
- Missing confidence scores: Use NaN or 0.0 (TensorFlow always has likelihood)

---

## 5. File Locations and Naming

### 5.1 Output Directory
**TensorFlow**: `{project_path}/evaluation-results/iteration-{train_frac}-shuffle-{shuffle}/`
**PyTorch**: `{project_path}/evaluation-results-pytorch/iteration-{train_frac}-shuffle-{shuffle}/`

### 5.2 Filename Convention
**Pattern**: `{DLCscorer}-frame-level-results.csv`
**Example**: `DLC_resnet50_reaching-taskJan30shuffle1_50000-frame-level-results.csv`

This follows the existing pattern used for:
- `{DLCscorer}.h5` (predictions HDF5)
- `{DLCscorer}-results.csv` (summary metrics)
- `{DLCscorer}-keypoint-results.csv` (per-keypoint RMSE)

---

## 6. Files to Modify

### 6.1 Core Implementation Files

| File | Purpose | Changes |
|------|---------|---------|
| `deeplabcut/pose_estimation_tensorflow/core/evaluate.py` | TensorFlow evaluation | Add helper function, parameter, and CSV export logic |
| `deeplabcut/pose_estimation_pytorch/apis/evaluation.py` | PyTorch evaluation | Add helper function, parameter, and CSV export logic |

### 6.2 No New Files Required
All functionality will be added to existing files to maintain consistency with the codebase structure.

---

## 7. Implementation Details

### 7.1 TensorFlow Helper Function Implementation

**Key Considerations**:
1. **MultiIndex DataFrame Access**: Ground truth and predictions use MultiIndex columns
   - Level 0: scorer name
   - Level 1: bodypart name
   - Level 2: coordinate ("x", "y", "likelihood")

2. **Index Handling**: `test_indices` are integer positions in the DataFrame
   - Use `.iloc[test_indices]` to filter test set

3. **Image Path Extraction**: DataFrame index contains image paths
   - May be tuples (e.g., `('labeled-data', 'video1', 'img001.png')`)
   - Convert to string path using `os.path.join()` or `Path()`

4. **Bodypart Ordering**: Extract from `ground_truth[scorer_gt].columns.get_level_values('bodyparts').unique()`

**Pseudocode**:
```python
def create_frame_level_results_tf(ground_truth, predictions, test_indices, scorer_gt, scorer_pred):
    # Get bodyparts list
    bodyparts = ground_truth[scorer_gt].columns.get_level_values('bodyparts').unique().tolist()

    # Filter to test set
    gt_test = ground_truth.iloc[test_indices]
    pred_test = predictions.iloc[test_indices]

    # Build column names
    columns = ['frame_index', 'image_path']
    for bp in bodyparts:
        columns.extend([f'gt_{bp}_x', f'gt_{bp}_y', f'pred_{bp}_x', f'pred_{bp}_y', f'conf_{bp}'])

    # Build rows
    rows = []
    for idx, (frame_idx, image_path) in enumerate(zip(test_indices, gt_test.index)):
        row = [idx, str(Path(*image_path)) if isinstance(image_path, tuple) else str(image_path)]

        for bp in bodyparts:
            # Ground truth
            gt_x = gt_test.loc[image_path, (scorer_gt, bp, 'x')]
            gt_y = gt_test.loc[image_path, (scorer_gt, bp, 'y')]

            # Predictions
            pred_x = pred_test.loc[image_path, (scorer_pred, bp, 'x')]
            pred_y = pred_test.loc[image_path, (scorer_pred, bp, 'y')]
            conf = pred_test.loc[image_path, (scorer_pred, bp, 'likelihood')]

            row.extend([gt_x, gt_y, pred_x, pred_y, conf])

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
```

### 7.2 PyTorch Helper Function Implementation

**Key Considerations**:
1. **Array Structure**: Arrays are shape `[n_individuals, n_bodyparts, 3]`
   - For single-animal: `n_individuals = 1`, so use index `[0, :, :]`
   - Last dimension: `[x, y, confidence/visibility]`

2. **Image Path Ordering**: Maintain consistent ordering across GT and predictions

3. **Missing Data**: Handle cases where image might be in GT but not predictions (or vice versa)

**Pseudocode**:
```python
def create_frame_level_results_pytorch(ground_truth, predictions, bodyparts, image_paths):
    columns = ['frame_index', 'image_path']
    for bp in bodyparts:
        columns.extend([f'gt_{bp}_x', f'gt_{bp}_y', f'pred_{bp}_x', f'pred_{bp}_y', f'conf_{bp}'])

    rows = []
    for idx, image_path in enumerate(image_paths):
        row = [idx, image_path]

        # Get arrays for this image (single animal: use index 0)
        gt_array = ground_truth.get(image_path, np.full((1, len(bodyparts), 3), np.nan))
        pred_array = predictions.get(image_path, np.full((1, len(bodyparts), 3), np.nan))

        for bp_idx, bp in enumerate(bodyparts):
            # Ground truth (x, y)
            gt_x = gt_array[0, bp_idx, 0]
            gt_y = gt_array[0, bp_idx, 1]

            # Predictions (x, y, confidence)
            pred_x = pred_array[0, bp_idx, 0]
            pred_y = pred_array[0, bp_idx, 1]
            conf = pred_array[0, bp_idx, 2]

            row.extend([gt_x, gt_y, pred_x, pred_y, conf])

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests
**File**: Create `tests/test_frame_level_results.py`

**Test Cases**:
1. Test TensorFlow helper function with mock DataFrames
2. Test PyTorch helper function with mock dictionaries
3. Test column naming is correct
4. Test handling of NaN values
5. Test filtering to test set only

### 8.2 Integration Tests
**Approach**: Use existing test projects

1. Run `evaluate_network()` on TensorFlow test project
2. Verify CSV file is created
3. Verify CSV has correct number of rows (matches test set size)
4. Verify CSV has correct columns
5. Verify data matches existing HDF5 predictions
6. Repeat for PyTorch backend

### 8.3 Manual Testing
1. Test with real DeepLabCut project (single-animal)
2. Verify CSV can be loaded and analyzed
3. Verify backward compatibility (existing code still works)

---

## 9. Edge Cases and Error Handling

### 9.1 Edge Cases to Handle
1. **Empty test set**: Should create empty CSV with headers
2. **Missing bodyparts**: Handle gracefully with NaN
3. **Tuple vs string image paths**: Convert consistently to strings
4. **Multi-animal projects**: Out of scope for Phase 1, but document limitation

### 9.2 Error Messages
- If `save_frame_level_results=True` but data is unavailable, log warning and skip
- No errors should break existing evaluation workflow

---

## 10. Documentation Updates

### 10.1 Docstring Updates
- Update `evaluate_network()` docstrings in both backends
- Add docstrings to new helper functions

### 10.2 User Documentation
**File**: Update relevant user guide sections

**Content to Add**:
- Description of new CSV output
- Example of how to load and analyze the CSV
- Use cases for frame-level analysis

---

## 11. Implementation Checklist

### Phase 1: Helper Functions
- [ ] Implement `create_frame_level_results_tf()` in TensorFlow evaluate.py
- [ ] Implement `create_frame_level_results_pytorch()` in PyTorch evaluation.py
- [ ] Add comprehensive docstrings

### Phase 2: TensorFlow Backend
- [ ] Add `save_frame_level_results` parameter to `evaluate_network()`
- [ ] Update docstring with parameter description
- [ ] Insert CSV export logic after line 936
- [ ] Test with sample project

### Phase 3: PyTorch Backend
- [ ] Add `save_frame_level_results` parameter to `evaluate_network()`
- [ ] Add parameter to `evaluate_snapshot()`
- [ ] Update docstrings with parameter description
- [ ] Insert CSV export logic after line 640
- [ ] Test with sample project

### Phase 4: Testing
- [ ] Write unit tests for helper functions
- [ ] Write integration tests
- [ ] Manual testing with real projects
- [ ] Verify backward compatibility

### Phase 5: Documentation
- [ ] Update function docstrings
- [ ] Add user guide section
- [ ] Create example notebook (optional)

---

## 12. Future Enhancements (Out of Scope for Phase 1)

1. **Multi-animal support**: Extend to handle multiple individuals
   - Column naming: `gt_{individual}_{bodypart}_x`
   - Requires different data structure

2. **Training set output**: Add option to save training set results in addition to test set
   - **Current implementation**: Test set ONLY
   - **Future enhancement**: Parameter like `save_frame_level_results="test"` or `"train"` or `"both"`
   - **Note**: Training set output is generally not needed for validation purposes

3. **Additional formats**: Support JSON, HDF5, or Parquet output
   - Parameter: `frame_level_format="csv"` or `"json"` or `"hdf5"`

4. **Computed metrics per frame**: Add RMSE per frame to CSV
   - Additional columns: `rmse_{bodypart}`, `rmse_total`

5. **Filtering options**: Allow filtering by confidence threshold
   - Only include frames/bodyparts above certain confidence

---

## 13. Summary

This implementation plan provides a comprehensive roadmap for adding frame-level validation output to DeepLabCut. The approach:

1. **Follows existing patterns**: Mirrors `per_keypoint_evaluation` implementation
2. **Maintains compatibility**: No breaking changes, optional parameter with default `True`
3. **Supports both backends**: Separate implementations for TensorFlow and PyTorch
4. **Well-tested**: Comprehensive testing strategy
5. **Documented**: Clear docstrings and user documentation
6. **Test set only**: CSV output contains **ONLY test set frames**, which is the standard for validation/evaluation

The implementation is straightforward because all necessary data is already available at the insertion points, and we're following established patterns in the codebase.

### Key Constraint: Test Set Only
The CSV will contain **only frames from the test set**, not the training set. This is intentional and follows standard machine learning evaluation practices:
- Training data is used to fit the model
- Test data is used to evaluate model performance on unseen data
- The frame-level CSV is for detailed analysis of test set predictions


