# Research: Frame-Level Validation Output Implementation

## Overview
This document identifies the relevant files, functions, and code sections needed to implement frame-level validation output in DeepLabCut's `evaluate_network()` function.

## Architecture Overview

DeepLabCut has two backend implementations:
1. **TensorFlow Backend**: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`
2. **PyTorch Backend**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`

Both backends need to be modified to support the new CSV output.

---

## TensorFlow Backend

### Primary File
**File**: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`

### Key Function: `evaluate_network()`
- **Location**: Lines 533-1027
- **Purpose**: Main entry point for network evaluation
- **Key Parameters**:
  - `config`: Path to config.yaml
  - `Shuffles`: List of shuffle indices
  - `trainingsetindex`: Which training set fraction to use
  - `plotting`: Whether to plot predictions
  - `show_errors`: Whether to display errors
  - `comparisonbodyparts`: Which bodyparts to evaluate
  - `per_keypoint_evaluation`: Boolean flag for per-keypoint CSV output (lines 598-600)

### Critical Code Sections

#### 1. Data Loading (Lines 700-707)
```python
Data = pd.read_hdf(
    os.path.join(
        cfg["project_path"],
        str(trainingsetfolder),
        "CollectedData_" + cfg["scorer"] + ".h5",
    )
)
```
- **Purpose**: Loads ground truth annotations
- **Data Structure**: `Data` is a pandas DataFrame with MultiIndex columns (scorer, bodyparts, coords)
- **Contains**: Ground truth x, y coordinates for all labeled frames

#### 2. Train/Test Split Indices (Lines 753-755)
```python
_, trainIndices, testIndices, _ = auxiliaryfunctions.load_metadata(
    Path(cfg["project_path"], train_pose_cfg["metadataset"])
)
```
- **Purpose**: Loads indices that separate train and test sets
- **Variables**: 
  - `trainIndices`: List of indices for training frames
  - `testIndices`: List of indices for test frames
- **Usage**: These indices are used to filter `Data` DataFrame

#### 3. Prediction Loop (Lines 849-874)
```python
for imageindex, imagename in tqdm(enumerate(Data.index)):
    image = imread(
        os.path.join(cfg["project_path"], *imagename),
        mode="skimage",
    )
    if scale != 1:
        image = imresize(image, scale)

    image_batch = data_to_input(image)
    # Compute prediction with the CNN
    outputs_np = sess.run(
        outputs, feed_dict={inputs: image_batch}
    )
    scmap, locref = predict.extract_cnn_output(
        outputs_np, test_pose_cfg
    )

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(
        scmap, locref, test_pose_cfg["stride"]
    )
    PredicteData[imageindex, :] = (
        pose.flatten()
    )
```
- **Purpose**: Iterates through all frames and generates predictions
- **Key Variables**:
  - `imageindex`: Index of current frame
  - `imagename`: Path to the image (from `Data.index`)
  - `PredicteData`: NumPy array storing all predictions (shape: [num_images, 3*num_bodyparts])
  - `pose`: Flattened array containing [x, y, likelihood] for each bodypart

#### 4. Creating Predictions DataFrame (Lines 877-890)
```python
index = pd.MultiIndex.from_product(
    [
        [DLCscorer],
        test_pose_cfg["all_joints_names"],
        ["x", "y", "likelihood"],
    ],
    names=["scorer", "bodyparts", "coords"],
)

# Saving results
DataMachine = pd.DataFrame(
    PredicteData, columns=index, index=Data.index
)
DataMachine.to_hdf(resultsfilename, "df_with_missing")
```
- **Purpose**: Converts predictions to DataFrame and saves to HDF5
- **Data Structure**: 
  - `DataMachine`: DataFrame with same structure as `Data` (ground truth)
  - MultiIndex columns: (scorer, bodyparts, coords)
  - Index: Same as `Data.index` (image paths)
  - Contains: x, y, likelihood for each bodypart

#### 5. Combining Ground Truth and Predictions (Lines 896-898)
```python
DataCombined = pd.concat(
    [Data.T, DataMachine.T], axis=0, sort=False
).T
```
- **Purpose**: Combines ground truth and predictions into single DataFrame
- **Data Structure**: 
  - `DataCombined`: Has both scorers (human annotator and DLC model) as top-level columns
  - Can access ground truth: `DataCombined[cfg["scorer"]]`
  - Can access predictions: `DataCombined[DLCscorer]`

#### 6. Per-Keypoint Evaluation Example (Lines 929-936)
```python
if per_keypoint_evaluation:
    df_keypoint_error = keypoint_error(
        RMSE, RMSEpcutoff, trainIndices, testIndices
    )
    kpt_filename = DLCscorer + "-keypoint-results.csv"
    df_keypoint_error.to_csv(
        Path(evaluationfolder) / kpt_filename
    )
```
- **Purpose**: Shows existing pattern for saving additional CSV files
- **Location**: `evaluationfolder` is the output directory
- **Naming**: Uses `DLCscorer` prefix for consistency

### Helper Functions

#### `pairwisedistances()` (Lines 25-43)
- **Purpose**: Calculates RMSE between ground truth and predictions
- **Input**: `DataCombined` DataFrame with both scorers
- **Output**: RMSE values per bodypart per frame
- **Note**: Shows how to extract data from the combined DataFrame

#### `keypoint_error()` (Lines 484-530)
- **Purpose**: Computes per-bodypart RMSE for train/test sets
- **Shows**: How to use `trainIndices` and `testIndices` to filter data

---

## PyTorch Backend

### Primary File
**File**: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`

### Key Function: `evaluate_network()`
- **Location**: Lines 685-850
- **Purpose**: Main entry point for PyTorch-based evaluation
- **Key Parameters**: Similar to TensorFlow version, plus:
  - `per_keypoint_evaluation`: Boolean for per-bodypart results (line 696)
  - `pcutoff`: Confidence threshold (can be float or list)

### Critical Code Sections

#### 1. Main Evaluation Loop (Lines 582-614)
```python
for split in ["train", "test"]:
    results, predictions_for_split = evaluate(
        pose_runner=pose_runner,
        loader=loader,
        mode=split,
        pcutoff=pcutoff,
        detector_runner=detector_runner,
        comparison_bodyparts=comparison_bodyparts,
        per_keypoint_evaluation=per_keypoint_evaluation,
        parameters=parameters,
    )
    if per_keypoint_evaluation:
        rmse_per_bodypart[split] = _extract_rmse_per_bodypart(
            results,
            eval_parameters.bodyparts,
            eval_parameters.unique_bpts,
        )

    df_split_predictions = build_predictions_dataframe(
        scorer=scorer,
        predictions=predictions_for_split,
        parameters=eval_parameters,
        image_name_to_index=image_to_dlc_df_index,
    )
```
- **Purpose**: Evaluates both train and test splits
- **Key Variables**:
  - `split`: Either "train" or "test"
  - `predictions_for_split`: Dict mapping image paths to predictions
  - `df_split_predictions`: DataFrame of predictions for this split

#### 2. Saving Results (Lines 615-634)
```python
results_filename = f"{scorer}.h5"
df_predictions = pd.concat(predictions.values(), axis=0)
df_predictions = df_predictions.reindex(loader.df.index)
output_filename = loader.evaluation_folder / results_filename
output_filename.parent.mkdir(parents=True, exist_ok=True)
df_predictions.to_hdf(output_filename, key="df_with_missing")

df_scores = pd.DataFrame([scores]).set_index(...)
scores_filepath = output_filename.with_suffix(".csv")
scores_filepath = scores_filepath.with_stem(scores_filepath.stem + "-results")
print(f"Evaluation results file: {scores_filepath.name}")
save_evaluation_results(df_scores, scores_filepath, show_errors, pcutoff)

if per_keypoint_evaluation:
    rmse_per_bpt_path = output_filename.with_name(
        output_filename.stem + "-keypoint-results.csv"
    )
    save_rmse_per_bodypart(rmse_per_bodypart, rmse_per_bpt_path, show_errors)
```
- **Purpose**: Saves predictions and evaluation results
- **Output Files**:
  - HDF5 file: `{scorer}.h5` with predictions
  - CSV file: `{scorer}-results.csv` with summary metrics
  - CSV file (optional): `{scorer}-keypoint-results.csv` with per-bodypart RMSE
- **Location**: `loader.evaluation_folder`

#### 3. The `evaluate()` Function (Lines 108-234)
```python
def evaluate(
    pose_runner: InferenceRunner,
    loader: Loader,
    mode: str,
    ...
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]]]:
```
- **Purpose**: Core evaluation logic for a single split (train or test)
- **Returns**:
  - `results`: Dict with metrics (rmse, rmse_pcutoff, etc.)
  - `predictions`: Dict mapping image paths to prediction arrays

#### 4. Getting Ground Truth and Predictions (Lines 159-160)
```python
gt_pose = loader.ground_truth_keypoints(mode)
pred_pose = {filename: pred["bodyparts"] for filename, pred in predictions.items()}
```
- **Purpose**: Extracts ground truth and predictions
- **Data Structure**:
  - `gt_pose`: Dict mapping image paths to ground truth arrays (shape: [n_individuals, n_bodyparts, 3])
  - `pred_pose`: Dict mapping image paths to prediction arrays (same shape)
  - Last dimension (3): [x, y, visibility/confidence]

#### 5. Computing Metrics (Lines 205-214)
```python
results = metrics.compute_metrics(
    gt_pose,
    pred_pose,
    single_animal=False if force_multi_animal else parameters.max_num_animals == 1,
    pcutoff=pcutoff,
    unique_bodypart_poses=pred_unique,
    unique_bodypart_gt=gt_unique,
    per_keypoint_rmse=per_keypoint_evaluation,
    compute_detection_rmse=False,
)
```
- **Purpose**: Computes evaluation metrics
- **Module**: `deeplabcut.core.metrics.api`
- **Returns**: Dict with rmse, rmse_pcutoff, and optionally per-keypoint metrics

---

## Data Structures

### TensorFlow Backend Data Structures

#### Ground Truth DataFrame (`Data`)
- **Type**: `pandas.DataFrame`
- **Index**: Image paths (tuples or strings)
- **Columns**: MultiIndex with 3 levels:
  - Level 0 (scorer): Human annotator name (e.g., "experimenter1")
  - Level 1 (bodyparts): Bodypart names (e.g., "nose", "left_ear", "right_ear")
  - Level 2 (coords): "x" and "y"
- **Example Access**: `Data[scorer]["nose"]["x"]` → Series of x-coordinates for nose

#### Predictions DataFrame (`DataMachine`)
- **Type**: `pandas.DataFrame`
- **Index**: Same as `Data` (image paths)
- **Columns**: MultiIndex with 3 levels:
  - Level 0 (scorer): DLC model name (e.g., "DLC_resnet50_...")
  - Level 1 (bodyparts): Bodypart names
  - Level 2 (coords): "x", "y", "likelihood"
- **Example Access**: `DataMachine[DLCscorer]["nose"]["likelihood"]` → Series of confidence scores

#### Combined DataFrame (`DataCombined`)
- **Type**: `pandas.DataFrame`
- **Structure**: Concatenation of `Data` and `DataMachine`
- **Columns**: MultiIndex with both scorers at level 0
- **Usage**: Allows side-by-side comparison of ground truth and predictions

### PyTorch Backend Data Structures

#### Ground Truth Dict (`gt_pose`)
- **Type**: `dict[str, np.ndarray]`
- **Keys**: Image paths (strings)
- **Values**: NumPy arrays of shape `[n_individuals, n_bodyparts, 3]`
  - For single-animal: `[1, n_bodyparts, 3]`
  - Last dimension: `[x, y, visibility]` where visibility is 0, 1, or 2

#### Predictions Dict (`pred_pose`)
- **Type**: `dict[str, np.ndarray]`
- **Keys**: Image paths (strings)
- **Values**: NumPy arrays of shape `[n_individuals, n_bodyparts, 3]`
  - For single-animal: `[1, n_bodyparts, 3]`
  - Last dimension: `[x, y, confidence]` where confidence is 0.0 to 1.0

---

## Key Variables and Indices

### Common Variables Across Both Backends

1. **`testIndices`**: List/array of integer indices for test set frames
2. **`trainIndices`**: List/array of integer indices for training set frames
3. **`evaluationfolder`**: Path to output directory for evaluation results
4. **`DLCscorer`**: String name of the DLC model (used in filenames)
5. **`cfg["scorer"]`**: String name of the human annotator
6. **Bodypart names**: List of bodypart names (order matters!)

### TensorFlow-Specific
- **`test_pose_cfg["all_joints_names"]`**: List of bodypart names in order
- **`PredicteData`**: NumPy array of shape `[n_images, 3*n_bodyparts]`
- **`Data.index`**: Index of the DataFrame (image paths)

### PyTorch-Specific
- **`loader`**: Data loader object with methods:
  - `loader.ground_truth_keypoints(mode)`: Get ground truth for train/test
  - `loader.evaluation_folder`: Output directory path
  - `loader.df`: DataFrame with all annotations
- **`parameters`**: `PoseDatasetParameters` object with:
  - `parameters.bodyparts`: List of bodypart names
  - `parameters.individuals`: List of individual names
  - `parameters.max_num_animals`: Number of animals (1 for single-animal)

---

## Output File Locations and Naming Conventions

### TensorFlow Backend
- **Base Directory**: `{project_path}/evaluation-results/`
- **Evaluation Folder**: `{project_path}/evaluation-results/iteration-{train_frac}-shuffle-{shuffle}/`
- **Existing Files**:
  - `{DLCscorer}.h5`: Predictions in HDF5 format
  - `{DLCscorer}-results.csv`: Summary metrics
  - `{DLCscorer}-keypoint-results.csv`: Per-keypoint RMSE (if `per_keypoint_evaluation=True`)

### PyTorch Backend
- **Base Directory**: `{project_path}/evaluation-results-pytorch/`
- **Evaluation Folder**: Similar structure to TensorFlow
- **Existing Files**: Same naming convention as TensorFlow

### Proposed New File
- **Name**: `{DLCscorer}-frame-level-results.csv`
- **Location**: Same `evaluationfolder` as other results
- **Content**: Frame-by-frame ground truth, predictions, and confidence scores

---

## Implementation Points

### Where to Add Frame-Level CSV Export

#### TensorFlow Backend (`evaluate.py`)
**Insertion Point**: After line 936 (after per_keypoint_evaluation block)
- At this point, we have:
  - `Data`: Ground truth DataFrame
  - `DataMachine`: Predictions DataFrame
  - `testIndices`: Test set indices
  - `evaluationfolder`: Output directory
  - `DLCscorer`: Model name for filename

**Pseudocode**:
```python
# After line 936
# Create frame-level CSV for test set
frame_level_data = create_frame_level_csv(
    ground_truth=Data,
    predictions=DataMachine,
    test_indices=testIndices,
    scorer_gt=cfg["scorer"],
    scorer_pred=DLCscorer,
)
frame_level_filename = DLCscorer + "-frame-level-results.csv"
frame_level_data.to_csv(
    Path(evaluationfolder) / frame_level_filename,
    index=True
)
```

#### PyTorch Backend (`evaluation.py`)
**Insertion Point**: After line 640 (after per_keypoint_evaluation block)
- At this point, we have:
  - `predictions`: Dict with predictions for both train and test
  - `loader`: Can access ground truth via `loader.ground_truth_keypoints("test")`
  - `output_filename`: Path object for output files
  - `scorer`: Model name

**Pseudocode**:
```python
# After line 640
# Create frame-level CSV for test set
gt_test = loader.ground_truth_keypoints("test")
pred_test = {img: pred["bodyparts"] for img, pred in predictions["test"].items()}
frame_level_data = create_frame_level_csv_pytorch(
    ground_truth=gt_test,
    predictions=pred_test,
    parameters=eval_parameters,
)
frame_level_path = output_filename.with_name(
    output_filename.stem + "-frame-level-results.csv"
)
frame_level_data.to_csv(frame_level_path)
```

---

## Related Files for Reference

### Utility Functions
- **`deeplabcut/utils/auxiliaryfunctions.py`**: Helper functions for file paths, metadata loading
- **`deeplabcut/core/metrics/api.py`**: Metrics computation (lines 90-104 show example usage)
- **`deeplabcut/core/metrics/distance_metrics.py`**: RMSE computation functions

### Data Conversion
- **`deeplabcut/pose_estimation_pytorch/apis/videos.py`**:
  - `create_df_from_prediction()` function (lines 828-878)
  - Shows how to convert predictions dict to DataFrame
  - Demonstrates CSV export with `save_as_csv` parameter

### Testing
- **`tests/pose_estimation_pytorch/apis/test_apis_evaluate.py`**:
  - Test cases for evaluation (lines 22-47)
  - Shows expected data structures and formats

---

## Summary of Key Findings

1. **Both backends have similar evaluation flow**:
   - Load ground truth
   - Generate predictions
   - Combine and compute metrics
   - Save results to files

2. **Data is already available** at the right points:
   - Ground truth and predictions are in memory
   - Test/train indices are known
   - Output directory is established

3. **Existing patterns to follow**:
   - `per_keypoint_evaluation` flag shows how to add optional CSV output
   - File naming follows `{DLCscorer}-{description}.csv` pattern
   - Files are saved in `evaluationfolder`

4. **Data structures differ between backends**:
   - TensorFlow: Uses pandas DataFrames with MultiIndex
   - PyTorch: Uses dicts of NumPy arrays
   - Need separate helper functions for each backend

5. **Implementation approach**:
   - Create helper function to convert data to frame-level CSV format
   - Add optional parameter (e.g., `save_frame_level_csv=True`)
   - Insert CSV export after existing evaluation logic
   - Follow existing naming and location conventions

