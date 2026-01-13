# Frame-Level Validation Output - Design Documentation

## Overview
This directory contains design documentation for adding frame-level validation output to DeepLabCut's `evaluate_network()` function.

## Documents

### 1. `user_story.md`
**Purpose**: Captures the user intent and requirements for this feature.

**Contents**:
- User requirements and goals
- Expected CSV output format
- Use cases
- Scope (in/out of scope)
- Success criteria

**Read this first** to understand what we're trying to achieve.

---

### 2. `research.md`
**Purpose**: Technical research identifying all relevant code sections for implementation.

**Contents**:
- Architecture overview (TensorFlow vs PyTorch backends)
- Detailed code section analysis with line numbers
- Data structure documentation
- Key variables and their purposes
- Output file locations and naming conventions
- Specific implementation insertion points
- Related files for reference

**Read this second** to understand how to implement the feature.

---

## Quick Reference

### Key Files to Modify

1. **TensorFlow Backend**:
   - File: `deeplabcut/pose_estimation_tensorflow/core/evaluate.py`
   - Function: `evaluate_network()` (lines 533-1027)
   - Insertion point: After line 936

2. **PyTorch Backend**:
   - File: `deeplabcut/pose_estimation_pytorch/apis/evaluation.py`
   - Function: `evaluate_network()` (lines 685-850)
   - Insertion point: After line 640

### Data Available at Implementation Points

**TensorFlow**:
- `Data`: Ground truth DataFrame (MultiIndex: scorer/bodyparts/coords)
- `DataMachine`: Predictions DataFrame (MultiIndex: scorer/bodyparts/coords)
- `testIndices`: Test set frame indices
- `trainIndices`: Training set frame indices
- `evaluationfolder`: Output directory path
- `DLCscorer`: Model name string
- `cfg["scorer"]`: Human annotator name

**PyTorch**:
- `loader.ground_truth_keypoints("test")`: Dict of ground truth arrays
- `predictions["test"]`: Dict of prediction arrays
- `eval_parameters.bodyparts`: List of bodypart names
- `output_filename`: Path object for output files
- `scorer`: Model name string

### Proposed Output

**Filename**: `{DLCscorer}-frame-level-results.csv`

**Location**: Same evaluation folder as other results

**Columns**:
- `frame_index`: Integer index
- `image_path`: Path to the image
- `gt_{bodypart}_x`, `gt_{bodypart}_y`: Ground truth coordinates
- `pred_{bodypart}_x`, `pred_{bodypart}_y`: Predicted coordinates
- `conf_{bodypart}`: Confidence/likelihood scores

---

## Implementation Strategy

### Phase 1: TensorFlow Backend (Single-Animal)
1. Create helper function to convert DataFrames to frame-level CSV format
2. Add optional parameter `save_frame_level_csv=True` to `evaluate_network()`
3. Insert CSV export logic after existing evaluation
4. Test with single-animal projects

### Phase 2: PyTorch Backend (Single-Animal)
1. Create helper function to convert dicts/arrays to frame-level CSV format
2. Add same parameter to PyTorch `evaluate_network()`
3. Insert CSV export logic after existing evaluation
4. Test with single-animal projects

### Phase 3: Multi-Animal Support (Future)
1. Extend to handle multiple individuals
2. Adjust column naming for multi-animal format
3. Test with multi-animal projects

---

## Next Steps

1. ✅ **Complete**: User story and research documentation
2. **TODO**: Design the helper functions for data conversion
3. **TODO**: Implement TensorFlow backend changes
4. **TODO**: Implement PyTorch backend changes
5. **TODO**: Add tests
6. **TODO**: Update documentation

---

## Notes

- Both backends need separate implementations due to different data structures
- TensorFlow uses pandas DataFrames with MultiIndex
- PyTorch uses dictionaries of NumPy arrays
- Follow existing patterns (e.g., `per_keypoint_evaluation`) for consistency
- Maintain backward compatibility - no breaking changes

