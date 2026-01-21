# Training History Analysis Tools

This document describes how to use the training history visualization and filtering tools.

## Overview

Two new tools have been added to help you analyze and manage training history:

1. **`visualize_history.py`** - Visualize and analyze training history with highlighted key epochs
2. **`filter_history.py`** - Filter and save selected epochs from training history

## 1. Visualize Training History

### Usage

```bash
python visualize_history.py [OPTIONS]
```

### Options

- `--history PATH` - Path to training history JSON file (default: `training_history.json`)
- `--output PATH` - Output path for visualization plot (default: `training_history_analysis.png`)
- `--show` - Display the plot interactively

### Examples

```bash
# Basic usage - analyze default training_history.json
python visualize_history.py

# Analyze specific history file
python visualize_history.py --history my_training_history.json

# Save to custom output location
python visualize_history.py --output my_analysis.png

# Show interactive plot
python visualize_history.py --show
```

### Output

The script provides:

1. **Detailed Console Output**:
   - Key epochs summary (best validation, best training, overfitting start)
   - Epoch-by-epoch breakdown with highlights
   - Recommendations on which epochs to keep/discard

2. **Visualization Plot** with 4 panels:
   - **Loss Curves**: Training and validation loss with key epochs highlighted
   - **Overfitting Gap**: Bar chart showing train/val loss gap (overfitting indicator)
   - **Learning Rate Schedule**: Shows LR changes over epochs
   - **Recommendations**: Summary of which epochs to keep/discard

### Key Highlights

The tool automatically identifies and highlights:

- 🏆 **Best Validation Loss** - The epoch with lowest validation loss (PRIMARY checkpoint)
- ⭐ **Best Training Loss** - The epoch with lowest training loss
- ⚠️ **Overfitting Start** - When validation loss starts increasing while training decreases
- 📉 **Learning Rate Changes** - When learning rate is adjusted
- ✓ **Stable Epochs** - Epochs with low loss variation

## 2. Filter Training History

### Usage

```bash
python filter_history.py [OPTIONS]
```

### Options

- `--history PATH` - Path to training history JSON file (default: `training_history.json`)
- `--output PATH` - Output path for filtered history (default: `training_history_filtered.json`)
- `--keep-epochs EPOCHS` - Comma-separated list of epochs to keep (e.g., "1,5,10,21")
- `--keep-range RANGE` - Range of epochs to keep (e.g., "1-21")
- `--keep-best N` - Keep N best epochs by validation loss
- `--remove-epochs EPOCHS` - Comma-separated list of epochs to remove
- `--interactive` - Interactive mode to select epochs

### Examples

#### Keep Specific Epochs

```bash
# Keep only epochs 1, 5, 10, and 21
python filter_history.py --keep-epochs "1,5,10,21"
```

#### Keep Epoch Range

```bash
# Keep epochs 1 through 21 (before overfitting)
python filter_history.py --keep-range "1-21" --output history_clean.json
```

#### Keep Best N Epochs

```bash
# Keep the 5 best epochs by validation loss
python filter_history.py --keep-best 5 --output history_top5.json

# Keep the 10 best epochs
python filter_history.py --keep-best 10
```

#### Remove Specific Epochs

```bash
# Remove epochs 1-10 (warmup) and 30+ (overfitting)
python filter_history.py --remove-epochs "1,2,3,4,5,6,7,8,9,10,30,31,32,33,34,35,36"
```

#### Interactive Mode

```bash
python filter_history.py --interactive
```

Interactive mode provides 5 options:
1. Keep specific epochs (enter comma-separated list)
2. Keep epoch range (enter range like "1-21")
3. Keep best N epochs by validation loss
4. Remove specific epochs (enter comma-separated list)
5. Keep all epochs (no filtering)

## Recommended Workflow

### Step 1: Visualize and Analyze

```bash
python visualize_history.py
```

Review the console output and generated plot to understand:
- Which epoch has the best validation loss
- When overfitting starts
- Which epochs are stable
- Overall training dynamics

### Step 2: Decide What to Keep

Based on the analysis, decide which epochs to keep. Common strategies:

1. **Keep Only Best Model**:
   ```bash
   python filter_history.py --keep-best 1
   ```

2. **Keep Pre-Overfitting Epochs**:
   ```bash
   python filter_history.py --keep-range "1-21"  # if overfitting starts at epoch 22
   ```

3. **Keep Key Checkpoints**:
   ```bash
   python filter_history.py --keep-epochs "7,16,21,27"
   ```

4. **Remove Warmup and Overfitting**:
   ```bash
   python filter_history.py --remove-epochs "1,2,3,4,5,30,31,32,33,34,35,36"
   ```

### Step 3: Verify Filtered History

```bash
# Visualize the filtered history
python visualize_history.py --history training_history_filtered.json --output filtered_analysis.png
```

Compare the original and filtered visualizations to ensure you kept the right epochs.

## Example Analysis Session

```bash
# Step 1: Analyze original history
python visualize_history.py

# Output shows:
# - Best val loss at epoch 21
# - Overfitting starts at epoch 12
# - High val loss in early epochs (1-10)

# Step 2: Filter to keep pre-overfitting epochs with good performance
python filter_history.py --keep-range "7-21" --output history_good_epochs.json

# Step 3: Verify the filtered result
python visualize_history.py --history history_good_epochs.json --output filtered_viz.png
```

## Tips

- **Always visualize first** to understand your training dynamics
- **Keep the best validation epoch** (21 in the example) as your primary checkpoint
- **Consider keeping epochs before overfitting** starts
- **Remove early warmup epochs** (typically 1-10) as they have high loss
- **Keep learning rate change epochs** for comparison purposes
- **Test your filtered history** by visualizing it to confirm it looks correct

## Files Generated

- `training_history_analysis.png` - Visualization of original history
- `training_history_filtered.json` - Filtered history file (can be customized with `--output`)
- `training_history_top5.json` - Example: Top 5 epochs by validation loss
- `training_history_before_overfit.json` - Example: Epochs before overfitting

## Integration with Training

These filtered history files can be used for:
- Analyzing training behavior
- Selecting which checkpoints to keep
- Documentation and reporting
- Comparing different training runs
- Understanding model convergence

The filtered history maintains the same JSON format as the original, so it can be used with any downstream tools that read training history.
