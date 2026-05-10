# Threshold Predictor Usage

## Overview

The `predict_threshold.py` script trains a random forest regressor that predicts an optimal ΔRT (absolute difference between predicted and observed retention time) threshold. The script reads peptide‑level data from a fixed file path (`abs_diff.tsv`), carries out bootstrap‑based training, evaluates on a test set, and saves the trained model.

## Input Data Format

The script expects a **tab‑separated file** named `abs_diff.tsv` located in the same working directory. This file must contain at least two columns:

| Column | Description |
|--------|-------------|
| `abs_diff` | Numeric column. Absolute difference between predicted and measured retention time for each peptide. |
| `labels` | Integer column (0 or 1). Ground‑truth validation label (e.g., 1 = validated, 0 = not validated). |

- The file path is named as `"abs_diff.tsv"` in the script. Place your file in the script’s working directory or modify the path in the source code.
- Missing values in the feature matrix are automatically filled with column medians during regressor training, but the `abs_diff` and `labels` columns themselves should be complete.

## Usage

1. **Prepare your data file** – Create `abs_diff.tsv` with the columns described above.
2. **Install dependencies** – The script requires `numpy`, `pandas`, `scikit‑learn`, and so on.
3. **Run the script** – Execute `python predict_threshold.py` from the directory containing `abs_diff.tsv`.

## What the Script Does

- **Data splitting** – Randomly shuffles the input and splits it into 80% training and 20% test sets (with seed 42 for reproducibility).
- **Bootstrap resampling** – From the training partition, generates some bootstrap samples. For each sample, computes dataset‑level features (mean, median, standard deviation, 25th and 75th percentiles, min, max of `abs_diff`) and finds the optimal ΔRT threshold by maximising the F1‑score over the candidate thresholds.
- **Regressor training** – Fits a random forest to predict the optimal threshold from the dataset‑level features.
- **Model persistence** – Saves the trained regressor in the same directory.
- **Evaluation** – Computes accuracy and F1 for both the predicted threshold and the optimal threshold on the test set, then runs a bootstrap comparison to estimate 95% confidence intervals and approximate p‑values for the performance differences.
