# DEMATEL + Random Forest Analysis

A pipeline to perform DEMATEL multi‐expert influence analysis, identify cause vs. effect factors, and train a Random Forest classifier to distinguish them.

## Features

* **DEMATEL Steps**

  1. Aggregate expert matrices (Z)
  2. Normalize to X
  3. Compute total influence T
  4. Identify Cause / Effect factors (R±C)
  5. Plot Influential Relation Map (IRM)
  6. Export all matrices & metrics to Excel

* **Classification**

  * Build & evaluate a Random Forest on the aggregated factor profiles
  * Handle class imbalance with SMOTE
  * Show cross‐validation scores
  * Report feature importances
  * Generate a SHAP summary plot

## Requirements

* Python 3.7+
* numpy, pandas, scikit‑learn, imbalanced‑learn, shap, openpyxl, matplotlib

Install with:

```bash
pip install numpy pandas scikit-learn imbalanced-learn shap openpyxl matplotlib
```

## Project Structure

```
.
├── inputs/
│   ├── factors.txt          # one factor name per line
│   └── matrices/
│       ├── expert1.csv
│       └── expert2.csv      # each a direct‑influence matrix
├── outputs/
│   └── DEMATELAnalysis.xlsx # exported results
└── dematel_random_forest.py # main pipeline
```

## Usage

```bash
python dematel_random_forest.py
```

* Reads factor names and all CSV matrices under `inputs/`.
* Runs DEMATEL steps, plots IRM, saves Excel to `outputs/`.
* Prepares a dataset of factor influence profiles + “Cause/Effect” labels.
* Trains a Random Forest classifier (SMOTE + 5‑fold CV), prints accuracy & importances, and shows a SHAP summary plot.

## Inputs

1. **`inputs/factors.txt`** – each line: a factor name
2. **`inputs/matrices/*.csv`** – each file: one expert’s square matrix of direct influences

## Outputs

* **Excel** (`outputs/DEMATELAnalysis.xlsx`):

  * Sheet “R-C Metrics” containing R, C, R+C, R–C per factor
  * Sheets “Z”, “X”, “T” for the aggregated, normalized, and total‐influence matrices
* **Console**:

  * List of identified Cause vs Effect factors
  * Random Forest cross‐validation accuracy and feature importances
* **Plot**: IRM scatter plot and SHAP feature‐importance bar chart

