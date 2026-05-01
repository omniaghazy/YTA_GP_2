# 📖 Detailed Setup and Usage Guide

This guide provides step-by-step instructions for setting up the **Hard Drive Failure Prediction System** and running the pipeline on your own machine.

---

## 1. Environment Initialization

### Prerequisites
- Python 3.11 or higher
- Conda (recommended) or Virtualenv

### Option A: Using Conda (Recommended)
This ensures all system-level dependencies (like LightGBM's OpenMP) are correctly mapped.
```bash
# Create the environment
conda env create -f conda_env.yml

# Activate it
conda activate drive_failure_prediction
```

### Option B: Using Pip
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Data Preparation

The system expects a cleaned CSV file named `seagate_cleaned_cols.csv` in the project root.

### For New Users:
1. **Download the Backblaze Dataset**: Obtain the quarterly CSV files from [Backblaze's website](https://www.backblaze.com/b2/hard-drive-test-data.html).
2. **Configure the Path**: Update the `raw_csv_dir` in `src/drive_failure_system.py` (Line 86) to point to your downloaded CSV folder.
3. **Build the Cleaned Dataset**:
   Run the following snippet in a Python shell or update the `__main__` block in `drive_failure_system.py`:
   ```python
   from src.drive_failure_system import DataLoader, PipelineConfig
   cfg = PipelineConfig(raw_csv_dir="path/to/raw/data")
   loader = DataLoader(cfg)
   loader.build_merged_csv()       # Merges daily CSVs
   loader.drop_high_null_columns()  # Removes low-quality features
   ```

---

## 3. Training the Pipeline

Once your `seagate_cleaned_cols.csv` is ready, simply trigger the orchestrator:

```bash
python src/drive_failure_system.py
```

### What happens during execution?
1. **Data Loading**: Reads the cleaned CSV using Polars (Memory-Safe).
2. **Preprocessing**: Labels drives with a 14-day failure horizon and downsamples healthy drives.
3. **Feature Engineering**: Generates 200+ features (Lag, Rolling, Trend, Age).
4. **Stacking**: Trains base models (XGB, LGBM) and a meta-ensemble.
5. **Evaluation**: Generates PR-AUC curves, Confusion Matrices, and SHAP explanations.
6. **Artifact Saving**: Saves the final `InferencePipeline` to the `models/` folder.

---

## 4. Troubleshooting

- **Memory Errors**: Ensure you are using the Polars-based loader. If you have less than 16GB of RAM, reduce the `healthy_drive_multiplier` in the config.
- **CatBoost Missing**: The system will gracefully skip CatBoost if it's not installed. You can install it via `pip install catboost`.
- **GPU Acceleration**: Both XGBoost and LightGBM support GPU. To enable it, update the params in `PipelineConfig` with `tree_method='gpu_hist'` (XGB) or `device='gpu'` (LGBM).

---

## 5. Deployment Readiness

The system generates a `joblib` artifact in the `models/` directory. This artifact contains the entire preprocessing + model stack.

To use it for inference:
```python
import joblib
pipeline = joblib.load("models/final_pipeline.joblib")
predictions = pipeline.predict_proba(new_data)
```

---

## 6. Progress Tracking (NEW)
The system now automatically logs evaluation metrics to **`metrics_history.csv`** every time a pipeline run completes. This allows you to track model performance improvements over time and compare different experiment results easily.
