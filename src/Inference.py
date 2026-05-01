import joblib
import polars as pl
import sys
import os

# Module Fix: Redirect __main__ references back to the source module
try:
    import drive_failure_system as dfs
except ImportError:
    import src.drive_failure_system as dfs

sys.modules['__main__'].InferencePipeline = dfs.InferencePipeline
sys.modules['__main__'].PipelineConfig = dfs.PipelineConfig
sys.modules['__main__'].FeatureEngineer = dfs.FeatureEngineer
sys.modules['__main__'].FeatureSelector = dfs.FeatureSelector
sys.modules['__main__'].DNNTrainer = dfs.DNNTrainer
sys.modules['__main__'].Preprocessor = dfs.Preprocessor
sys.modules['__main__'].EarlyWarningSystem = dfs.EarlyWarningSystem

# Load the production-ready pipeline
pipeline = joblib.load("models/inference_pipeline.joblib")

# Load new daily SMART data (Using cleaned data for test)
new_data = pl.read_csv("seagate_cleaned_cols.csv").head(100).with_columns(pl.col("date").str.to_date())

# Get failure probabilities and alerts
results = pipeline.score(new_data)
print(results[["serial_number", "failure_prob", "alert"]])
