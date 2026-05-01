"""
drive_failure_system.py
=======================
Sophisticated Deep Learning Hard Drive Failure Prediction System.
Replaced legacy GBDT/Stacking logic with a powerful TensorFlow DNN architecture.

Requirements:
- TensorFlow/Keras for Deep Learning
- Polars for high-performance data processing
- Sklearn for metrics and scaling
"""

import os
# Silence TensorFlow C++ logs and oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import time
import logging
import datetime
import warnings

# Filter out verbose warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Determining the column names of a LazyFrame")
warnings.filterwarnings("ignore", message=".*is_in.*is ambiguous")

import joblib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    f1_score, recall_score, precision_score,
)
from sklearn.feature_selection import mutual_info_classif

# ── Logger setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DriveFailure")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    raw_csv_dir: str      = "data"
    cleaned_csv: str      = "seagate_cleaned_cols.csv"
    artifacts_dir: str    = "models"
    history_file: str     = "metrics_history.csv"

    # ── Data filters ───────────────────────────────────────────────────────
    model_prefix: str     = "ST"
    null_threshold: float = 0.80

    # ── Target ─────────────────────────────────────────────────────────────
    prediction_horizon: int = 14

    # ── Drive downsampling ─────────────────────────────────────────────────
    healthy_drive_multiplier: int = 3
    seed: int = 42

    # ── Feature engineering ────────────────────────────────────────────────
    rolling_windows: List[int]  = field(default_factory=lambda: [7, 14, 30])
    lag_days: List[int]         = field(default_factory=lambda: [1, 3, 7])
    smart_features: List[str]   = field(default_factory=lambda: [
        "smart_1_raw", "smart_5_raw", "smart_7_raw",
        "smart_187_raw", "smart_197_raw", "smart_9_normalized",
        "smart_190_normalized", "smart_193_normalized",
    ])

    # ── Feature selection ──────────────────────────────────────────────────
    mi_sample_n: int    = 200_000
    top_n_features: int = 35

    # ── Time split ─────────────────────────────────────────────────────────
    val_start: datetime.date  = datetime.date(2025, 11, 16)
    test_start: datetime.date = datetime.date(2025, 12,  1)
    end_date: datetime.date   = datetime.date(2025, 12, 17)

    # ── Deep Learning (Sophisticated DNN) ──────────────────────────────────
    dl_params: Dict = field(default_factory=lambda: dict(
        hidden_dims=[256, 128, 64],
        dropout=0.4,
        learning_rate=1e-3,
        batch_size=512,
        epochs=150,
        early_stopping=20,
    ))

    # ── Threshold ──────────────────────────────────────────────────────────
    recall_target: float = 0.85

    # ── Early warning ──────────────────────────────────────────────────────
    alert_threshold: float = 0.60
    alert_consecutive_days: int = 2

    def __post_init__(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DataLoader
# ──────────────────────────────────────────────────────────────────────────────
class DataLoader:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def load(self) -> pl.DataFrame:
        if not os.path.exists(self.cfg.cleaned_csv):
            raise FileNotFoundError(f"Data file '{self.cfg.cleaned_csv}' not found. Run dummy data generator first.")
        
        lf         = pl.scan_csv(self.cfg.cleaned_csv)
        # Use collect_schema() to avoid PerformanceWarning
        smart_cols = [c for c in lf.collect_schema().names() if c.startswith("smart_")]

        df = lf.with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("failure").cast(pl.Int8),
            *[pl.col(c).cast(pl.Float32, strict=False) for c in smart_cols],
        ).collect()

        df = df.sort(["serial_number", "date"])
        log.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessor
# ──────────────────────────────────────────────────────────────────────────────
class Preprocessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def remove_post_failure_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("date").filter(pl.col("failure") == 1).min().over("serial_number").alias("_first_fail")
        )
        df = df.filter(pl.col("_first_fail").is_null() | (pl.col("date") <= pl.col("_first_fail"))).drop("_first_fail")
        return df

    def create_target(self, df: pl.DataFrame) -> pl.DataFrame:
        h = self.cfg.prediction_horizon
        df = df.with_columns(
            pl.col("failure").reverse().rolling_max(window_size=h, min_samples=1).reverse().over("serial_number").fill_null(0).cast(pl.Int8).alias("target_14d")
        )
        return df

    def downsample_drives(self, df: pl.DataFrame) -> pl.DataFrame:
        failed_serials  = df.filter(pl.col("failure") == 1).select("serial_number").unique()
        n_failed        = failed_serials.height
        healthy_serials = df.join(failed_serials, on="serial_number", how="anti").select("serial_number").unique()

        n_healthy_sample = min(n_failed * self.cfg.healthy_drive_multiplier, healthy_serials.height)
        sampled_healthy  = healthy_serials.sample(n=n_healthy_sample, seed=self.cfg.seed)

        selected         = pl.concat([failed_serials, sampled_healthy])
        selected_list    = selected["serial_number"].to_list()
        df = df.filter(pl.col("serial_number").is_in(selected_list))
        log.info(f"Downsampled: {n_failed} failed + {n_healthy_sample} healthy drives")
        return self.impute_and_clip(df)

    def impute_and_clip(self, df: pl.DataFrame) -> pl.DataFrame:
        log.info("Imputing missing values (ffill + zero) and clipping outliers...")
        smart_cols = [c for c in df.columns if c.startswith("smart_")]
        
        # Impute
        df = df.with_columns([
            pl.col(c).forward_fill().over("serial_number").fill_null(0)
            for c in smart_cols
        ])
        
        # Clip
        df = df.with_columns([
            pl.col(c).clip(-1e7, 1e7)
            for c in smart_cols
        ])
        return df

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self.remove_post_failure_rows(df)
        df = self.create_target(df)
        df = self.downsample_drives(df)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FeatureEngineer
# ──────────────────────────────────────────────────────────────────────────────
class FeatureEngineer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        smart_cols = [c for c in self.cfg.smart_features if c in df.columns]
        
        # Temporal
        df = df.with_columns(
            (pl.int_range(pl.len()).over("serial_number") + 1).alias("drive_age_days"),
            pl.col("date").dt.weekday().cast(pl.Int8).alias("day_of_week"),
        )
        
        # Rolling & Trend
        exprs = []
        for col in smart_cols:
            for w in self.cfg.rolling_windows:
                exprs += [
                    pl.col(col).rolling_mean(w, min_samples=1).over("serial_number").alias(f"{col}_roll{w}_mean"),
                    pl.col(col).rolling_std(w,  min_samples=2).over("serial_number").alias(f"{col}_roll{w}_std"),
                ]
            # Delta
            exprs.append((pl.col(col) - pl.col(col).shift(7).over("serial_number")).alias(f"{col}_delta7"))

        log.info(f"Generated features for {len(smart_cols)} SMART columns.")
        return df.with_columns(*exprs).fill_null(0)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FeatureSelector
# ──────────────────────────────────────────────────────────────────────────────
class FeatureSelector:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.top_features: List[str] = []

    def fit(self, train_pl: pl.DataFrame, feature_cols: List[str]) -> "FeatureSelector":
        log.info(f"Selecting top {self.cfg.top_n_features} features via MI...")
        sample = train_pl.sample(n=min(len(train_pl), self.cfg.mi_sample_n), seed=self.cfg.seed).to_pandas()
        X_s = sample[feature_cols].fillna(0)
        y_s = sample["target_14d"]

        mi = mutual_info_classif(X_s, y_s, random_state=self.cfg.seed)
        self.top_features = pd.Series(mi, index=X_s.columns).sort_values(ascending=False).head(self.cfg.top_n_features).index.tolist()
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Modeling: DNNTrainer
# ──────────────────────────────────────────────────────────────────────────────
class DNNTrainer:
    def __init__(self, cfg: PipelineConfig, input_dim: int):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim: int):
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(self.cfg.dl_params['hidden_dims'][0], activation='relu'),
            BatchNormalization(),
            Dropout(self.cfg.dl_params['dropout']),
            Dense(self.cfg.dl_params['hidden_dims'][1], activation='relu'),
            BatchNormalization(),
            Dropout(self.cfg.dl_params['dropout'] * 0.75),
            Dense(self.cfg.dl_params['hidden_dims'][2], activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.dl_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=[Precision(name='precision'), Recall(name='recall'), AUC(curve='PR', name='pr_auc')]
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        log.info("Training Sophisticated DNN with class weights...")
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        
        # Save Scaler
        joblib.dump(self.scaler, os.path.join(self.cfg.artifacts_dir, "scaler.joblib"))

        # Weights
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = {int(k): float(v) for k, v in zip(np.unique(y_train), cw)}

        es = EarlyStopping(monitor='val_pr_auc', mode='max', patience=self.cfg.dl_params['early_stopping'], restore_best_weights=True)
        
        self.model.fit(
            X_train_s, y_train.values,
            validation_data=(X_val_s, y_val.values),
            epochs=self.cfg.dl_params['epochs'],
            batch_size=self.cfg.dl_params['batch_size'],
            class_weight=cw_dict,
            callbacks=[es],
            verbose=1
        )
        self.model.save(os.path.join(self.cfg.artifacts_dir, "failure_prediction_dnn.keras"))

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s, verbose=0).flatten()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Evaluator & Others
# ──────────────────────────────────────────────────────────────────────────────
class InferencePipeline:
    def __init__(self, cfg, feat_eng, selector, trainer, preprocessor, ews):
        self.cfg = cfg
        self.feat_eng = feat_eng
        self.selector = selector
        self.trainer = trainer
        self.preprocessor = preprocessor
        self.ews = ews

    def score(self, raw_df: pl.DataFrame) -> pd.DataFrame:
        """End-to-end scoring for new daily data."""
        # 1. Feature Engineering
        df_feat = self.feat_eng.run(raw_df)

        # 2. Feature Selection (Top-N)
        X = df_feat.select(self.selector.top_features).to_pandas().astype("float32")

        # 3. Predict Proba
        proba = self.trainer.predict_proba(X)

        # 4. Metadata & Alerts
        meta = df_feat.select(["serial_number", "date"]).to_pandas()
        result = meta.copy()
        result["failure_prob"] = proba
        
        # 5. Early Warning Alerts
        result = self.ews.generate_alerts(result)
        return result

    def save(self):
        joblib.dump(self, os.path.join(self.cfg.artifacts_dir, "inference_pipeline.joblib"))

class Evaluator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def find_best_threshold(self, y_true, y_proba):
        prec, rec, threshs = precision_recall_curve(y_true, y_proba)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
        idx = np.argmax(f1)
        return float(threshs[idx]), float(f1[idx])

    def report(self, y_true, y_proba, label="Model"):
        thresh, _ = self.find_best_threshold(y_true, y_proba)
        y_pred = (y_proba >= thresh).astype(int)
        roc = roc_auc_score(y_true, y_proba)
        pr  = average_precision_score(y_true, y_proba)
        rec = recall_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        
        log.info(f"\n{label} Results:")
        log.info(f" ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
        log.info(f" Recall : {rec:.4f} | Precision: {pre:.4f} (at thresh {thresh:.3f})")
        
        # Log to CSV
        hist_path = self.cfg.history_file
        row = pd.DataFrame([{
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": label, "roc_auc": roc, "pr_auc": pr, "recall": rec, "precision": pre
        }])
        row.to_csv(hist_path, mode='a', index=False, header=not os.path.exists(hist_path))

class EarlyWarningSystem:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
    def generate_alerts(self, df):
        df["alert"] = df["failure_prob"] >= self.cfg.alert_threshold
        return df


class DriveFailurePipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or PipelineConfig()
        self.loader = DataLoader(self.cfg)
        self.preprocessor = Preprocessor(self.cfg)
        self.feat_eng = FeatureEngineer(self.cfg)
        self.selector = FeatureSelector(self.cfg)
        self.evaluator = Evaluator(self.cfg)
        self.ews = EarlyWarningSystem(self.cfg)

    def run(self):
        log.info("Starting Sophisticated DNN Pipeline...")
        df = self.loader.load()
        df = self.preprocessor.run(df)
        df = self.feat_eng.run(df)
        
        # Split
        train_df = df.filter(pl.col("date") < self.cfg.val_start)
        val_df   = df.filter((pl.col("date") >= self.cfg.val_start) & (pl.col("date") < self.cfg.test_start))
        test_df  = df.filter(pl.col("date") >= self.cfg.test_start)
        
        feat_cols = [c for c in train_df.columns if c not in ["date", "serial_number", "model", "failure", "target_14d"]]
        self.selector.fit(train_df, feat_cols)
        top_feats = self.selector.top_features
        
        def to_xy(df_pl):
            X = df_pl.select(top_feats).to_pandas().astype("float32")
            y = df_pl.select("target_14d").to_pandas().iloc[:, 0]
            return X, y

        X_train, y_train = to_xy(train_df)
        X_val,   y_val   = to_xy(val_df)
        X_test,  y_test  = to_xy(test_df)

        self.trainer = DNNTrainer(self.cfg, input_dim=len(top_feats))
        self.trainer.fit(X_train, y_train, X_val, y_val)
        
        y_proba = self.trainer.predict_proba(X_test)
        self.evaluator.report(y_test, y_proba, label="Sophisticated DNN")
        
        # Save Inference
        self.inference = InferencePipeline(self.cfg, self.feat_eng, self.selector, self.trainer, self.preprocessor, self.ews)
        self.inference.save()
        log.info("✅ Pipeline Complete.")

if __name__ == "__main__":
    DriveFailurePipeline().run()
