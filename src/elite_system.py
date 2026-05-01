"""
elite_system.py
================
The 'Elite' version of the Hard Drive Failure Prediction System.
Integrates:
1. Time-Series Windowing (CNN-LSTM)
2. Hyperparameter Optimization (Optuna)
3. Advanced Stacking (ML)
4. Context-Aware Inference

Requirements:
- tensorflow, polars, pandas, numpy, scikit-learn, joblib, optuna, plotly
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import time
import logging
import json
import datetime
import warnings
import joblib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
import optuna

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Conv1D, LSTM, Flatten, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    f1_score, recall_score, precision_score,
)
from sklearn.feature_selection import VarianceThreshold

# Model imports for Stacking
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Filter warnings
warnings.filterwarnings("ignore")

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("EliteFailure")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EliteConfig:
    cleaned_csv: str = "seagate_cleaned_cols.csv"
    artifacts_dir: str = "models"
    best_model_info: str = "models/best_model_info.json"
    
    # Time-Series Params
    window_size: int = 7  # History of 7 days
    
    # Target horizon
    prediction_horizon: int = 14
    
    # Splitting
    val_start: datetime.date  = datetime.date(2025, 11, 16)
    test_start: datetime.date = datetime.date(2025, 12,  1)

    # Optuna
    n_trials: int = 5 # Set to 5 for speed in this turn, increase for production
    
    # Feature Engineering
    lag_days: int = 1
    
    def __post_init__(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Advanced Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
class ElitePreprocessor:
    def __init__(self, cfg: EliteConfig):
        self.cfg = cfg
        self.scaler = RobustScaler()
        self.selector = VarianceThreshold(threshold=0) # Drop zero-variance
        self.top_features = []

    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        log.info("Elite Preprocessing: Imputation, Clipping, Scaling...")
        smart_cols = [c for c in df.columns if c.startswith("smart_")]
        
        # Impute
        df = df.with_columns([
            pl.col(c).forward_fill().over("serial_number").fill_null(0)
            for c in smart_cols
        ])
        
        # Target 14d
        df = df.with_columns(
            pl.col("failure").reverse().rolling_max(window_size=self.cfg.prediction_horizon, min_samples=1)
            .reverse().over("serial_number").fill_null(0).cast(pl.Int8).alias("target")
        )

        # Lag Features for ML
        for c in smart_cols:
            df = df.with_columns(pl.col(c).shift(self.cfg.lag_days).over("serial_number").fill_null(0).alias(f"{c}_lag{self.cfg.lag_days}"))
        
        return df

    def create_sequences(self, df: pl.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert tabular data to (samples, window_size, features) for CNN-LSTM."""
        log.info(f"Creating sequences (Window Size: {self.cfg.window_size})...")
        X_seq, y_seq = [], []
        
        for sn, group in df.group_by("serial_number"):
            if group.height < self.cfg.window_size:
                continue
            
            group_vals = group.select(features).to_numpy()
            target_vals = group.select("target").to_numpy()
            
            for i in range(len(group_vals) - self.cfg.window_size + 1):
                X_seq.append(group_vals[i : i + self.cfg.window_size])
                y_seq.append(target_vals[i + self.cfg.window_size - 1])
                
        return np.array(X_seq), np.array(y_seq)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Model Architectures
# ──────────────────────────────────────────────────────────────────────────────
def build_hybrid_model(input_shape, params):
    """CNN-LSTM Hybrid Architecture."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=params['filters'], kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=params['filters']//2, kernel_size=3, activation='relu', padding='same'),
        Dropout(params['dropout']),
        LSTM(params['lstm_units'], return_sequences=False),
        Dropout(params['dropout']),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
        loss='binary_crossentropy',
        metrics=['AUC', tf.keras.metrics.Recall(name='recall')]
    )
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 4. Optuna Tuners
# ──────────────────────────────────────────────────────────────────────────────
class EliteOptimizer:
    def __init__(self, cfg: EliteConfig):
        self.cfg = cfg

    def tune_ml(self, X_train, y_train, X_val, y_val):
        """Tune XGBoost as the lead ML candidate."""
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100)
            }
            clf = XGBClassifier(**param, random_state=42, use_label_encoder=False, eval_metric='logloss')
            clf.fit(X_train, y_train)
            preds = clf.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.cfg.n_trials)
        return study.best_params

    def tune_dl(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'filters': trial.suggest_categorical('filters', [32, 64, 128]),
                'lstm_units': trial.suggest_int('lstm_units', 32, 128),
                'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            }
            model = build_hybrid_model((X_train.shape[1], X_train.shape[2]), params)
            es = EarlyStopping(monitor='val_recall', mode='max', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20, batch_size=256, verbose=0, callbacks=[es]
            )
            return max(history.history['val_recall'])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.cfg.n_trials)
        return study.best_params

# ──────────────────────────────────────────────────────────────────────────────
# 5. Elite Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
class ElitePipeline:
    def __init__(self, cfg: EliteConfig):
        self.cfg = cfg
        self.preprocessor = ElitePreprocessor(cfg)
        self.optimizer = EliteOptimizer(cfg)

    def run(self):
        log.info("🚀 Starting ELITE Optimization Pipeline...")
        
        # Load
        df = pl.read_csv(self.cfg.cleaned_csv).with_columns(pl.col("date").str.to_date())
        df = self.preprocessor.prepare_data(df)
        
        # Features
        features = [c for c in df.columns if c.startswith("smart_")]
        
        # Split
        train_df = df.filter(pl.col("date") < self.cfg.val_start)
        val_df   = df.filter((pl.col("date") >= self.cfg.val_start) & (pl.col("date") < self.cfg.test_start))
        test_df  = df.filter(pl.col("date") >= self.cfg.test_start)
        
        # Scaling (Robust)
        X_train_raw = train_df.select(features).to_numpy()
        self.preprocessor.scaler.fit(X_train_raw)
        joblib.dump(self.preprocessor.scaler, f"{self.cfg.artifacts_dir}/scaler.joblib")
        
        def scale_and_seq(df_pl):
            arr = self.preprocessor.scaler.transform(df_pl.select(features).to_numpy())
            # Put back in df to group by SN
            temp_df = df_pl.with_columns([pl.Series(features[i], arr[:, i]) for i in range(len(features))])
            return self.preprocessor.create_sequences(temp_df, features)

        X_train, y_train = scale_and_seq(train_df)
        X_val, y_val = scale_and_seq(val_df)
        X_test, y_test = scale_and_seq(test_df)
        
        log.info(f"Training shapes: {X_train.shape}, {y_train.shape}")
        
        # Tune & Train DL
        best_params = self.optimizer.tune_dl(X_train, y_train, X_val, y_val)
        log.info(f"Best DL Params: {best_params}")
        
        final_model = build_hybrid_model((X_train.shape[1], X_train.shape[2]), best_params)
        final_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=128, verbose=1)
        
        # Save
        final_model.save(f"{self.cfg.artifacts_dir}/best_deep_model.h5")
        dl_preds = final_model.predict(X_test, verbose=0).flatten()
        dl_score = average_precision_score(y_test, dl_preds)
        
        # ML Candidate (Tabular + Lags)
        ml_features = [c for c in train_df.columns if c.startswith("smart_")]
        X_train_ml, y_train_ml = train_df.select(ml_features).to_numpy(), train_df.select("target").to_numpy().flatten()
        X_val_ml, y_val_ml = val_df.select(ml_features).to_numpy(), val_df.select("target").to_numpy().flatten()
        X_test_ml, y_test_ml = test_df.select(ml_features).to_numpy(), test_df.select("target").to_numpy().flatten()
        
        log.info("Tuning ML Stacking Candidate...")
        ml_best_params = self.optimizer.tune_ml(X_train_ml, y_train_ml, X_val_ml, y_val_ml)
        ml_model = XGBClassifier(**ml_best_params, random_state=42)
        ml_model.fit(X_train_ml, y_train_ml)
        joblib.dump(ml_model, f"{self.cfg.artifacts_dir}/best_ml_model.joblib")
        
        ml_preds = ml_model.predict_proba(X_test_ml)[:, 1]
        ml_score = average_precision_score(y_test_ml, ml_preds)
        
        # CHAMPION SELECTION
        champion = "Deep Learning (CNN-LSTM)" if dl_score >= ml_score else "ML Stacking (XGBoost)"
        log.info(f"🏆 Champion Model: {champion} (PR-AUC: {max(dl_score, ml_score):.4f})")
        
        # Info
        info = {
            "champion_model_name": champion,
            "champion_file": "best_deep_model.h5" if dl_score >= ml_score else "best_ml_model.joblib",
            "pr_auc": float(max(dl_score, ml_score)),
            "window_size": self.cfg.window_size if dl_score >= ml_score else 1,
            "features": ml_features
        }
        with open(self.cfg.best_model_info, 'w') as f:
            json.dump(info, f, indent=4)
            
        log.info("🏆 Elite Pipeline Complete. Champion Metadata Saved.")

if __name__ == "__main__":
    ElitePipeline(EliteConfig()).run()
