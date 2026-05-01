# Final Project Audit Report: AI Predictive Maintenance Hub

## 1. Executive Summary
The YTA Elite Predictive Maintenance system has been successfully developed and optimized. After a rigorous comparison between optimized Machine Learning (Stacking/XGBoost) and Deep Learning (CNN-LSTM), the **CNN-LSTM Hybrid** has been selected as the **Champion Model**. The system is now production-ready with a context-aware Streamlit dashboard.

## 2. Preprocessing & Feature Engineering Audit
### Data Cleaning
- **Imputation**: Forward-fill per drive serial number, followed by constant zero filling.
- **Zero-Variance Filter**: Automatically dropped static columns that provide no predictive signal.
- **Outlier Handling**: Implemented `RobustScaler` to preserve signal in extreme SMART attribute spikes.

### Feature Engineering
- **Time-Lag Features**: Added 1-day lags for ML models to provide a 'memory' effect.
- **Windowing**: Implemented a 7-day sliding window for the Deep Learning sequences.

## 3. Model Optimization Ledger
The following models were tuned using **Optuna** over multiple trials:

| Model Type | Primary Algorithm | Optimization Metric | Status |
|------------|-------------------|---------------------|--------|
| **Deep Learning** | CNN-LSTM Hybrid | Recall / PR-AUC | **Champion** |
| **Machine Learning** | Optimized XGBoost | PR-AUC | Candidate |

### Champion Hyperparameters (CNN-LSTM)
- **CNN Filters**: 64
- **LSTM Units**: 127
- **Dropout Rate**: 0.40
- **Learning Rate**: 0.0025

## 4. Production Readiness & UI Specifications
### Dynamic UI Mapping
The Streamlit dashboard (`ui.py`) now automatically detects the model type and adjusts:
- **DL Mode**: Performs windowed inference and displays time-series trend charts.
- **ML Mode**: Performs direct tabular inference and displays SHAP-based feature importance.

### 'No-Crash' Logic
Implemented a smart imputation layer in the UI:
- **Missing Columns**: Automatically filled with default values (100.0).
- **Confidence Scoring**: Alerts the user if missing data reduces prediction reliability.

## 5. Artifacts Exported
- **Model**: `models/best_deep_model.h5`
- **Scaler**: `models/scaler.joblib`
- **Metadata**: `models/best_model_info.json`
- **Explainability**: `DEEP_LEARNING_EXPLAINED.md`

**Audit Conclusion**: The system is stable, highly optimized, and ready for deployment in an enterprise environment.
