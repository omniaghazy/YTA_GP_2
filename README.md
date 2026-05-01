# 🔩 Hard Drive Failure Prediction System
### *Production-Grade Predictive Maintenance using the Backblaze Dataset*

![Banner](hard_drive_banner_1777548775629.png)

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-Polars%20%7C%20Stacking-orange.svg)]()
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()

---

## 📖 Overview
This project implements an **Elite Stacking Ensemble** to predict hard drive failures up to **14 days in advance**. Leveraging the massive Backblaze SMART dataset, the system identifies subtle degradation patterns that precede catastrophic failures, allowing for proactive data migration and reduced downtime.

### 🚀 Key Features
- **High-Performance Processing**: Built with **Polars** for memory-safe handling of 10M+ rows.
- **Advanced Feature Engineering**: Time-aware rolling metrics, lag features, and SMART trend analysis.
- **Robust Model Architecture**:
  - **Core**: Python 3.11+, Polars, Pandas, Scikit-Learn
  - **Models**: LightGBM, XGBoost, CatBoost, **PyTorch (Deep Learning)**
  - **Experiment Tracking**: Custom Metrics History, **Optuna (HPO)**
- **Deployment Ready**: Modular codebase designed for seamless integration into enterprise monitoring pipelines.

---

## 🧠 Hybrid Stacking Architecture
The system uses a 2-layer stacking ensemble:

1.  **Level 1 (Base Models)**: 
    *   **LightGBM UnderBagging**: 5 bags to handle class imbalance.
    *   **XGBoost**: Captures complex non-linearities.
    *   **CatBoost**: Handles categorical structures and prevents overfitting.
    *   **Tabular Deep Learning (MLP)**: A PyTorch-based neural network for high-dimensional feature interaction.
2.  **Level 2 (Meta-Model)**: A LightGBM classifier trained on leak-free Out-of-Fold (OOF) predictions.

**Training Protocol**: Uses Time-Series Out-of-Fold (OOF) cross-validation to ensure the meta-model never sees "future" data during training.

---

## 📂 Project Structure
```text
.
├── config/             # Configuration files and hyperparameters
├── data/               # Dataset placeholders and generation scripts
├── models/             # Trained model artifacts and scalers
├── notebooks/          # Exploratory Data Analysis (EDA) and research
│   └── Ml.ipynb        # Data preparation and analysis
├── src/                # Core production codebase
│   └── drive_failure_system.py  # Main pipeline and logic
├── conda_env.yml       # Conda environment definition
├── requirements.txt    # Pip dependencies
└── README.md           # Project documentation
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/drive-failure-prediction.git
cd drive-failure-prediction
```

### 2. Set Up the Environment
We recommend using **Conda** for a clean setup:
```bash
conda env create -f conda_env.yml
conda activate drive_failure_prediction
```
*Or via Pip:*
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
To trigger the full training and evaluation pipeline:
```bash
python src/drive_failure_system.py
```

---

## 📊 Maintenance Dashboard (Concept)
![Predictive Maintenance Animation Placeholder](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHp5emZ4b2d5emZ4b2d5emZ4b2d5emZ4b2d5emZ4b2d5emZ4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o7TKVUn7iM8FMEU24/giphy.gif)
*(Above: A visual representation of a real-time maintenance monitoring system)*

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---
**Developed with ❤️ by the YTA ML Team**
