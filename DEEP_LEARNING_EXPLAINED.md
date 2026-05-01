# Deep Learning Architecture: CNN-LSTM Hybrid

## Why this Architecture?
Hard drive SMART data is intrinsically **Time-Series** data. A single snapshot of a drive's health is often less informative than the **trend** leading up to that point. 

We chose a **Hybrid CNN-LSTM** architecture for several reasons:
1.  **CNN (1D Convolutional Layers)**: These act as automated feature extractors. They slide across the 7-day window to detect spatial patterns and local variations in SMART attributes (e.g., a sudden spike in reallocated sectors).
2.  **LSTM (Long Short-Term Memory)**: LSTMs are designed to remember long-term dependencies. They process the features extracted by the CNN to understand the temporal progression—identifying if a drive is slowly degrading over time.

## Preprocessing Logic
### 1. Windowing (7-Day Sequence)
Instead of feeding the model one row per drive, we transform the data into "sequences." 
- **Input Shape**: `(samples, 7, 16)`
- Each sample contains the last 7 days of history. This allows the model to "see" the history of the drive before making a prediction for day 8.

### 2. Robust Scaling
We use `RobustScaler`, which removes the median and scales the data according to the Interquartile Range (IQR). This is critical for SMART data because failures often manifest as extreme outliers; standard scaling would squash these important signals.

## Hyperparameter Tuning Results (Optuna)
Using Optuna, we searched through multiple configurations to find the optimal balance:
- **Best Filters**: 64 (balanced feature extraction)
- **Best LSTM Units**: 127 (high capacity for temporal patterns)
- **Dropout**: 0.40 (to prevent overfitting on the rare failure class)
- **Learning Rate**: ~0.0025

This combination provided the highest **Recall** on the validation set, ensuring we catch as many failures as possible.
