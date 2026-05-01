# Drive Failure System — Line-by-Line Code Explanation

---

## SECTION 0 — Imports & Logger (Lines 1–75)

---

### Lines 1–29 — Module Docstring
```python
"""
drive_failure_system.py
...
"""
```
This is a **docstring** — a plain-English description of the entire file.
It is not executed; Python only reads it if you call `help(module)`.
It describes:
- What the system does (predict failures 14 days in advance)
- The full data flow from CSV → final prediction
- Key design decisions (why Polars, why time-ordered splits, etc.)

---

### Lines 34–42 — Standard Library Imports
```python
import os        # file paths, check if file exists, make directories
import glob      # find all CSV files matching a pattern like "*.csv"
import time      # measure how long things take (time.time())
import logging   # structured log messages instead of print()
import datetime  # work with date objects for the time split
import warnings  # suppress or show library warnings
import joblib    # save/load Python objects (models) to disk
from dataclasses import dataclass, field  # clean config class
from typing import List, Dict, Tuple, Optional  # type hints for readability
```
These are all built into Python — no installation needed.

---

### Lines 44–48 — Data & Plotting Libraries
```python
import numpy as np        # fast numeric arrays, math operations
import pandas as pd       # DataFrame for sklearn/model training ONLY
import polars as pl       # main data engine — faster and lighter than pandas
import matplotlib.pyplot as plt  # drawing charts
import seaborn as sns     # prettier charts built on top of matplotlib
```
**Why two DataFrame libraries (pandas AND polars)?**
Polars is used for all data work (10M rows, 3GB).
Pandas is used only for the small feature matrix passed to sklearn/LightGBM,
because those libraries only accept pandas/numpy arrays, not Polars frames.

---

### Lines 50–57 — Model Libraries + CatBoost Guard
```python
import lightgbm as lgb   # LightGBM gradient boosting
import xgboost as xgb    # XGBoost gradient boosting
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed — skipping CatBoost model.")
```
`try/except ImportError` is a **graceful degradation** pattern.
If CatBoost is not installed, the pipeline still runs with LightGBM + XGBoost.
`CATBOOST_AVAILABLE` is a flag checked later before trying to use CatBoost.
Without this guard, the whole file would crash on import if CatBoost is missing.

---

### Lines 59–67 — Sklearn Imports
```python
import shap  # explain model predictions feature-by-feature

from sklearn.linear_model import LogisticRegression   # simple meta-model option
from sklearn.preprocessing import StandardScaler       # normalise meta-features
from sklearn.metrics import (
    roc_auc_score,            # area under ROC curve
    average_precision_score,  # area under Precision-Recall curve (PRIMARY metric)
    precision_recall_curve,   # gives P/R at every possible threshold
    confusion_matrix,         # TP/FP/FN/TN table
    classification_report,    # full text report with all metrics
    f1_score,                 # harmonic mean of precision and recall
    recall_score,             # fraction of actual failures caught
    precision_score,          # fraction of alarms that are real failures
)
from sklearn.feature_selection import mutual_info_classif  # rank features by relevance
```

---

### Lines 70–75 — Logger Setup
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DriveFailure")
```
`logging.basicConfig(...)` configures the global logger once for the whole program.

- `level=logging.INFO` — show INFO, WARNING, ERROR messages (skip DEBUG)
- `format=...` — every log line looks like:
  `2025-12-01 14:23:05 | INFO     | DriveFailure | Loaded: 1,200,000 rows`
- `datefmt=...` — timestamp format
- `log = logging.getLogger("DriveFailure")` — creates a named logger
  so all messages from this module are tagged "DriveFailure"

**Why logging instead of print()?**
`print()` has no timestamp, no level, can't be filtered, and can't be redirected
to a file. In production you always use logging.

---

## SECTION 1 — PipelineConfig (Lines 78–155)

---

### Lines 81–83 — Dataclass Declaration
```python
@dataclass
class PipelineConfig:
    """All tunable knobs live here — change once, propagates everywhere."""
```
`@dataclass` is a Python decorator that auto-generates `__init__`, `__repr__`,
and `__eq__` from the class fields. Without `@dataclass` you would need to
write `def __init__(self, raw_csv_dir=..., merged_csv=..., ...)` manually.
The benefit is: all 25+ settings are visible in one place.

---

### Lines 86–89 — Path Settings
```python
raw_csv_dir: str   = r"E:\WE Work\..."  # folder with raw daily CSVs (utility only)
merged_csv: str    = "seagate_full_data.csv"
cleaned_csv: str   = "seagate_cleaned_cols.csv"  # ← main entry point
artifacts_dir: str = "artifacts"
```
The `r"..."` prefix means **raw string** — backslashes are not treated as
escape characters. `r"C:\new\file"` is the path literally, not `C:` + newline + `file`.
`cleaned_csv` is the file the normal pipeline reads from.
`artifacts_dir` is the folder where trained models are saved.

---

### Lines 92–93 — Data Filter Settings
```python
model_prefix: str   = "ST"    # only keep drives whose model starts with "ST"
null_threshold: float = 0.80  # drop any column that is 80%+ null
```

---

### Lines 96 — Target Definition
```python
prediction_horizon: int = 14  # predict failures within 14 days
```
This single number controls what "failure" means in the target column.
Changing it to `7` would predict 7-day-ahead failures, `30` for 30 days.

---

### Lines 99–100 — Downsampling Settings
```python
healthy_drive_multiplier: int = 3  # keep 3 healthy drives per failed drive
seed: int = 42                     # random seed for all sampling operations
```
The multiplier controls the class ratio at DRIVE level.
`seed=42` ensures every random operation is reproducible.

---

### Lines 103–109 — Feature Engineering Settings
```python
rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
lag_days: List[int]        = field(default_factory=lambda: [1, 3, 7])
smart_features: List[str]  = field(default_factory=lambda: [...])
```
`field(default_factory=lambda: [...])` is required for mutable defaults
(lists, dicts) in dataclasses. In Python, you cannot write `x: list = [1, 2]`
in a dataclass because that same list object would be shared across all instances.
`default_factory` creates a fresh list for each new `PipelineConfig()` object.

`rolling_windows = [7, 14, 30]` → rolling features computed over 7, 14, and 30 days.
`lag_days = [1, 3, 7]` → lag features: value from 1, 3, and 7 days ago.
`smart_features` → the specific SMART attributes to compute rolling/lag stats on.

---

### Lines 112–113 — Feature Selection Settings
```python
mi_sample_n: int    = 200_000  # rows to use for mutual information calculation
top_n_features: int = 35       # keep only the 35 most informative features
```
`200_000` — Python allows underscores in numbers for readability (same as 200000).
MI on 200K rows is statistically equivalent to MI on 10M rows but 50× faster.

---

### Lines 116–118 — Time Split Boundaries
```python
val_start: datetime.date  = datetime.date(2025, 11, 16)
test_start: datetime.date = datetime.date(2025, 12,  1)
end_date: datetime.date   = datetime.date(2025, 12, 17)  # right-censor cutoff
```
The three boundaries define the three splits:
- Train:  Oct 1 → Nov 15 (everything before `val_start`)
- Val:    Nov 16 → Nov 30 (between `val_start` and `test_start`)
- Test:   Dec 1 → Dec 16 (between `test_start` and `end_date`)

`end_date` is the right-censor cutoff. Rows from Dec 17–31 are dropped
because we don't know if those drives failed in January — their target
label would be wrong.

---

### Lines 121–122 — UnderBagging Settings
```python
n_lgbm_bags: int       = 5   # train 5 separate LightGBM models
desired_neg_ratio: int = 20  # 20 healthy rows per 1 failure row in each bag
```

---

### Lines 125–139 — Model Hyperparameters
```python
lgbm_params: Dict = field(default_factory=lambda: dict(
    n_estimators=1000,      # max trees to build
    learning_rate=0.05,     # how much each tree corrects the previous one
    num_leaves=63,          # tree complexity (63 = moderately deep)
    min_child_samples=50,   # min rows per leaf — prevents overfitting on large data
    scale_pos_weight=2,     # extra weight on positive (failure) class
    n_jobs=-1,              # use all CPU cores
    random_state=42,        # reproducibility
))
```
`scale_pos_weight=2` tells LightGBM: "treat each failure row as if it were
2 rows" — makes the model pay more attention to the rare failure class.

`xgb_params` and `catboost_params` follow the same idea with model-specific names:
- XGBoost uses `max_depth=6` instead of `num_leaves`
- CatBoost uses `auto_class_weights="Balanced"` which automatically calculates
  the right weight instead of you having to compute it manually

---

### Lines 142, 145, 148, 151–152 — Remaining Config Fields
```python
n_meta_folds: int       = 5     # OOF folds for stacking (explained in Section 9)
recall_target: float    = 0.85  # we want to catch at least 85% of all failures
shap_sample_n: int      = 1_000 # rows for SHAP (1000 is statistically sufficient)
alert_threshold: float  = 0.60  # probability > 0.60 triggers a potential alert
alert_consecutive_days: int = 2 # must be above threshold 2 days in a row to alert
```

---

### Lines 154–155 — Post-Init Hook
```python
def __post_init__(self):
    os.makedirs(self.artifacts_dir, exist_ok=True)
```
`__post_init__` runs automatically after the dataclass `__init__` finishes.
`os.makedirs(path, exist_ok=True)` creates the `artifacts/` folder if it
doesn't exist. `exist_ok=True` means: don't crash if the folder already exists.

---

## SECTION 2 — DataLoader (Lines 158–247)

---

### Lines 171–172 — Constructor
```python
def __init__(self, cfg: PipelineConfig):
    self.cfg = cfg
```
Stores the config object. `self.cfg` is how every method in this class
accesses settings like `self.cfg.cleaned_csv`.

---

### Lines 174–199 — `build_merged_csv()` (Utility Only)
```python
files = sorted(glob.glob(os.path.join(self.cfg.raw_csv_dir, "*.csv")))
```
`glob.glob("path/*.csv")` returns a list of all CSV filenames in the folder.
`sorted(...)` sorts them alphabetically — since filenames are dates
(e.g. `2025-10-01.csv`), sorted order = chronological order.
`os.path.join(...)` builds the full path correctly on both Windows and Linux.

```python
if not files:
    raise FileNotFoundError(f"No CSV files found in {self.cfg.raw_csv_dir}")
```
**Fail fast** — if the folder is empty, stop immediately with a clear message
instead of silently producing an empty dataset.

```python
lf = (
    pl.scan_csv(fp, infer_schema_length=0)
    .filter(pl.col("model").str.starts_with(self.cfg.model_prefix))
)
frames.append(lf)
```
`pl.scan_csv(fp, infer_schema_length=0)` — creates a **lazy query** (nothing
is read yet). `infer_schema_length=0` tells Polars to read all columns as strings
first, avoiding schema-conflict errors when different daily files have different
column orders or missing columns.
`.filter(...)` keeps only rows where the `model` column starts with `"ST"`.
`.append(lf)` collects all lazy queries into a list — no data in memory yet.

```python
pl.concat(frames).sink_csv(self.cfg.merged_csv)
```
`pl.concat(frames)` merges all 92 lazy queries into one logical query.
`.sink_csv(...)` executes the whole query and writes directly to disk in
micro-batches — the full 10M rows never live in RAM simultaneously.

```python
size_mb = os.path.getsize(self.cfg.merged_csv) / 1_048_576
```
`os.path.getsize()` returns file size in bytes. Dividing by 1,048,576
(= 1024²) converts to megabytes.

---

### Lines 201–216 — `drop_high_null_columns()` (Utility Only)
```python
lf = pl.scan_csv(self.cfg.merged_csv)
total_rows = lf.select(pl.len()).collect().item()
```
`pl.scan_csv()` — lazy again, no data loaded.
`pl.len()` — count total rows.
`.collect()` — execute just this count query (fast, reads no column data).
`.item()` — extract the single number from the 1×1 result frame.

```python
null_counts = lf.select(pl.all().null_count()).collect().row(0, named=True)
```
`pl.all().null_count()` — for every column, count nulls.
`.row(0, named=True)` — extract the first (only) row as a Python dictionary:
`{"smart_1_raw": 4200, "smart_10_raw": 9800000, ...}`

```python
keep = [col for col, nc in null_counts.items()
        if nc / total_rows < self.cfg.null_threshold]
```
Loop over every column's null count. Keep it only if null fraction < 80%.

```python
lf.select(keep).sink_csv(self.cfg.cleaned_csv)
```
Re-scan the merged CSV, select only the kept columns, write to new file.

---

### Lines 218–247 — `load()` — Normal Entry Point
```python
if not os.path.exists(self.cfg.cleaned_csv):
    raise FileNotFoundError(...)
```
Check the file exists before trying to read it.
The error message includes exact instructions on how to create the file.

```python
lf = pl.scan_csv(self.cfg.cleaned_csv)
smart_cols = [c for c in lf.columns if c.startswith("smart_")]
```
`lf.columns` is available on a lazy frame without reading any data —
Polars reads just the header line to get column names.
List comprehension keeps only column names that start with `"smart_"`.

```python
df = lf.with_columns(
    pl.col("date").str.to_date("%Y-%m-%d"),   # "2025-10-01" string → Date type
    pl.col("failure").cast(pl.Int8),           # 0/1 string → 1-byte integer
    *[pl.col(c).cast(pl.Float32, strict=False) for c in smart_cols],
).collect()
```
`with_columns(...)` adds or replaces columns.
`str.to_date("%Y-%m-%d")` parses date strings into actual Date objects,
enabling date arithmetic later.
`cast(pl.Float32, strict=False)` converts SMART columns from string to
32-bit float. `strict=False` means: if a value can't be parsed (e.g. empty
string), produce `null` instead of crashing.
`*[...]` — the asterisk **unpacks** the list of expressions so `with_columns`
receives them as separate arguments, not as a single list argument.
`.collect()` — execute the lazy query and load the result into RAM.

```python
df = df.sort(["serial_number", "date"])
```
Sort by drive ID first, then by date within each drive.
This is essential for all rolling/lag features — they assume chronological order.

---

## SECTION 3 — Preprocessor (Lines 250–330)

---

### Lines 264–278 — `remove_post_failure_rows()`
```python
df = df.with_columns(
    pl.col("date")
      .filter(pl.col("failure") == 1)   # look only at failure rows
      .min()                             # find the earliest failure date
      .over("serial_number")             # do this separately per drive
      .alias("_first_fail")
)
```
`.over("serial_number")` is Polars' window function — like SQL `PARTITION BY`.
For drive A: find the minimum date where `failure == 1`.
For drive B: same, independently.
Healthy drives (never failed) get `null` for `_first_fail`.

```python
df = df.filter(
    pl.col("_first_fail").is_null() |             # keep all rows for healthy drives
    (pl.col("date") <= pl.col("_first_fail"))      # keep only pre-failure rows for failed drives
).drop("_first_fail")
```
For a drive that failed on Dec 5: keep Oct 1, Oct 2, ..., Dec 5. Drop Dec 6–31.
**Why?** Rows after failure are useless for predicting failure — the event already happened.
`.drop("_first_fail")` removes the helper column since we don't need it anymore.

---

### Lines 280–303 — `create_target()`
```python
h = self.cfg.prediction_horizon  # = 14
df = df.with_columns(
    pl.col("failure")
      .reverse()                         # flip the time order
      .rolling_max(window_size=h, min_periods=1)  # look at next 14 rows
      .reverse()                         # flip back
      .over("serial_number")
      .fill_null(0)
      .cast(pl.Int8)
      .alias("target_14d")
)
```
**The trick explained step by step:**

Imagine a drive with this `failure` column: `[0, 0, 0, 1, 0]`
(failed on day 4, data continues after for illustration)

After `.reverse()`: `[0, 1, 0, 0, 0]`

After `.rolling_max(window_size=3)`: `[1, 1, 1, 0, 0]`
(rolling max looks at 3 rows going forward in the reversed array)

After second `.reverse()`: `[0, 0, 1, 1, 1]`

This means: rows 3, 4, 5 are labelled as "failure within 3 days" — correct!
`.fill_null(0)` — drives with no failures get 0 everywhere.
`.cast(pl.Int8)` — store as 1-byte integer (0 or 1), saves RAM.

---

### Lines 305–324 — `downsample_drives()`
```python
failed_serials = df.filter(pl.col("failure") == 1).select("serial_number").unique()
n_failed       = failed_serials.height
```
Find all unique serial numbers of drives that ever failed.
`.height` = number of rows (equivalent to `len(df)` in pandas).

```python
healthy_serials = df.join(failed_serials, on="serial_number", how="anti")
                    .select("serial_number").unique()
```
`how="anti"` is an **anti-join**: keep rows from the left table that have
NO match in the right table. In SQL: `WHERE serial_number NOT IN (failed_serials)`.
This gives us only the drives that never failed.

```python
n_healthy_sample = min(n_failed * self.cfg.healthy_drive_multiplier, healthy_serials.height)
sampled_healthy  = healthy_serials.sample(n=n_healthy_sample, seed=self.cfg.seed)
```
`min(...)` prevents asking for more drives than exist.
`.sample(n=..., seed=...)` randomly picks `n` rows with a fixed seed for reproducibility.

```python
selected = pl.concat([failed_serials, sampled_healthy])
df_out   = df.join(selected, on="serial_number", how="semi")
```
`how="semi"` = keep only left-table rows whose `serial_number` appears in the right table.
This is a **semi-join** — the opposite of anti-join.
Result: full history of all selected drives, nothing from unselected drives.

---

## SECTION 4 — FeatureEngineer (Lines 333–429)

---

### Lines 352–353 — `_resolve_smarts()`
```python
def _resolve_smarts(self, df: pl.DataFrame) -> List[str]:
    return [c for c in self.cfg.smart_features if c in df.columns]
```
The leading underscore `_resolve_smarts` means "private method" — it is
only called from within this class, not from outside.
This guard prevents crashes if a SMART column configured in settings
doesn't actually exist in the data (e.g. after null-dropping removed it).

---

### Lines 355–361 — `add_temporal_features()`
```python
(pl.int_range(pl.len()).over("serial_number") + 1).alias("drive_age_days")
```
`pl.int_range(pl.len())` → `[0, 1, 2, 3, ...]` for the whole DataFrame.
`.over("serial_number")` → restart at 0 for each drive.
`+ 1` → make it 1-indexed: first day = age 1, second day = age 2.

```python
pl.col("date").dt.weekday().cast(pl.Int8).alias("day_of_week")
pl.col("date").dt.month().cast(pl.Int8).alias("month")
```
`.dt.weekday()` → 0=Monday, 6=Sunday.
These capture seasonal patterns (e.g. drives fail more in summer due to heat).

---

### Lines 363–373 — `add_rolling_features()`
```python
for col in smart_cols:
    for w in self.cfg.rolling_windows:  # [7, 14, 30]
        exprs += [
            pl.col(col).rolling_mean(w, min_periods=1).over("serial_number")...
            pl.col(col).rolling_std(w,  min_periods=2).over("serial_number")...
            pl.col(col).rolling_max(w,  min_periods=1).over("serial_number")...
            pl.col(col).rolling_min(w,  min_periods=1).over("serial_number")...
        ]
return df.with_columns(*exprs)
```
`rolling_mean(7)` — average of the last 7 days. Smooths out noise.
`rolling_std(7)` — standard deviation of last 7 days. High std = volatile = suspicious.
`rolling_max / rolling_min` — worst and best reading in the window.
`min_periods=1` → compute even if fewer than 7 days of history exist.
`min_periods=2` → need at least 2 values to compute std (otherwise it's undefined).

All expressions are collected in the `exprs` list and applied in **one single
`with_columns()` call** — more efficient than calling `with_columns()` in a loop.

---

### Lines 375–382 — `add_lag_features()`
```python
pl.col(col).shift(lag).over("serial_number").alias(f"{col}_lag{lag}")
```
`.shift(lag)` — shift the column down by `lag` positions.
Result: for each row, `_lag7` = what the value was 7 days ago for this drive.
This tells the model: "has this SMART value been rising for a week?"

---

### Lines 384–396 — `add_trend_features()`
```python
lag7  = pl.col(col).shift(7).over("serial_number")
delta = pl.col(col) - lag7
rate  = delta / (lag7.abs() + 1e-6)
```
`delta` = today's value minus the value 7 days ago → direction and magnitude of change.
`rate` = delta divided by the magnitude of the old value → percentage change.
`+ 1e-6` → adds a tiny number (0.000001) to avoid dividing by zero when `lag7 = 0`.

---

### Lines 398–404 — `add_missingness_indicators()`
```python
pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_missing")
```
For each SMART column, create a 0/1 column: was this value missing today?
**Why?** A missing SMART value is often informative — it sometimes means the
drive's firmware stopped reporting because it was malfunctioning.
This gives the model a signal even when the actual value is unavailable.

---

### Lines 406–414 — `add_lifetime_aggregates()`
```python
pl.col(col).cum_max().over("serial_number").alias(f"{col}_cummax")
pl.col(col).cum_min().over("serial_number").alias(f"{col}_cummin")
```
`cum_max()` = cumulative maximum: for each row, the highest value this drive
has ever reported for this SMART attribute up to today.
Captures: "has this drive ever had a bad sector? What was the worst temperature?"
This is a **lifetime** signal, not just recent history.

---

## SECTION 5 — FeatureSelector (Lines 432–511)

---

### Lines 436–451 — Module-Level Drop Lists
```python
_DEAD_COLS   = ["smart_10_raw", ...]   # constant columns — zero variance
_CORR_COLS   = [...]                   # highly correlated with other columns
_INFRA_COLS  = ["capacity_bytes", ...] # infrastructure metadata, not SMART health
_REDUND_NORM = [...]                   # normalized duplicates of raw columns we keep
```
These are module-level constants (not inside any class), marked with a
leading underscore to indicate "internal use only".
Putting them here keeps the `FeatureSelector` class clean.

---

### Lines 475–499 — `fit()`
```python
pos = train_pl.filter(pl.col("target_14d") == 1)
neg = train_pl.filter(pl.col("target_14d") == 0)
ratio    = len(pos) / max(len(train_pl), 1)
n_pos    = max(1, int(self.cfg.mi_sample_n * ratio))
n_neg    = self.cfg.mi_sample_n - n_pos
```
Stratified sampling: maintain the same failure rate in the sample as in
the full training set. If 1% of training rows are failures, 1% of the
200K sample will be failures. This prevents MI from being biased.

```python
sample = pl.concat([
    pos.sample(n=min(n_pos, len(pos)), seed=self.cfg.seed),
    neg.sample(n=min(n_neg, len(neg)), seed=self.cfg.seed),
]).to_pandas()
```
`min(n_pos, len(pos))` — don't ask for more positive samples than exist.
`.to_pandas()` — the only conversion to pandas here, on a small 200K-row slice.

```python
X_s = sample[feature_cols].select_dtypes(include=[np.number]).fillna(-1)
```
`select_dtypes(include=[np.number])` — keep only numeric columns.
Some engineered features may be strings or booleans; MI can only handle numbers.
`.fillna(-1)` — MI cannot handle NaN values; replace with -1 as a sentinel.

```python
mi = mutual_info_classif(X_s, y_s, random_state=self.cfg.seed)
self.mi_series = pd.Series(mi, index=X_s.columns).sort_values(ascending=False)
self.top_features = self.mi_series.head(self.cfg.top_n_features).index.tolist()
```
`mutual_info_classif` returns one score per column — higher = more informative.
`pd.Series(mi, index=X_s.columns)` — pair each score with its column name.
`.sort_values(ascending=False)` — best features first.
`.head(top_n_features).index.tolist()` — take the names of the top 35.

---

## SECTION 6 — Splitter (Lines 514–542)

---

### Lines 530–542 — `split()`
```python
df_clean = df.filter(pl.col("date") < self.cfg.end_date)
```
Drop the last 14 days. Those rows' labels are unreliable because we don't
know what happens after the dataset ends.

```python
train = df_clean.filter(pl.col("date") < self.cfg.val_start)
val   = df_clean.filter((pl.col("date") >= self.cfg.val_start) &
                         (pl.col("date") < self.cfg.test_start))
test  = df_clean.filter(pl.col("date") >= self.cfg.test_start)
```
Three non-overlapping, chronologically ordered slices.
`&` is the Polars bitwise AND for combining filter conditions.

```python
pos = split["target_14d"].sum()
log.info(f"... failure rate: {pos/max(len(split),1)*100:.3f}%")
```
`max(len(split), 1)` — avoid division by zero if a split is empty.
`.3f%` — format as percentage with 3 decimal places.

---

## SECTION 7 — Helper Functions (Lines 545–571)

---

### Lines 548–556 — `polars_to_xy()`
```python
cols  = [c for c in feature_cols + [target_col] if c in df_pl.columns]
df_pd = df_pl.select(cols).to_pandas()
X     = df_pd[feature_cols].astype("float32")
y     = df_pd[target_col].astype("int8")
```
**Minimal conversion** — select only the needed columns before converting.
Converting 10M rows × 200 columns to pandas would OOM; we convert only
the ~35 top features + 1 target.
`astype("float32")` — 32-bit float (4 bytes) instead of default 64-bit (8 bytes)
= 50% memory saving.
`astype("int8")` — 1-byte integer for 0/1 target.

---

### Lines 559–571 — `under_bag_sample()`
```python
pos_idx = np.where(y == 1)[0]   # indices of all failure rows
neg_idx = np.where(y == 0)[0]   # indices of all healthy rows
n_neg   = min(len(pos_idx) * desired_neg_ratio, len(neg_idx))
```
`np.where(condition)[0]` returns the array of index positions where condition is True.
`min(...)` — don't request more negative rows than exist.

```python
rng         = np.random.default_rng(seed)
neg_sampled = rng.choice(neg_idx, size=n_neg, replace=False)
idx         = np.concatenate([pos_idx, neg_sampled])
rng.shuffle(idx)
```
`np.random.default_rng(seed)` — modern numpy random generator, reproducible per seed.
`replace=False` — no duplicate rows.
`np.concatenate` — combine all failure rows + sampled healthy rows.
`rng.shuffle(idx)` — mix them so the model doesn't see all failures then all healthy.

---

## SECTION 8 — BaseModelTrainer (Lines 574–677)

---

### Lines 589–593 — Constructor
```python
self.lgbm_models: List = []   # will hold 5 LightGBM objects after fit_lgbm()
self.xgb_model         = None # will hold 1 XGBoost object
self.cat_model         = None # will hold 1 CatBoost object (if installed)
```
Starting as empty/None signals "not trained yet" — used in guard checks.

---

### Lines 596–618 — `fit_lgbm()`
```python
for i in range(self.cfg.n_lgbm_bags):  # i = 0, 1, 2, 3, 4
    X_b, y_b = under_bag_sample(..., seed=self.cfg.seed + i)
```
Each of the 5 bags gets a different seed (`42, 43, 44, 45, 46`), so each
samples a different random subset of healthy drives. The 5 models train on
different data but are averaged → **UnderBagging ensemble**.

```python
params = {**self.cfg.lgbm_params, "random_state": self.cfg.seed + i}
```
`{**self.cfg.lgbm_params, ...}` — dictionary unpacking. Copies all params
from config, then overrides `random_state` with the bag-specific seed.
This ensures each LightGBM model also has different internal randomness.

```python
clf.fit(
    X_b, y_b,
    eval_set=[(X_b, y_b), (X_val, y_val)],
    eval_metric="aucpr",
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(200),
    ],
)
```
`eval_set` — monitor performance on both train and val during training.
`eval_metric="aucpr"` — stop based on PR-AUC (our primary metric), not accuracy.
`early_stopping(50)` — stop training if val PR-AUC hasn't improved for 50 rounds.
This prevents overfitting and saves time.
`log_evaluation(200)` — print metrics every 200 trees.

---

### Lines 620–626 — `predict_lgbm()`
```python
proba_sum = np.zeros(len(X))
for m in self.lgbm_models:
    proba_sum += m.predict_proba(X)[:, 1]
return proba_sum / len(self.lgbm_models)
```
`predict_proba(X)` returns a 2-column array: `[[prob_healthy, prob_failure], ...]`
`[:, 1]` — take only column 1 = failure probability.
Sum all 5 models' probabilities then divide by 5 = average.
This averaging reduces variance and makes predictions more stable.

---

### Lines 629–638 — `fit_xgb()`
```python
self.xgb_model = xgb.XGBClassifier(**self.cfg.xgb_params)
self.xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=200,
)
```
`**self.cfg.xgb_params` — unpack the xgb_params dict as keyword arguments.
XGBoost uses the full imbalanced training data (not undersampled) because
`scale_pos_weight=2` handles imbalance through weighting instead.

---

### Lines 646–657 — `fit_catboost()`
```python
if not CATBOOST_AVAILABLE:
    log.warning("CatBoost not installed — skipping.")
    return
```
Guard check — if the import failed at the top of the file, skip gracefully.

```python
self.cat_model = CatBoostClassifier(**self.cfg.catboost_params)
self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
```
CatBoost's `eval_set` takes a tuple `(X, y)` not a list of tuples like LightGBM.
`auto_class_weights="Balanced"` — CatBoost computes optimal weights automatically.

---

### Lines 670–677 — `predict_all()`
```python
preds: Dict[str, np.ndarray] = {
    "lgbm": self.predict_lgbm(X),
    "xgb" : self.predict_xgb(X),
}
if self.cat_model is not None:
    preds["catboost"] = self.predict_catboost(X)
return preds
```
Returns a dictionary of `{"model_name": array_of_probabilities}`.
The stacking ensemble uses this dictionary to build its meta-features.
CatBoost is added only if it was actually trained.

---

## SECTION 9 — StackingEnsemble (Lines 680–769)

---

### Lines 702–710 — Constructor
```python
self.meta_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=15,
    scale_pos_weight=2, random_state=cfg.seed, n_jobs=-1,
)
self.scaler  = StandardScaler()
self._fitted = False
```
The meta-model is a smaller LightGBM (`num_leaves=15`, `n_estimators=200`)
because its input is only 2–3 numbers (the base model probabilities), not 35 features.
`StandardScaler` normalises the meta-features before passing to the meta-model.
`_fitted = False` → a guard to prevent predicting before training.

---

### Lines 712–714 — `_get_meta_features()`
```python
def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
    preds = self.base_trainer.predict_all(X)
    return np.column_stack(list(preds.values()))
```
`predict_all(X)` returns `{"lgbm": [0.1, 0.9, ...], "xgb": [0.2, 0.8, ...], ...}`
`list(preds.values())` → `[[0.1, 0.9, ...], [0.2, 0.8, ...]]`
`np.column_stack(...)` → 2D array where each column is one model's probabilities:
```
[[0.1, 0.2],   ← row 1: lgbm=0.1, xgb=0.2
 [0.9, 0.8],   ← row 2: lgbm=0.9, xgb=0.8
 ...]
```
This is the input to the meta-model.

---

### Lines 716–763 — `fit()` — The Stacking Core
```python
n         = len(train_pl)
fold_size = n // self.cfg.n_meta_folds  # e.g. 8M / 5 = 1.6M rows per fold
```
Integer division `//` gives whole number (no decimals).

```python
for fold in range(1, self.cfg.n_meta_folds):  # folds 1, 2, 3, 4 (not 0)
    cutoff        = fold * fold_size
    fold_train_pl = train_pl.slice(0, cutoff)        # rows 0 to cutoff
    fold_val_pl   = train_pl.slice(cutoff, fold_size) # rows cutoff to cutoff+fold_size
```
`train_pl.slice(start, length)` — slice by row position.
Fold 1: train on rows 0–1.6M, validate on rows 1.6M–3.2M
Fold 2: train on rows 0–3.2M, validate on rows 3.2M–4.8M
This is an **expanding window** — each fold's training set includes all
previous folds. This respects time order: future data never trains a model
that then predicts on past data.

```python
tmp_trainer = BaseModelTrainer(self.cfg)
tmp_trainer.fit_lgbm(X_ft, y_ft, X_fv, y_fv)
tmp_trainer.fit_xgb(X_ft, y_ft, X_fv, y_fv)
tmp_trainer.fit_catboost(X_ft, y_ft, X_fv, y_fv)
preds  = tmp_trainer.predict_all(X_fv)
meta_f = np.column_stack(list(preds.values()))
oof_meta.append(meta_f)
oof_label.append(y_fv.values)
```
For each fold: train temporary base models, predict on held-out fold.
These predictions are the **OOF (Out-Of-Fold) meta-features**.
They represent "what would the base models say about this row if they hadn't
seen it during training?" — honest, unbiased predictions.

```python
X_meta = np.vstack(oof_meta)         # stack all fold results vertically
y_meta = np.concatenate(oof_label)   # combine all fold labels
X_meta_scaled = self.scaler.fit_transform(X_meta)
```
`np.vstack` — vertical stack: `[[fold1_rows], [fold2_rows], ...]` → one big matrix.
`scaler.fit_transform` — learn mean/std from OOF meta-features and normalise them.
Scaling ensures LightGBM doesn't treat probability 0.9 as numerically more
important than 0.1 just because it's larger.

```python
X_val_meta = self.scaler.transform(self._get_meta_features(X_val))
self.meta_model.fit(
    X_meta_scaled, y_meta,
    eval_set=[(X_val_meta, y_val.values)],
    ...
)
```
Note: `.transform()` not `.fit_transform()` here — use the scaler already
fitted on OOF data. Do NOT refit on val data (that would be leakage).
The meta-model trains on OOF predictions and validates on val predictions.

---

### Lines 765–769 — `predict_proba()`
```python
meta_f = self.scaler.transform(self._get_meta_features(X))
return self.meta_model.predict_proba(meta_f)[:, 1]
```
At inference time: get base model probabilities → scale → meta-model → final probability.
Same scaler that was fitted during training — ensures consistent normalisation.

---

## SECTION 10 — Evaluator (Lines 772–916)

---

### Lines 788–804 — `find_best_threshold()`
```python
prec, rec, threshs = precision_recall_curve(y_true, y_proba)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
```
`precision_recall_curve` returns arrays of precision, recall, and threshold
values at every possible decision boundary.
`prec[:-1]` — sliced to match `threshs` length (sklearn returns one extra point).
`+ 1e-10` — prevents division by zero when both precision and recall are 0.

```python
mask = rec[:-1] >= self.cfg.recall_target  # True where recall >= 0.85
if mask.any():
    idx = np.argmax(f1 * mask)   # best F1 among thresholds that meet recall target
else:
    idx = np.argmax(f1)          # fallback: just best F1
```
`f1 * mask` — multiplies F1 by 1 (keep) or 0 (exclude) based on recall constraint.
`np.argmax(...)` — index of the maximum value.
Business logic: catching 85%+ of failures matters more than precision.

---

### Lines 834–872 — `drive_level_report()`
```python
df_agg = pd.DataFrame({"serial": serials, "proba": y_proba, "label": y_true})
drive  = df_agg.groupby("serial").agg(
    max_proba=("proba", "max"),  # worst-case probability for this drive
    failed   =("label", "max"), # did this drive ever fail? (max of 0/1 = 1 if any row failed)
).reset_index()
```
`groupby("serial").agg(...)` — collapse all rows for each drive into one summary row.
`max_proba` — the single highest probability the model assigned on any day.
`failed` — 1 if the drive failed at any point in the test window, 0 otherwise.

This converts the problem from "did we flag the right rows?" to
"did we flag the right drives?" — which is what matters in production.

---

### Lines 874–890 — `threshold_sweep_table()`
```python
candidates = np.linspace(0.3, 0.95, 14)
```
`np.linspace(start, stop, num)` — 14 evenly spaced values between 0.30 and 0.95.
For each threshold, compute how many failures were caught (TP), how many
false alarms raised (FP), and how many failures were missed (FN).
This gives the business team a table to pick the right operating point.

---

## SECTION 11 — Explainer (Lines 919–968)

---

### Lines 928–954 — `shap_summary()`
```python
pos_idx = y_test[y_test == 1].index
neg_idx = y_test[y_test == 0].index
n_pos   = min(100, len(pos_idx))
n_neg   = min(self.cfg.shap_sample_n - n_pos, len(neg_idx))
```
Stratified sampling for SHAP: 100 failure rows + 900 healthy rows.
Running SHAP on the full test set (1.7M rows) would take hours and OOM.
SHAP plots are statistically stable at 500–2000 samples.

```python
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
```
`TreeExplainer` uses the actual tree structure — fast, exact (not approximate).
Older SHAP versions return a list `[shap_for_class_0, shap_for_class_1]`.
`shap_values[1]` = SHAP values for the failure class.
Newer versions return a single array — `isinstance(..., list)` handles both.

---

## SECTION 12 — EarlyWarningSystem (Lines 971–1024)

---

### Lines 989–1014 — `generate_alerts()`
```python
df_scored = df_scored.sort_values(["serial_number", "date"]).copy()
```
`.copy()` — create a new DataFrame instead of modifying the input in place.
This is good practice to avoid pandas' `SettingWithCopyWarning`.

```python
df_scored["above_thresh"] = (
    df_scored["failure_prob"] >= self.cfg.alert_threshold
).astype(int)
```
Binary flag: is today's probability above 0.60?

```python
df_scored["consec_days_above"] = (
    df_scored.groupby("serial_number")["above_thresh"]
    .transform(lambda x: x.rolling(
        self.cfg.alert_consecutive_days, min_periods=self.cfg.alert_consecutive_days
    ).sum())
)
```
`groupby().transform()` — applies a function per group but returns a result
with the same shape as the original DataFrame (unlike `.agg()` which collapses).
`x.rolling(2, min_periods=2).sum()` — for each row, sum of current and previous day.
If both days are 1 (above threshold), sum = 2 = consecutive days threshold met.

```python
df_scored["alert"] = (
    df_scored["consec_days_above"] >= self.cfg.alert_consecutive_days
)
```
`alert = True` only where the rolling sum equals the required consecutive days.
Single-day spikes (sum = 1) do not trigger an alert.

---

### Lines 1016–1024 — `summarise_alerts()`
```python
alerted = df_alerts[df_alerts["alert"]].groupby("serial_number").agg(
    first_alert_date=("date",         "min"),  # earliest day the alert fired
    max_probability =("failure_prob", "max"),  # highest probability ever
    alert_days      =("alert",        "sum"),  # how many days the alert was active
).reset_index().sort_values("max_probability", ascending=False)
```
One row per alerted drive, sorted by highest probability first.
`("date", "min")` — named aggregation syntax: `(column, function)`.

---

## SECTION 13 — InferencePipeline (Lines 1027–1103)

---

### Lines 1053–1091 — `score()`
```python
df_feat = self.feat_eng.run(raw_df)
```
Apply exactly the same feature engineering as training.
**No target creation** — there is no `failure` column in new data.

```python
available = [c for c in self.selector.top_features if c in df_feat.columns]
if len(available) < len(self.selector.top_features):
    missing = set(self.selector.top_features) - set(available)
    log.warning(f"Inference: {len(missing)} feature(s) missing — filling with 0: {missing}")
```
**Schema validation** — check that all features the model was trained on
are present in the new data. Log a warning if any are missing.

```python
X = df_feat.select(available).to_pandas().astype("float32")
for feat in self.selector.top_features:
    if feat not in X.columns:
        X[feat] = 0.0
X = X[self.selector.top_features]
```
Fill missing features with 0 (neutral value).
`X = X[self.selector.top_features]` — enforce exact column order.
Models are sensitive to column order; this guarantees the order matches training.

```python
proba  = self.ensemble.predict_proba(X)
meta   = df_feat.select(["serial_number", "date"]).to_pandas()
result = meta.copy()
result["failure_prob"] = proba
result = self.ews.generate_alerts(result)
return result
```
Full scoring pipeline: features → stacking ensemble → probabilities →
attach to metadata → run early warning → return.

---

### Lines 1093–1103 — `save()` and `load()`
```python
def save(self, path: Optional[str] = None) -> str:
    path = path or os.path.join(self.cfg.artifacts_dir, "inference_pipeline.joblib")
    joblib.dump(self, path)
    return path
```
`path or ...` — if no path given, use the default artifacts directory.
`joblib.dump(self, path)` — serialize the ENTIRE `InferencePipeline` object
(including the FeatureEngineer, FeatureSelector, StackingEnsemble, and all
their trained models) to one file. One file = complete, portable pipeline.

```python
@staticmethod
def load(path: str) -> "InferencePipeline":
    pipeline = joblib.load(path)
    return pipeline
```
`@staticmethod` — does not receive `self` or `cls` as first argument.
Called as `InferencePipeline.load("path")` not `instance.load("path")`.
Used because there is no instance yet — we're creating one from disk.

---

## SECTION 14 — DriveFailurePipeline Orchestrator (Lines 1106–1237)

---

### Lines 1128–1143 — Constructor
```python
def __init__(self, cfg: Optional[PipelineConfig] = None):
    self.cfg = cfg or PipelineConfig()
```
`Optional[PipelineConfig]` means the argument can be either a `PipelineConfig`
object or `None`. `cfg or PipelineConfig()` means: if `cfg` is None, create
a default config automatically.

```python
self.loader       = DataLoader(self.cfg)
self.preprocessor = Preprocessor(self.cfg)
self.feat_eng     = FeatureEngineer(self.cfg)
...
```
Each component is instantiated and stored. They all receive the same `cfg`
object — this is **dependency injection**: components don't create their own
config, they receive it from outside. Makes testing and customisation easy.

```python
self.stacking:    Optional[StackingEnsemble]   = None
self.inference:   Optional[InferencePipeline]  = None
self.feature_cols: List[str]                   = []
```
These are set to None/empty before training. After `run()` completes they
will hold the trained objects. Explicitly declaring them here (rather than
creating them only inside `run()`) makes the class structure clear.

---

### Lines 1145–1237 — `run()` — The Master Orchestrator
```python
t_start = time.time()
```
Record start time; compute elapsed minutes at the end.

```python
df = self.loader.load()
```
Step 1: loads `seagate_cleaned_cols.csv` as typed Polars DataFrame.

```python
df = self.preprocessor.run(df)
```
Step 2: removes post-failure rows, creates 14-day target, downsamples drives.

```python
df = self.feat_eng.run(df)
```
Step 3: adds all rolling/lag/trend/missingness/lifetime features.

```python
train_pl, val_pl, test_pl = self.splitter.split(df)
```
Step 4: time-based split into three non-overlapping chronological sets.

```python
META = ["date", "serial_number", "model", "failure", "target_14d"]
all_feat_cols = [c for c in train_pl.columns if c not in META]
self.selector.fit(train_pl, all_feat_cols)
self.feature_cols = self.selector.top_features
self.selector.plot_mi()
```
Step 5: compute MI on training data only → select top 35 features → plot chart.
`META` columns are excluded from features: they are identifiers or labels,
not predictors.

```python
X_train, y_train = polars_to_xy(train_pl, self.feature_cols)
X_val,   y_val   = polars_to_xy(val_pl,   self.feature_cols)
X_test,  y_test  = polars_to_xy(test_pl,  self.feature_cols)
serials_test = test_pl.select("serial_number").to_pandas()["serial_number"].values
```
Step 6: convert Polars splits to typed pandas arrays for sklearn.
`serials_test` is saved separately for drive-level evaluation later.

```python
self.base_trainer.fit_all(X_train, y_train, X_val, y_val)
```
Step 7: trains all three base models (LGBM bags + XGB + CatBoost).

```python
self.stacking = StackingEnsemble(self.cfg, self.base_trainer)
self.stacking.fit(train_pl, self.feature_cols, X_val, y_val)
```
Step 8: builds OOF meta-features and trains meta-LightGBM on them.
Passes `self.base_trainer` (already trained on full train set) so the
stacking ensemble can use it for final predictions.

```python
lgbm_proba  = self.base_trainer.predict_lgbm(X_test)
xgb_proba   = self.base_trainer.predict_xgb(X_test)
stack_proba = self.stacking.predict_proba(X_test)
all_results = {
    "LightGBM (bags)": (y_test.values, lgbm_proba),
    "XGBoost"        : (y_test.values, xgb_proba),
    "Stacking"       : (y_test.values, stack_proba),
}
```
Step 9: generate test-set predictions for all models.
`y_test.values` — `.values` converts pandas Series to numpy array.
The dictionary structure allows looping over all models cleanly.

```python
for name, (yt, yp) in all_results.items():
    self.evaluator.row_level_report(yt, yp, label=name)
    self.evaluator.drive_level_report(yt, yp, serials_test, label=name)
```
Evaluate each model at both row level and drive level.
`(yt, yp)` — tuple unpacking: `yt` = true labels, `yp` = predicted probabilities.

```python
sweep = self.evaluator.threshold_sweep_table(y_test.values, stack_proba)
log.info("Threshold Sweep:\n" + sweep.to_string(index=False))
```
Print the full threshold trade-off table for the stacking model.
`sweep.to_string(index=False)` — convert DataFrame to a text table without row numbers.

```python
thresh, _ = self.evaluator.find_best_threshold(y_test.values, stack_proba)
cm = confusion_matrix(y_test.values, (stack_proba >= thresh).astype(int))
self.evaluator.plot_confusion_matrix(cm, ...)
```
`_` is a Python convention for "I need to unpack this tuple but I don't
need the second value" (`best_f1` in this case).

```python
best_lgbm = self.base_trainer.lgbm_models[-1]
```
`[-1]` — the last element of the list = the 5th bag.
Any of the 5 LightGBM bags can be used for SHAP; the last one is arbitrary.

```python
self.inference = InferencePipeline(
    cfg=self.cfg,
    feature_engineer=self.feat_eng,
    selector=self.selector,
    ensemble=self.stacking,
    preprocessor=self.preprocessor,
    ews=self.ews,
)
self.inference.save()
```
Step 11: bundle all trained components into one inference object and save to disk.
After this line, scoring new data requires only:
```python
inference = InferencePipeline.load("artifacts/inference_pipeline.joblib")
result = inference.score(new_data)
```

---

## SECTION 15 — Entry Point (Lines 1240–1250)

---

```python
if __name__ == "__main__":
```
This block runs ONLY when the file is executed directly:
`python drive_failure_system.py`

It does NOT run when the file is imported:
`from drive_failure_system import DriveFailurePipeline`

This is standard Python pattern for making a module both importable and runnable.

```python
cfg = PipelineConfig(
    cleaned_csv   = "seagate_cleaned_cols.csv",
    artifacts_dir = "artifacts",
    seed          = 42,
)
pipeline = DriveFailurePipeline(cfg)
pipeline.run()
```
Create config, create pipeline, run everything.
Only `cleaned_csv`, `artifacts_dir`, and `seed` are specified — all other
settings use their defaults from `PipelineConfig`.
