# SmartImpute: Intelligent Missing Data Imputation System

**SmartImpute** is an enterprise-grade, leakage-free automated missing data imputation system. Rather than relying on naive assumptions (such as replacing every missing value with the mean or median), SmartImpute dynamically benchmarks multiple multivariate and similarity-based strategies using **Artificial Masking** on verified observed rows, and chooses the optimal strategy per column.

---

## 🚀 Key Features

1. **Scientific Strategy Selection (Artificial Masking):**
   - Hides subsets of observed ground truth across multiple random seeds to measure reconstruction error (RMSE, MAE for numeric; Accuracy, Macro-F1 for categorical).
   - Only selects strategies with empirically proven accuracy and stability.

2. **Rich Strategy Library:**
   - **Numeric:** Mean, Median, Constant, KNN (with scale normalization), Iterative (MICE with Bayesian Ridge), Group-based.
   - **Categorical:** Mode, Constant Category, Group-based, Model-based Decision Trees.
   - **Similarity-Based (`SimilarityBasedImputation`):** Gower normalized distance across mixed data types, neighbor reliability scoring, and full donor-level explainability.
   - **Time Series:** Causal forward fill, backward fill, and linear interpolation.

3. **Strict Zero Data Leakage:**
   - In Machine Learning Pipeline mode, the target column is strictly isolated and never used to impute features.
   - Full Scikit-Learn transformer interface (`fit`, `transform`, `fit_transform`).

4. **Production-Ready Persistence & Reporting:**
   - Save fitted imputation pipelines with `imputer.save('model.joblib')` and apply on test/inference sets with `SmartImputer.load('model.joblib').transform(test_df)`.
   - Comprehensive Markdown reports detailing profiling, diagnostics, and justifications.

---

## 📁 Project Architecture

```
SmartImpute/
│
├── pyproject.toml
├── README.md
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── input/
│   │   └── sample_health_data.csv
│   └── output/
│       └── cleaned_health.csv
│
├── reports/
│   └── health_report.md
│
├── src/
│   └── smartimpute/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py
│       ├── profiling.py
│       ├── missingness.py
│       ├── similarity.py
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── numeric.py
│       │   ├── categorical.py
│       │   └── timeseries.py
│       ├── evaluation.py
│       ├── selector.py
│       ├── optimization.py
│       ├── pipeline.py
│       ├── reporting.py
│       └── exceptions.py
│
└── tests/
    ├── test_data_loader.py
    ├── test_profiling.py
    ├── test_similarity.py
    ├── test_strategies.py
    ├── test_evaluation.py
    └── test_pipeline_and_leakage.py
```

---

## 🛠️ Installation

```bash
# Clone or navigate to the project directory
cd SmartImpute

# Install dependencies
pip install -e .

# Or install optional dev dependencies for testing
pip install -e .[dev]
```

---

## 💻 CLI Usage

SmartImpute provides an intuitive command-line interface:

### 1. Profile Dataset & Missingness
```bash
smartimpute analyze data/input/sample_health_data.csv --target target_health_score
```

### 2. Benchmark Imputation Strategies on a Column
```bash
smartimpute evaluate data/input/sample_health_data.csv --column weight
```

### 3. Automatically Impute & Generate Markdown Report
```bash
smartimpute impute data/input/sample_health_data.csv --output data/output/cleaned.csv --report reports/summary.md
```

### 4. ML Pipeline Mode (Zero Leakage)
```bash
smartimpute run data/input/sample_health_data.csv --target target_health_score --output data/output/cleaned_ml.csv
```

---

## 🐍 Python API Usage

### General Imputation Mode:
```python
import pandas as pd
from smartimpute import SmartImputer

df = pd.read_csv("data/input/sample_health_data.csv")

imputer = SmartImputer()
df_cleaned = imputer.fit_transform(df)

# Save pipeline for production deployment
imputer.save("saved_pipeline.joblib")
```

### ML Pipeline Mode with Inference:
```python
from smartimpute import SmartImputer
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Fit exclusively on training data
imputer = SmartImputer(target_column="target_health_score", mode="ml_pipeline")
train_cleaned = imputer.fit_transform(train_df)

# Save imputer
imputer.save("ml_imputer.joblib")

# Apply on test data without re-fitting (Zero Leakage)
loaded_imputer = SmartImputer.load("ml_imputer.joblib")
test_cleaned = loaded_imputer.transform(test_df)
```

---

## 🧪 Running Unit & Integration Tests

SmartImpute comes with a comprehensive suite of 23 unit and integration tests covering all 15 edge cases:

```bash
pytest SmartImpute/tests -v
```
