"""
Rigorous 3-Way Empirical Benchmark:
Method 1: Raw Baseline (Native NaNs left as is)
Method 2: Traditional Imputation (Global Median for numeric, Global Mode for categorical)
Method 3: SmartImpute (Adaptive Group/Similarity Imputation + missingness_features_count)

Evaluates on balanced_accuracy_score using 3-Fold Stratified Cross-Validation on competition data.
"""
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Add SmartImpute to path
sys.path.insert(0, str(Path("SmartImpute/src").resolve()))
from smartimpute.strategies.numeric import MedianImputer, GroupBasedNumericImputer
from smartimpute.strategies.categorical import ModeImputer

print("=" * 75)
print("  3-WAY EMPIRICAL BENCHMARK: RAW vs. TRADITIONAL vs. SMARTIMPUTE")
print("=" * 75)

# 1. Load representative subset for fast & rigorous CV
SAMPLE_SIZE = 50000
print(f"\n[1/4] Loading {SAMPLE_SIZE:,} stratified sample from data/train.csv...")
df_full = pd.read_csv("data/train.csv")
# Sample with fixed random state for reproducibility
df_sample = df_full.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

TARGET_COL = "health_condition"
base_feature_cols = [
    'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure', 
    'step_count', 'exercise_duration', 'water_intake', 'diet_type', 
    'stress_level', 'sleep_quality', 'physical_activity_level', 
    'smoking_alcohol', 'gender'
]

le = LabelEncoder()
y_true = le.fit_transform(df_sample[TARGET_COL])
class_names = list(le.classes_)
print(f"Target distribution: {dict(zip(class_names, np.bincount(y_true)))}")

# Feature Engineering Function
def build_features(df_input, include_missing_count=False):
    df_feat = df_input.copy()
    stress_map = {'low': 0, 'medium': 1, 'high': 2}
    df_feat['stress_level_num'] = df_feat['stress_level'].map(stress_map)
    
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    df_feat['sleep_quality_num'] = df_feat['sleep_quality'].map(sleep_map)
    
    act_map = {'sedentary': 0, 'moderate': 1, 'active': 2}
    df_feat['act_num'] = df_feat['physical_activity_level'].map(act_map)
    
    smoke_map = {'no': 0, 'occasional': 1, 'yes': 2}
    df_feat['smoke_num'] = df_feat['smoking_alcohol'].map(smoke_map)

    # Key interaction ratios
    df_feat['sleep_debt_stress'] = (8.0 - df_feat['sleep_duration']) * (df_feat['stress_level_num'] + 1.0)
    df_feat['cals_per_heart'] = df_feat['calorie_expenditure'] / (df_feat['heart_rate'] + 1e-5)
    df_feat['cals_per_step'] = df_feat['calorie_expenditure'] / (df_feat['step_count'] + 1.0)
    df_feat['water_per_bmi'] = df_feat['water_intake'] / (df_feat['bmi'] + 1e-5)

    if include_missing_count and 'missing_features_count' in df_feat.columns:
        # Keep missingness signal
        pass

    # Convert categoricals to category dtype for LightGBM
    cat_cols = ['diet_type', 'stress_level', 'sleep_quality', 'physical_activity_level', 'smoking_alcohol', 'gender']
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype('category')

    features = [c for c in df_feat.columns if c not in ['id', TARGET_COL]]
    return df_feat[features]

# Prepare datasets for 3 methods:

print("\n[2/4] Preparing data representations for all 3 methods...")

# --- Method 1: Raw (Native NaNs) ---
df_raw = df_sample.copy()

# --- Method 2: Traditional Imputation (Global Median & Mode) ---
df_traditional = df_sample.copy()
for col in base_feature_cols:
    if pd.api.types.is_numeric_dtype(df_traditional[col]):
        med = df_traditional[col].median()
        df_traditional[col] = df_traditional[col].fillna(med)
    else:
        m = df_traditional[col].mode().iloc[0]
        df_traditional[col] = df_traditional[col].fillna(m)

# --- Method 3: SmartImpute (Adaptive Group/Similarity + missing_features_count) ---
df_smart = df_sample.copy()
df_smart['missing_features_count'] = df_sample[base_feature_cols].isna().sum(axis=1)

# Group-based imputers
imputers = {
    'sleep_duration': GroupBasedNumericImputer(group_column='sleep_quality', strategy='median'),
    'calorie_expenditure': GroupBasedNumericImputer(group_column='physical_activity_level', strategy='median'),
    'step_count': GroupBasedNumericImputer(group_column='physical_activity_level', strategy='median'),
    'exercise_duration': GroupBasedNumericImputer(group_column='physical_activity_level', strategy='median'),
    'heart_rate': MedianImputer(),
    'bmi': MedianImputer(),
    'water_intake': MedianImputer(),
    'diet_type': ModeImputer(),
    'stress_level': ModeImputer(),
    'sleep_quality': ModeImputer(),
    'physical_activity_level': ModeImputer(),
    'smoking_alcohol': ModeImputer(),
    'gender': ModeImputer(),
}
for col, imp in imputers.items():
    imp.columns = [col]
    imp.fit(df_smart[base_feature_cols])
    df_smart = imp.transform(df_smart)

# Build features for each
X_raw = build_features(df_raw, include_missing_count=False)
X_trad = build_features(df_traditional, include_missing_count=False)
X_smart = build_features(df_smart, include_missing_count=True)

# 3. Stratified K-Fold CV Evaluation
print("\n[3/4] Running 3-Fold Stratified Cross Validation on all 3 methods...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def evaluate_method(X_data, y_data, method_name):
    t0 = time.time()
    balanced_accs = []
    macro_f1s = []

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'learning_rate': 0.08,
        'num_leaves': 31,
        'n_estimators': 150,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 2,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        X_tr, y_tr = X_data.iloc[train_idx], y_data[train_idx]
        X_va, y_val = X_data.iloc[val_idx], y_data[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_va)
        bacc = balanced_accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")

        balanced_accs.append(bacc)
        macro_f1s.append(f1)

    elapsed = round(time.time() - t0, 2)
    return {
        'name': method_name,
        'bacc_mean': np.mean(balanced_accs),
        'bacc_std': np.std(balanced_accs),
        'f1_mean': np.mean(macro_f1s),
        'f1_std': np.std(macro_f1s),
        'time_sec': elapsed,
    }

res_raw = evaluate_method(X_raw, y_true, "1. Raw (Native NaNs)")
print(f"  [Method 1 Done] Raw: Balanced Accuracy = {res_raw['bacc_mean']:.5f} (+/- {res_raw['bacc_std']:.5f}) in {res_raw['time_sec']}s")

res_trad = evaluate_method(X_trad, y_true, "2. Traditional (Global Median/Mode)")
print(f"  [Method 2 Done] Traditional: Balanced Accuracy = {res_trad['bacc_mean']:.5f} (+/- {res_trad['bacc_std']:.5f}) in {res_trad['time_sec']}s")

res_smart = evaluate_method(X_smart, y_true, "3. SmartImpute (Adaptive + Signal)")
print(f"  [Method 3 Done] SmartImpute: Balanced Accuracy = {res_smart['bacc_mean']:.5f} (+/- {res_smart['bacc_std']:.5f}) in {res_smart['time_sec']}s")

# 4. Summary Table & Comparison
print("\n" + "=" * 75)
print("                      FINAL BENCHMARK COMPARISON TABLE")
print("=" * 75)
print(f"{'Method':<35} | {'Balanced Accuracy':<20} | {'Macro F1':<12} | {'Time (s)':<8}")
print("-" * 75)
for res in [res_raw, res_trad, res_smart]:
    bacc_str = f"{res['bacc_mean']:.5f} (+/- {res['bacc_std']:.4f})"
    f1_str = f"{res['f1_mean']:.5f}"
    print(f"{res['name']:<35} | {bacc_str:<20} | {f1_str:<12} | {res['time_sec']:<8}")
print("=" * 75)
