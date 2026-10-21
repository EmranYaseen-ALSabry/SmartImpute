import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add SmartImpute src to path
sys.path.insert(0, str(Path("SmartImpute/src").resolve()))

from smartimpute.evaluation import ImputationEvaluator
from smartimpute.strategies.numeric import MeanImputer, MedianImputer, GroupBasedNumericImputer
from smartimpute.strategies.categorical import ModeImputer, ConstantCategoryImputer, GroupBasedCategoricalImputer

print("Loading 20,000 rows sample from data/train.csv...")
df_sample = pd.read_csv("data/train.csv", nrows=20000)

base_cols = [
    'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure', 
    'step_count', 'exercise_duration', 'water_intake', 'diet_type', 
    'stress_level', 'sleep_quality', 'physical_activity_level', 
    'smoking_alcohol', 'gender'
]

evaluator = ImputationEvaluator(mask_ratio=0.15, n_seeds=2, random_state=42)

print("\n--- Empirical Benchmark on Competition Features ---")
results = {}

# 1. sleep_duration
res_sleep = evaluator.evaluate_column_strategies(df_sample, column='sleep_duration', candidate_strategies=['median', 'mean'])
# Test group-based for sleep_duration using sleep_quality
grp_sleep = GroupBasedNumericImputer(group_column='sleep_quality', strategy='median')
# measure grp_sleep on masked sleep_duration
mask_idx = np.random.RandomState(42).choice(df_sample[df_sample['sleep_duration'].notna()].index, size=1500, replace=False)
df_masked = df_sample.copy()
true_sleep = df_sample.loc[mask_idx, 'sleep_duration']
df_masked.loc[mask_idx, 'sleep_duration'] = np.nan
grp_sleep.fit(df_masked)
pred_grp_sleep = grp_sleep.transform(df_masked).loc[mask_idx, 'sleep_duration']
grp_rmse = float(np.sqrt(np.mean((true_sleep - pred_grp_sleep)**2)))
res_sleep['strategies']['group_sleep_quality'] = {'metrics': {'rmse': {'mean': round(grp_rmse, 4)}}}

print("sleep_duration results:")
for k, v in res_sleep['strategies'].items():
    print(f"  {k}: RMSE={v['metrics'].get('rmse', {}).get('mean')}")

# 2. calorie_expenditure
res_cals = evaluator.evaluate_column_strategies(df_sample, column='calorie_expenditure', candidate_strategies=['median', 'mean'])
grp_cals = GroupBasedNumericImputer(group_column='physical_activity_level', strategy='median')
true_cals = df_sample.loc[mask_idx, 'calorie_expenditure']
df_masked_cals = df_sample.copy()
df_masked_cals.loc[mask_idx, 'calorie_expenditure'] = np.nan
grp_cals.fit(df_masked_cals)
pred_grp_cals = grp_cals.transform(df_masked_cals).loc[mask_idx, 'calorie_expenditure']
grp_cals_rmse = float(np.sqrt(np.mean((true_cals.dropna() - pred_grp_cals.loc[true_cals.dropna().index])**2)))
res_cals['strategies']['group_activity'] = {'metrics': {'rmse': {'mean': round(grp_cals_rmse, 4)}}}

print("\ncalorie_expenditure results:")
for k, v in res_cals['strategies'].items():
    print(f"  {k}: RMSE={v['metrics'].get('rmse', {}).get('mean')}")

# 3. Categorical test: stress_level & sleep_quality
res_cat = evaluator.evaluate_column_strategies(df_sample, column='stress_level', candidate_strategies=['mode', 'constant'])
print("\nstress_level results:")
for k, v in res_cat['strategies'].items():
    print(f"  {k}: F1={v['metrics'].get('macro_f1', {}).get('mean')}")
