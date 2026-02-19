#!/usr/bin/env python3
"""
Model Training and Evaluation Script
=====================================

This script performs:
1. Data loading and preprocessing
2. Feature engineering (polynomial and interaction features)
3. K-Fold cross-validation scaling and setup
4. Training multiple regression models with K-Fold CV
5. Hyperparameter tuning with K-Fold CV
6. Model evaluation and comparison using CV scores
7. Training best model on entire dataset for test predictions

Goal: Achieve the highest possible R² score using K-Fold cross-validation

Data Flow: 
Load → Encode → Select Features → Feature Engineering → Scale (fit on all) → K-FOLD CV → 
Train Best Model → Predict → Save
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

import xgboost as xgb
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("MODEL TRAINING AND EVALUATION")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_POLY_FEATURES = True
USE_INTERACTIONS = True
POLY_DEGREE = 2
TOP_K_FOR_POLY = 5

SCALER_TYPE = "standard"  # "standard" or "robust"
USE_POWER_TRANSFORMER = False

RANDOM_STATE = 42

N_SPLITS = 5  # K-Fold cross-validation
TUNING_N_ITER = 5

print("[OK] Configuration loaded")
print(f"  Polynomial features: {USE_POLY_FEATURES}")
print(f"  Interaction features: {USE_INTERACTIONS}")
print(f"  Scaler type: {SCALER_TYPE}")
print(f"  K-Fold splits: {N_SPLITS}")
print(f"  Tuning iterations: {TUNING_N_ITER}")
print()

# ============================================================================
# 1. LOAD DATA AND SELECTED FEATURES
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_data = pd.read_csv('CW1_train.csv')
test_data = pd.read_csv('CW1_test.csv')

y_train = train_data['outcome']
X_train_full = train_data.drop('outcome', axis=1)
X_test = test_data.copy()

categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()

try:
    selected_features_df = pd.read_csv('selected_features.csv')
    selected_features = selected_features_df['feature'].tolist()
    print(f"[OK] Loaded {len(selected_features)} selected features from EDA")
except FileNotFoundError:
    print("[WARNING] selected_features.csv not found!")
    print("         Run EDA.py first to generate selected_features.csv")
    print("         Using all features as fallback...")
    X_train_full = pd.get_dummies(X_train_full, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    missing_cols = set(X_train_full.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train_full.columns]
    selected_features = X_train_full.columns.tolist()

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features to use: {len(selected_features)}")
print()

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================

print("=" * 80)
print("PREPARING FEATURES")
print("=" * 80)

X_train_encoded = pd.get_dummies(X_train_full, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Ensure test has same columns as train
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Create dimensions feature from x, y, z if they exist
if all(col in X_train_encoded.columns for col in ['x', 'y', 'z']):
    X_train_encoded['dimensions'] = X_train_encoded[['x', 'y', 'z']].mean(axis=1)
    X_test_encoded['dimensions'] = X_test_encoded[['x', 'y', 'z']].mean(axis=1)
    print("[OK] Created 'dimensions' feature from x, y, z")

# Filter selected features
target_var = 'outcome'
valid_selected_features = [f for f in selected_features 
                           if f in X_train_encoded.columns and f != target_var]

missing_features = [f for f in selected_features if f not in X_train_encoded.columns and f != target_var]
if missing_features:
    print(f"[WARNING] {len(missing_features)} selected features not found: {missing_features}")

X_train_selected = X_train_encoded[valid_selected_features]
X_test_selected = X_test_encoded[valid_selected_features]

print(f"[OK] Features prepared:")
print(f"  Training shape: {X_train_selected.shape}")
print(f"  Test shape: {X_test_selected.shape}")
print()

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

feature_target_corr = X_train_selected.corrwith(y_train).abs().sort_values(ascending=False)
top_features_for_poly = feature_target_corr.head(TOP_K_FOR_POLY).index.tolist()

X_train_poly_df = X_train_selected.copy()
X_test_poly_df = X_test_selected.copy()
new_poly_count = 0

if USE_POLY_FEATURES:
    poly_transformer = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False, interaction_only=False)
    poly_features = poly_transformer.fit_transform(X_train_selected[top_features_for_poly])
    poly_feature_names = poly_transformer.get_feature_names_out(top_features_for_poly)
    
    for idx, name in enumerate(poly_feature_names):
        if '^2' in name or '*' in name:
            new_name = f"poly_{name}"
            X_train_poly_df[new_name] = poly_features[:, idx]
            new_poly_count += 1
    
    poly_features_test = poly_transformer.transform(X_test_selected[top_features_for_poly])
    for idx, name in enumerate(poly_feature_names):
        if '^2' in name or '*' in name:
            new_name = f"poly_{name}"
            X_test_poly_df[new_name] = poly_features_test[:, idx]
    
    poly_info = f"Added {new_poly_count} polynomial features"
else:
    poly_info = "Disabled"

X_train_interact = X_train_poly_df.copy()
X_test_interact = X_test_poly_df.copy()
interaction_count = 0

if USE_INTERACTIONS:
    for i, feat1 in enumerate(top_features_for_poly):
        for feat2 in top_features_for_poly[i+1:]:
            interaction_name = f"interact_{feat1}_x_{feat2}"
            X_train_interact[interaction_name] = X_train_selected[feat1] * X_train_selected[feat2]
            X_test_interact[interaction_name] = X_test_selected[feat1] * X_test_selected[feat2]
            interaction_count += 1
    interact_info = f"Added {interaction_count} interaction features"
else:
    interact_info = "Disabled"

X_train_transformed = X_train_interact
X_test_transformed = X_test_interact

print(f"Features: {X_train_selected.shape[1]} -> {X_train_transformed.shape[1]}")
print(f"  Polynomial: {poly_info}")
print(f"  Interactions: {interact_info}")
print()

# ============================================================================
# 4. SCALING (FIT ON ENTIRE TRAINING SET)
# ============================================================================

print("=" * 80)
print("SCALING (FIT ON ENTIRE TRAINING SET)")
print("=" * 80)

standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

# FIT on entire training set (not on CV splits)
X_train_standard = standard_scaler.fit_transform(X_train_transformed)
X_test_standard = standard_scaler.transform(X_test_transformed)

if SCALER_TYPE.lower() == "standard":
    X_train_scaled = pd.DataFrame(X_train_standard, columns=X_train_transformed.columns)
    X_test_scaled = pd.DataFrame(X_test_standard, columns=X_test_transformed.columns)
    scaler_name = "StandardScaler"
else:
    X_train_robust = robust_scaler.fit_transform(X_train_transformed)
    X_test_robust = robust_scaler.transform(X_test_transformed)
    X_train_scaled = pd.DataFrame(X_train_robust, columns=X_train_transformed.columns)
    X_test_scaled = pd.DataFrame(X_test_robust, columns=X_test_transformed.columns)
    scaler_name = "RobustScaler"

print(f"[OK] {scaler_name} fit on entire training set ({X_train_scaled.shape[0]} samples)")

power_transformer = None
y_train_for_model = y_train.copy()

if USE_POWER_TRANSFORMER:
    power_transformer = PowerTransformer(method='yeo-johnson')
    y_train_for_model = power_transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    print("[OK] PowerTransformer applied to target")
else:
    print("[OK] PowerTransformer disabled")

print()

# ============================================================================
# 5. K-FOLD CROSS-VALIDATION SETUP
# ============================================================================

print("=" * 80)
print("K-FOLD CROSS-VALIDATION SETUP")
print("=" * 80)

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"K-Fold configuration: {N_SPLITS} splits")
print(f"Training set size: {X_train_scaled.shape}")
print(f"Test set size: {X_test_scaled.shape}")
print()

# Custom R² scorer
r2_scorer = make_scorer(r2_score)
print()

# ============================================================================
# 6. BASELINE MODELS WITH K-FOLD CV
# ============================================================================

print("=" * 80)
print("BASELINE MODELS (K-FOLD CROSS-VALIDATION)")
print("=" * 80)

baseline_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1),
}

baseline_results = {}
for name, model in baseline_models.items():
    cv_scores = cross_validate(
        model, X_train_scaled, y_train_for_model,
        cv=kfold,
        scoring={'r2': r2_scorer, 'neg_mse': 'neg_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'},
        n_jobs=-1
    )
    r2_mean = cv_scores['test_r2'].mean()
    r2_std = cv_scores['test_r2'].std()
    rmse_mean = np.sqrt(-cv_scores['test_neg_mse'].mean())
    baseline_results[name] = {
        'model': model,
        'cv_r2': r2_mean,
        'cv_r2_std': r2_std,
        'cv_rmse': rmse_mean,
        'cv_scores': cv_scores
    }
    print(f"{name:25} | CV R²: {r2_mean:.6f} ± {r2_std:.4f} | RMSE: {rmse_mean:.4f}")

print(f"\n[OK] Baseline models evaluated with {N_SPLITS}-fold CV")
print()

# ============================================================================
# 7. ADVANCED MODELS WITH K-FOLD CV
# ============================================================================

print("=" * 80)
print("ADVANCED MODELS (K-FOLD CROSS-VALIDATION)")
print("=" * 80)

advanced_models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
}

advanced_results = {}
for name, model in advanced_models.items():
    cv_scores = cross_validate(
        model, X_train_scaled, y_train_for_model,
        cv=kfold,
        scoring={'r2': r2_scorer, 'neg_mse': 'neg_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'},
        n_jobs=-1
    )
    r2_mean = cv_scores['test_r2'].mean()
    r2_std = cv_scores['test_r2'].std()
    rmse_mean = np.sqrt(-cv_scores['test_neg_mse'].mean())
    advanced_results[name] = {
        'model': model,
        'cv_r2': r2_mean,
        'cv_r2_std': r2_std,
        'cv_rmse': rmse_mean,
        'cv_scores': cv_scores
    }
    print(f"{name:25} | CV R²: {r2_mean:.6f} ± {r2_std:.4f} | RMSE: {rmse_mean:.4f}")

print(f"\n[OK] Advanced models evaluated with {N_SPLITS}-fold CV")
print()

# ============================================================================
# 8. HYPERPARAMETER TUNING WITH K-FOLD CV
# ============================================================================

print("=" * 80)
print(f"HYPERPARAMETER TUNING ({N_SPLITS}-FOLD CROSS-VALIDATION)")
print("=" * 80)

# Ridge tuning
print("\nTuning Ridge...")
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge_search = GridSearchCV(Ridge(), ridge_params, cv=kfold, scoring=r2_scorer, n_jobs=-1)
ridge_search.fit(X_train_scaled, y_train_for_model)
ridge_tuned_r2 = ridge_search.best_score_
print(f"  Best alpha: {ridge_search.best_params_['alpha']}, CV R²: {ridge_tuned_r2:.6f}")

# Random Forest tuning
print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_params, cv=kfold, scoring=r2_scorer, n_iter=TUNING_N_ITER,
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_search.fit(X_train_scaled, y_train_for_model)
rf_tuned_r2 = rf_search.best_score_
print(f"  Best depth: {rf_search.best_params_['max_depth']}, CV R²: {rf_tuned_r2:.6f}")

# XGBoost tuning
print("\nTuning XGBoost...")
xgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    xgb_params, cv=kfold, scoring=r2_scorer, n_iter=TUNING_N_ITER,
    n_jobs=-1, random_state=RANDOM_STATE
)
xgb_search.fit(X_train_scaled, y_train_for_model)
xgb_tuned_r2 = xgb_search.best_score_
print(f"  Best depth: {xgb_search.best_params_['max_depth']}, CV R²: {xgb_tuned_r2:.6f}")

# LightGBM tuning
print("\nTuning LightGBM...")
lgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
lgb_search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
    lgb_params, cv=kfold, scoring=r2_scorer, n_iter=TUNING_N_ITER,
    n_jobs=-1, random_state=RANDOM_STATE
)
lgb_search.fit(X_train_scaled, y_train_for_model)
lgb_tuned_r2 = lgb_search.best_score_
print(f"  Best depth: {lgb_search.best_params_['max_depth']}, CV R²: {lgb_tuned_r2:.6f}")

tuned_models = {
    'Ridge (Tuned)': ridge_search.best_estimator_,
    'Random Forest (Tuned)': rf_search.best_estimator_,
    'XGBoost (Tuned)': xgb_search.best_estimator_,
    'LightGBM (Tuned)': lgb_search.best_estimator_,
}

tuned_results_cv = {
    'Ridge (Tuned)': ridge_tuned_r2,
    'Random Forest (Tuned)': rf_tuned_r2,
    'XGBoost (Tuned)': xgb_tuned_r2,
    'LightGBM (Tuned)': lgb_tuned_r2,
}

print(f"\n[OK] Hyperparameter tuning complete")
print()

# ============================================================================
# 9. MODEL EVALUATION AND COMPARISON
# ============================================================================

print("=" * 80)
print("MODEL EVALUATION (K-FOLD CROSS-VALIDATION RESULTS)")
print("=" * 80)

# Compile all baseline results
all_cv_results = {}
for name, result in baseline_results.items():
    all_cv_results[name] = {
        'cv_r2': result['cv_r2'],
        'cv_r2_std': result['cv_r2_std'],
        'cv_rmse': result['cv_rmse']
    }

# Compile all advanced results
for name, result in advanced_results.items():
    all_cv_results[name] = {
        'cv_r2': result['cv_r2'],
        'cv_r2_std': result['cv_r2_std'],
        'cv_rmse': result['cv_rmse']
    }

# Compile tuned results
for name, cv_r2 in tuned_results_cv.items():
    all_cv_results[name] = {
        'cv_r2': cv_r2,
        'cv_r2_std': 0.0,
        'cv_rmse': 0.0
    }

# Create comparison DataFrame
comparison_df = pd.DataFrame([
    {'Model': name, 'CV R2': result['cv_r2'], 'CV Std': result['cv_r2_std'], 'CV RMSE': result['cv_rmse']}
    for name, result in all_cv_results.items()
]).sort_values('CV R2', ascending=False)

print("\nAll Model Results (sorted by CV R2):")
print(comparison_df.to_string(index=False))

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_cv_r2 = comparison_df.iloc[0]['CV R2']
best_cv_rmse = comparison_df.iloc[0]['CV RMSE']

print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model_name}")
print(f"CV R2: {best_cv_r2:.6f}")
print("=" * 80)
print()
print("=" * 80)
print()

# ============================================================================
# 10. TRAINING BEST MODEL ON ENTIRE TRAINING SET
# ============================================================================

print("=" * 80)
print("TRAINING BEST MODEL ON ENTIRE TRAINING SET")
print("=" * 80)

# Train best model on entire training set for test predictions
if best_model_name in tuned_models:
    # For tuned models, get the best estimator from search
    if 'Ridge' in best_model_name:
        best_model_final = Ridge(alpha=ridge_search.best_params_['alpha'])
    elif 'Random Forest' in best_model_name:
        best_model_final = RandomForestRegressor(**rf_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
    elif 'XGBoost' in best_model_name:
        best_model_final = xgb.XGBRegressor(**xgb_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
    elif 'LightGBM' in best_model_name:
        best_model_final = lgb.LGBMRegressor(**lgb_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
else:
    # For baseline models, get from results
    best_model_final = baseline_results[best_model_name]['model'] if best_model_name in baseline_results else advanced_results[best_model_name]['model']

# Train on entire training set
best_model_final.fit(X_train_scaled, y_train_for_model)
print(f"[OK] {best_model_name} trained on entire training set ({X_train_scaled.shape[0]} samples)")

print("\n" + "=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

# Generate predictions
y_pred_test = best_model_final.predict(X_test_scaled)
if power_transformer is not None:
    y_pred_test = power_transformer.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

print(f"\nPredictions generated: {len(y_pred_test)}")
print(f"Prediction statistics:")
print(f"  Mean: {y_pred_test.mean():.6f}")
print(f"  Std: {y_pred_test.std():.6f}")
print(f"  Min: {y_pred_test.min():.6f}")
print(f"  Max: {y_pred_test.max():.6f}")

# Create submission file
submission_df = pd.DataFrame({'yhat': y_pred_test})
output_filename = 'CW1_submission_best_model.csv'
submission_df.to_csv(output_filename, index=False)

print(f"\n[OK] Submission file saved: {output_filename}")
print()

# Alternative predictions from tuned models
print("=" * 80)
print("ALTERNATIVE PREDICTIONS (All Tuned Models)")
print("=" * 80)

# Train all tuned models on entire training set with best parameters
# Ridge
ridge_final = Ridge(alpha=ridge_search.best_params_['alpha'])
ridge_final.fit(X_train_scaled, y_train_for_model)
y_pred_ridge = ridge_final.predict(X_test_scaled)
if power_transformer is not None:
    y_pred_ridge = power_transformer.inverse_transform(y_pred_ridge.reshape(-1, 1)).ravel()
pd.DataFrame({'yhat': y_pred_ridge}).to_csv('CW1_submission_Ridge_Tuned.csv', index=False)
print("[OK] CW1_submission_Ridge_Tuned.csv")

# Random Forest
rf_final = RandomForestRegressor(**rf_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
rf_final.fit(X_train_scaled, y_train_for_model)
y_pred_rf = rf_final.predict(X_test_scaled)
if power_transformer is not None:
    y_pred_rf = power_transformer.inverse_transform(y_pred_rf.reshape(-1, 1)).ravel()
pd.DataFrame({'yhat': y_pred_rf}).to_csv('CW1_submission_Random_Forest_Tuned.csv', index=False)
print("[OK] CW1_submission_Random_Forest_Tuned.csv")

# XGBoost
xgb_final = xgb.XGBRegressor(**xgb_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
xgb_final.fit(X_train_scaled, y_train_for_model)
y_pred_xgb = xgb_final.predict(X_test_scaled)
if power_transformer is not None:
    y_pred_xgb = power_transformer.inverse_transform(y_pred_xgb.reshape(-1, 1)).ravel()
pd.DataFrame({'yhat': y_pred_xgb}).to_csv('CW1_submission_XGBoost_Tuned.csv', index=False)
print("[OK] CW1_submission_XGBoost_Tuned.csv")

# LightGBM
lgb_final = lgb.LGBMRegressor(**lgb_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
lgb_final.fit(X_train_scaled, y_train_for_model)
y_pred_lgb = lgb_final.predict(X_test_scaled)
if power_transformer is not None:
    y_pred_lgb = power_transformer.inverse_transform(y_pred_lgb.reshape(-1, 1)).ravel()
pd.DataFrame({'yhat': y_pred_lgb}).to_csv('CW1_submission_LightGBM_Tuned.csv', index=False)
print("[OK] CW1_submission_LightGBM_Tuned.csv")

print()

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("K-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 80)
print(f"Best Model: {best_model_name}")
print(f"CV R2: {best_cv_r2:.6f}")
print(f"CV RMSE: {best_cv_rmse:.6f}")
print(f"")
print(f"Cross-Validation Configuration:")
print(f"  N-Splits: {N_SPLITS}")
print(f"  Training Samples: {X_train_scaled.shape[0]}")
print(f"  Test Samples: {X_test_scaled.shape[0]}")
print(f"  Features Used: {X_train_scaled.shape[1]}")
print(f"")
print(f"Submission Files Generated:")
print(f"  - CW1_submission_best_model.csv ({best_model_name})")
print(f"  - CW1_submission_Ridge_Tuned.csv")
print(f"  - CW1_submission_Random_Forest_Tuned.csv")
print(f"  - CW1_submission_XGBoost_Tuned.csv")
print(f"  - CW1_submission_LightGBM_Tuned.csv")
print("=" * 80)
print()
print("[OK] K-fold cross-validation and model evaluation complete!")
print("You can now submit your best predictions file.")
