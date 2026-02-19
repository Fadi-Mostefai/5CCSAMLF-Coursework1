#!/usr/bin/env python3
"""
Model Training and Evaluation with K-Fold Cross-Validation
===========================================================

This script trains multiple machine learning models on the diamond dataset.

Pipeline:
  1. Load data and previously selected features (from EDA.py)
  2. Encode categorical variables
  3. Create polynomial and interaction features
  4. Standardize all features
  5. Use 5-fold cross-validation to evaluate baseline models
  6. Tune hyperparameters for best models
  7. Train final model on all training data
  8. Generate predictions on test set

K-Fold CV prevents overfitting and gives us multiple performance estimates.
Final model is chosen based on highest CV R² score.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

print("=" * 80)
print("DIAMOND PRICE PREDICTION - MODEL TRAINING")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

# Feature engineering
POLY_FEATURES = True
INTERACTION_FEATURES = True
POLY_DEGREE = 2
TOP_FEATURES_FOR_POLY = 5

# Scaling
SCALER_TYPE = "standard"

# Cross-validation
N_FOLDS = 5
TUNING_ITERATIONS = 5
RANDOM_STATE = 42

# ============================================================================
# Load data
# ============================================================================

train_data = pd.read_csv('CW1_train.csv')
test_data = pd.read_csv('CW1_test.csv')

y_train = train_data['outcome']
X_train_full = train_data.drop('outcome', axis=1)
X_test = test_data.copy()

categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()

# Load selected features from EDA
try:
    selected_features_df = pd.read_csv('selected_features.csv')
    selected_features = selected_features_df['feature'].tolist()
except FileNotFoundError:
    print("[WARNING] Run EDA.py first!")
    selected_features = None

# ============================================================================
# Prepare features
# ============================================================================

# Encode categorical variables
X_train_enc = pd.get_dummies(X_train_full, columns=categorical_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Make sure train and test have same columns
for col in X_train_enc.columns:
    if col not in X_test_enc.columns:
        X_test_enc[col] = 0
X_test_enc = X_test_enc[X_train_enc.columns]

# Create dimensions feature if we have x, y, z
if all(col in X_train_enc.columns for col in ['x', 'y', 'z']):
    X_train_enc['dimensions'] = X_train_enc[['x', 'y', 'z']].mean(axis=1)
    X_test_enc['dimensions'] = X_test_enc[['x', 'y', 'z']].mean(axis=1)

# Filter to selected features
if selected_features is not None:
    valid_features = [f for f in selected_features if f in X_train_enc.columns]
    X_train_selected = X_train_enc[valid_features]
    X_test_selected = X_test_enc[valid_features]
else:
    X_train_selected = X_train_enc
    X_test_selected = X_test_enc

# ============================================================================
# Feature engineering
# ============================================================================

X_train = X_train_selected.copy()
X_test_use = X_test_selected.copy()

# Get top features for polynomial expansion
top_corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
top_feats = top_corr.head(TOP_FEATURES_FOR_POLY).index.tolist()

# Add polynomial features
if POLY_FEATURES:
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False, interaction_only=False)
    poly_train = poly.fit_transform(X_train[top_feats])
    poly_names = poly.get_feature_names_out(top_feats)
    
    for i, name in enumerate(poly_names):
        if name not in X_train.columns:
            X_train[name] = poly_train[:, i]
    
    poly_test = poly.transform(X_test_use[top_feats])
    for i, name in enumerate(poly_names):
        if name not in X_test_use.columns:
            X_test_use[name] = poly_test[:, i]

# Add interaction terms
if INTERACTION_FEATURES:
    for i in range(len(top_feats)):
        for j in range(i+1, min(i+3, len(top_feats))):
            feat_name = f"{top_feats[i]}_x_{top_feats[j]}"
            X_train[feat_name] = X_train[top_feats[i]] * X_train[top_feats[j]]
            X_test_use[feat_name] = X_test_use[top_feats[i]] * X_test_use[top_feats[j]]

# ============================================================================
# Normalize features
# ============================================================================

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_use),
    columns=X_train.columns
)

# ============================================================================
# Setup K-Fold cross-validation
# ============================================================================

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
r2_scorer = make_scorer(r2_score)

# ============================================================================
# Train baseline models with cross-validation
# ============================================================================

baseline = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1),
}

baseline_scores = {}

for name, model in baseline.items():
    cv_results = cross_validate(
        model, X_train_scaled, y_train, 
        cv=kfold, 
        scoring={'r2': r2_scorer, 'rmse': 'neg_mean_squared_error'},
        return_train_score=False
    )
    
    r2_cv = cv_results['test_r2'].mean()
    rmse_cv = np.sqrt(-cv_results['test_rmse'].mean())
    
    baseline_scores[name] = {
        'r2': r2_cv,
        'rmse': rmse_cv,
        'model': model
    }

# ============================================================================
# Train advanced models with cross-validation
# ============================================================================

advanced = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
}

advanced_scores = {}

for name, model in advanced.items():
    cv_results = cross_validate(
        model, X_train_scaled, y_train,
        cv=kfold,
        scoring={'r2': r2_scorer, 'rmse': 'neg_mean_squared_error'},
        return_train_score=False
    )
    
    r2_cv = cv_results['test_r2'].mean()
    rmse_cv = np.sqrt(-cv_results['test_rmse'].mean())
    
    advanced_scores[name] = {
        'r2': r2_cv,
        'rmse': rmse_cv,
        'model': model
    }

# ============================================================================
# Hyperparameter tuning
# ============================================================================

tuned_scores = {}

# Ridge tuning
ridge_grid = GridSearchCV(
    Ridge(),
    {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    cv=kfold, scoring=r2_scorer, n_jobs=-1
)
ridge_grid.fit(X_train_scaled, y_train)
ridge_rmse_cv = np.sqrt(-cross_val_score(ridge_grid.best_estimator_, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')).mean()
tuned_scores['Ridge (Tuned)'] = {
    'r2': ridge_grid.best_score_,
    'rmse': ridge_rmse_cv,
    'params': ridge_grid.best_params_,
    'model': ridge_grid.best_estimator_
}

# Random Forest tuning
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
    },
    cv=kfold, scoring=r2_scorer, n_iter=TUNING_ITERATIONS, 
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_search.fit(X_train_scaled, y_train)
rf_rmse_cv = np.sqrt(-cross_val_score(rf_search.best_estimator_, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')).mean()
tuned_scores['Random Forest (Tuned)'] = {
    'r2': rf_search.best_score_,
    'rmse': rf_rmse_cv,
    'params': rf_search.best_params_,
    'model': rf_search.best_estimator_
}

# XGBoost tuning
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    },
    cv=kfold, scoring=r2_scorer, n_iter=TUNING_ITERATIONS,
    n_jobs=-1, random_state=RANDOM_STATE
)
xgb_search.fit(X_train_scaled, y_train)
xgb_rmse_cv = np.sqrt(-cross_val_score(xgb_search.best_estimator_, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')).mean()
tuned_scores['XGBoost (Tuned)'] = {
    'r2': xgb_search.best_score_,
    'rmse': xgb_rmse_cv,
    'params': xgb_search.best_params_,
    'model': xgb_search.best_estimator_
}

# LightGBM tuning
lgb_search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
    {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    },
    cv=kfold, scoring=r2_scorer, n_iter=TUNING_ITERATIONS,
    n_jobs=-1, random_state=RANDOM_STATE
)
lgb_search.fit(X_train_scaled, y_train)
lgb_rmse_cv = np.sqrt(-cross_val_score(lgb_search.best_estimator_, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')).mean()
tuned_scores['LightGBM (Tuned)'] = {
    'r2': lgb_search.best_score_,
    'rmse': lgb_rmse_cv,
    'params': lgb_search.best_params_,
    'model': lgb_search.best_estimator_
}

# ============================================================================
# Compare all models
# ============================================================================

all_scores = {}
all_scores.update(baseline_scores)
all_scores.update(advanced_scores)
all_scores.update(tuned_scores)

comparison = pd.DataFrame([
    {'Model': name, 'CV R2': scores['r2'], 'CV RMSE': scores['rmse']}
    for name, scores in all_scores.items()
]).sort_values('CV R2', ascending=False)

best_model_name = comparison.iloc[0]['Model']
best_r2 = comparison.iloc[0]['CV R2']

# ============================================================================
# Train final model and make predictions
# ============================================================================

best_model = all_scores[best_model_name]['model']
best_model.fit(X_train_scaled, y_train)

# Generate test predictions
y_pred_test = best_model.predict(X_test_scaled)

# Save main submission
submission = pd.DataFrame({'yhat': y_pred_test})
submission.to_csv('CW1_submission_best_model.csv', index=False)

# ============================================================================
# Alternative submissions from other tuned models
# ============================================================================

alternatives = {
    'Ridge (Tuned)': tuned_scores['Ridge (Tuned)']['model'],
    'Random Forest (Tuned)': tuned_scores['Random Forest (Tuned)']['model'],
    'XGBoost (Tuned)': tuned_scores['XGBoost (Tuned)']['model'],
    'LightGBM (Tuned)': tuned_scores['LightGBM (Tuned)']['model'],
}

for name, model in alternatives.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    filename = f"CW1_submission_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    pd.DataFrame({'yhat': preds}).to_csv(filename, index=False)

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"CV R2: {best_r2:.4f}")
print(f"\nPredictions saved to:")
print(f"  - CW1_submission_best_model.csv")
print(f"  - CW1_submission_Ridge_*.csv")
print(f"  - CW1_submission_Random_*.csv")
print(f"  - CW1_submission_XGBoost_*.csv")
print(f"  - CW1_submission_LightGBM_*.csv")
print("=" * 80)

# Save best model hyperparameters for reproducibility
import json
best_params = all_scores[best_model_name].get('params', {})
with open('best_model_params.json', 'w') as f:
    json.dump({
        'model_name': best_model_name,
        'best_params': best_params,
        'cv_r2': float(best_r2)
    }, f, indent=2)

