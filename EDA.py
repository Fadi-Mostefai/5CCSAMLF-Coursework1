"""
Exploratory Data Analysis for Diamond Price Prediction
========================================================

This script explores the diamond dataset step-by-step:
- Load and understand the raw data
- Create visualizations of key patterns
- Identify features that matter for outcome prediction
- Remove noise and redundancy
- Produce a clean feature set for modeling

The output is saved to selected_features.csv so the model training 
script can use it. Simple as that.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use file-based backend (no display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression, VarianceThreshold
from sklearn.linear_model import Ridge
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Setup
os.makedirs('EDA_images', exist_ok=True)

print("=" * 80)
print("DIAMOND DATASET ANALYSIS")
print("=" * 80)

# ============================================================================
# Configuration - Tune these to change feature selection behavior
# ============================================================================

FEATURE_SELECTION_OPTION = "B"  # A=keep few, B=balanced, C=keep more, D=keep all
VARIANCE_THRESHOLD = 0.01  # Minimum variation to keep a feature
CORRELATION_THRESHOLD = 0.95  # Don't keep too-similar features
TARGET_CORRELATION_THRESHOLD = 0.1  # Minimum correlation with price

# Features we specifically want (domain knowledge)
MANUAL_FEATURES_TO_KEEP = ['carat', 'price', 'x', 'y', 'z']
FEATURES_TO_MERGE = {'dimensions': ['x', 'y', 'z']}  # Combine similar features

RANDOM_STATE = 42

# ============================================================================
# Load data and explore
# ============================================================================

train_data = pd.read_csv('CW1_train.csv')
y_train = train_data['outcome']
X_train_full = train_data.drop('outcome', axis=1)

# ============================================================================
# Create visualizations
# ============================================================================

# 1. Correlation heatmap
fig, ax = plt.subplots(figsize=(15, 12))
corr_matrix = train_data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
plt.title('Feature Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('EDA_images/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Show top features
target_corr = train_data.corr(numeric_only=True)['outcome'].abs().sort_values(ascending=False)

# 2. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_train, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Outcome Distribution', fontweight='bold')
axes[0].set_xlabel('Outcome')
axes[0].set_ylabel('Count')
axes[1].boxplot(y_train)
axes[1].set_title('Outliers Check', fontweight='bold')
axes[1].set_ylabel('Price')
plt.tight_layout()
plt.savefig('EDA_images/02_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature distributions (first 4 numeric features)
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = [col for col in train_data.columns if col not in categorical_cols + ['outcome']]

if len(numeric_cols) >= 4:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, col in enumerate(numeric_cols[:4]):
        ax = axes[idx // 2, idx % 2]
        ax.hist(train_data[col], bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(col, fontweight='bold')
    plt.tight_layout()
    plt.savefig('EDA_images/03_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Feature preparation - Clean the data
# ============================================================================

# Encode categorical variables
X_train_encoded = pd.get_dummies(X_train_full, columns=categorical_cols, drop_first=True)

# Remove features with almost no variance
selector_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
X_train_encoded = pd.DataFrame(
    selector_var.fit_transform(X_train_encoded),
    columns=X_train_encoded.columns[selector_var.get_support()]
)

# ============================================================================
# Filter by target correlation
# ============================================================================

X_train_proc = X_train_encoded.copy()
target_corr = X_train_proc.corrwith(y_train).abs().sort_values(ascending=False)

# Plot it
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(target_corr)), target_corr.values, color='steelblue')
ax.set_yticks(range(len(target_corr)))
ax.set_yticklabels(target_corr.index, fontsize=9)
ax.set_xlabel('|Correlation with Outcome|')
ax.set_title('Which Features Actually Matter?', fontweight='bold')
ax.axvline(x=TARGET_CORRELATION_THRESHOLD, color='red', linestyle='--', linewidth=2, 
           label=f'Threshold ({TARGET_CORRELATION_THRESHOLD})')
ax.legend()
plt.tight_layout()
plt.savefig('EDA_images/04_target_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# Keep features above threshold + our manual features
features_above_threshold = target_corr[target_corr > TARGET_CORRELATION_THRESHOLD].index.tolist()
manual_in_data = [f for f in MANUAL_FEATURES_TO_KEEP if f in X_train_proc.columns]
features_to_keep = list(set(features_above_threshold + manual_in_data))

X_train_proc = X_train_proc[features_to_keep]

# ============================================================================
# Check for multicollinearity
# ============================================================================

corr_pairs = []
feature_corr = X_train_proc.corr().abs()
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        if feature_corr.iloc[i, j] > CORRELATION_THRESHOLD:
            corr_pairs.append({
                'F1': feature_corr.columns[i],
                'F2': feature_corr.columns[j],
                'Corr': feature_corr.iloc[i, j]
            })

if corr_pairs:
    pairs_df = pd.DataFrame(corr_pairs).sort_values('Corr', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_pairs = pairs_df.head(10)
    labels = [f"{row['F1']}\nvs\n{row['F2']}" for _, row in top_pairs.iterrows()]
    ax.barh(range(len(top_pairs)), top_pairs['Corr'].values, color='coral')
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Correlation')
    ax.set_title('Feature Multicollinearity', fontweight='bold')
    plt.tight_layout()
    plt.savefig('EDA_images/05_highly_correlated_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Feature merging
# ============================================================================

X_train_proc_temp = X_train_proc.copy()

for new_name, to_combine in FEATURES_TO_MERGE.items():
    if all(f in X_train_proc_temp.columns for f in to_combine):
        X_train_proc_temp[new_name] = X_train_proc_temp[to_combine].mean(axis=1)
        X_train_proc_temp = X_train_proc_temp.drop(to_combine, axis=1)

X_train_proc = X_train_proc_temp

# ============================================================================
# Try multiple feature selection methods
# ============================================================================

# Method 1: F-Score (statistical test)
selector_f = SelectKBest(f_regression, k=min(20, X_train_proc.shape[1]))
selector_f.fit(X_train_proc, y_train)
f_scores = pd.DataFrame({
    'Feature': X_train_proc.columns,
    'Score': selector_f.scores_
}).sort_values('Score', ascending=False)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
top_f = f_scores.head(15)
ax.barh(range(len(top_f)), top_f['Score'].values, color='skyblue')
ax.set_yticks(range(len(top_f)))
ax.set_yticklabels(top_f['Feature'].values, fontsize=10)
ax.set_xlabel('F-Score')
ax.set_title('Statistical Feature Importance (F-Score)', fontweight='bold')
plt.tight_layout()
plt.savefig('EDA_images/06_fscore_selection.png', dpi=300, bbox_inches='tight')
plt.close()

# Method 2: Mutual Information
mi_scores = mutual_info_regression(X_train_proc, y_train, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({'Feature': X_train_proc.columns, 'MI': mi_scores}).sort_values('MI', ascending=False)

# Method 3: Correlation
corr_with_target = X_train_proc.corrwith(y_train).abs().sort_values(ascending=False)

# Method 4: RFE (Model-based)
rfe = RFE(Ridge(alpha=1.0), n_features_to_select=min(15, X_train_proc.shape[1]))
rfe.fit(X_train_proc, y_train)
rfe_features = X_train_proc.columns[rfe.support_].tolist()

# ============================================================================
# Consensus voting - combine all methods
# ============================================================================

top_k = 15
top_f_set = set(f_scores.head(top_k)['Feature'].tolist())
top_mi_set = set(mi_df.head(top_k)['Feature'].tolist())
top_corr_set = set(corr_with_target.head(top_k).index.tolist())
top_rfe_set = set(rfe_features)

# Count votes
votes = {}
for feat in X_train_proc.columns:
    votes[feat] = sum([
        feat in top_f_set,
        feat in top_mi_set,
        feat in top_corr_set,
        feat in top_rfe_set
    ])

votes_df = pd.DataFrame(list(votes.items()), columns=['Feature', 'Votes']).sort_values('Votes', ascending=False)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
top_votes = votes_df[votes_df['Votes'] > 0].head(20)
colors = ['#2ecc71' if x == 4 else '#f39c12' if x >= 2 else '#e74c3c' for x in top_votes['Votes']]
ax.barh(range(len(top_votes)), top_votes['Votes'].values, color=colors)
ax.set_yticks(range(len(top_votes)))
ax.set_yticklabels(top_votes['Feature'].values, fontsize=10)
ax.set_xlabel('Votes (out of 4 methods)')
ax.set_title('Feature Selection Consensus', fontweight='bold')
ax.set_xlim(0, 4.5)
plt.tight_layout()
plt.savefig('EDA_images/07_consensus_voting.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Apply feature selection strategy
# ============================================================================

# Choose based on option
if FEATURE_SELECTION_OPTION.upper() == 'A':
    selected = set(votes_df[votes_df['Votes'] == 4]['Feature'].tolist())
elif FEATURE_SELECTION_OPTION.upper() == 'B':
    selected = set(votes_df[votes_df['Votes'] >= 2]['Feature'].tolist())
elif FEATURE_SELECTION_OPTION.upper() == 'C':
    selected = set(f_scores.head(20)['Feature'].tolist())
else:
    selected = set(X_train_proc.columns)

# Always include manual features
manual_found = [f for f in MANUAL_FEATURES_TO_KEEP if f in X_train_proc.columns]
selected = list(set(list(selected) + manual_found))
selected.sort()

# ============================================================================
# Save results
# ============================================================================

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

with open('selected_features.txt', 'w') as f:
    f.write('\n'.join(selected))

pd.DataFrame({'feature': selected}).to_csv('selected_features.csv', index=False)

print(f"\nSelected {len(selected)} features")
print("Saved: selected_features.csv")
print("Saved: EDA_images/")
print("=" * 80)
