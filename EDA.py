"""
Exploratory Data Analysis & Feature Selection
Processes training data, performs EDA, and applies feature selection using multiple methods.
All plots and images are saved to the EDA_images folder.
"""

import os
import pandas as pd
import numpy as np

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression, VarianceThreshold
from sklearn.linear_model import Ridge
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Create EDA_images folder if it doesn't exist
os.makedirs('EDA_images', exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS & FEATURE SELECTION")
print("=" * 80)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

# Feature selection settings
FEATURE_SELECTION_OPTION = "B"  # A=very conservative, B=balanced, C=moderate, D=keep all
VARIANCE_THRESHOLD = 0.01
RANDOM_STATE = 42
MANUAL_FEATURES_TO_KEEP = ['carat', 'price', 'x', 'y', 'z']
FEATURES_TO_MERGE = {'dimensions': ['x', 'y', 'z']}
FEATURES_TO_DROP = []  
CORRELATION_THRESHOLD = 0.95
TARGET_CORRELATION_THRESHOLD = 0.1
USE_TARGET_CORRELATION_FILTERING = True

print(f"\n[OK] Configuration loaded")
print(f"  Feature Selection Option: {FEATURE_SELECTION_OPTION}")
print(f"  Manual Features to Keep: {MANUAL_FEATURES_TO_KEEP}")
print(f"  Feature Merging: {len(FEATURES_TO_MERGE)} groups")
print(f"  Correlation Threshold: {CORRELATION_THRESHOLD}")

# ============================================================================
# SECTION 2: LOAD DATA
# ============================================================================

train_data = pd.read_csv('CW1_train.csv')

print("\n" + "=" * 60)
print("TRAINING DATA INFORMATION")
print("=" * 60)
print(f"Shape: {train_data.shape}")
print(f"\nFirst few rows:")
print(train_data.head())
print(f"\nData types:")
print(train_data.dtypes)
print(f"\nMissing values:")
print(train_data.isnull().sum())
print(f"\nBasic statistics:")
print(train_data.describe())

y_train = train_data['outcome']
X_train_full = train_data.drop('outcome', axis=1)

print(f"\nTarget variable (outcome) statistics:")
print(f"Mean: {y_train.mean():.4f}")
print(f"Std: {y_train.std():.4f}")
print(f"Min: {y_train.min():.4f}")
print(f"Max: {y_train.max():.4f}")

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# 3.1 Correlation Heatmap
print("\n[1/4] Generating Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(15, 12))
correlation_matrix = train_data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
plt.title('Correlation Heatmap - All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('EDA_images/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("      [SAVED] EDA_images/01_correlation_heatmap.png")

target_corr = train_data.corr(numeric_only=True)['outcome'].abs().sort_values(ascending=False)
print("\nFeatures most correlated with target (outcome):")
print(target_corr.head(10))

# 3.2 Target Variable Distribution
print("\n[2/4] Generating Target Variable Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_train, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Target Variable (outcome)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('outcome')
axes[0].set_ylabel('Frequency')
axes[1].boxplot(y_train)
axes[1].set_title('Boxplot of Target Variable', fontsize=12, fontweight='bold')
axes[1].set_ylabel('outcome')
plt.tight_layout()
plt.savefig('EDA_images/02_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("      [SAVED] EDA_images/02_target_distribution.png")

# 3.3 Feature Distributions
print("\n[3/4] Generating Feature Distributions...")
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = [col for col in train_data.columns if col not in categorical_cols + ['outcome']]

print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

if len(numeric_cols) >= 4:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols[:4]):
        axes[idx].hist(train_data[col], bins=40, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('EDA_images/03_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      [SAVED] EDA_images/03_feature_distributions.png")

# 3.4 Outlier Detection
print("\n[4/4] Analyzing Outliers...")

def detect_outliers_iqr(data, multiplier=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def detect_outliers_mad(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros(len(data))
    return np.abs(modified_z_scores) > threshold

iqr_outliers = detect_outliers_iqr(y_train)
zscore_outliers = detect_outliers_zscore(y_train)
mad_outliers = detect_outliers_mad(y_train)
combined_outliers = iqr_outliers | zscore_outliers | mad_outliers

print("      OUTLIER DETECTION IN TARGET VARIABLE")
print(f"      IQR method: {iqr_outliers.sum()} outliers")
print(f"      Z-Score method: {zscore_outliers.sum()} outliers")
print(f"      MAD method: {mad_outliers.sum()} outliers")
print(f"      Combined: {combined_outliers.sum()} outliers")
print(f"      Percentage: {(combined_outliers.sum() / len(y_train)) * 100:.2f}%")

# ============================================================================
# SECTION 4: FEATURE PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE PREPARATION")
print("=" * 80)

# 4.1 Handle Missing Values
print("\n[1/3] Checking for missing values...")
nan_counts = X_train_full.isnull().sum()
if nan_counts.sum() > 0:
    print(f"[WARNING] Missing values found: {nan_counts[nan_counts > 0].to_dict()}")
    for col in X_train_full.columns:
        if X_train_full[col].isnull().any():
            if X_train_full[col].dtype in ['float64', 'int64']:
                fill_value = X_train_full[col].median()
                X_train_full[col] = X_train_full[col].fillna(fill_value)
            else:
                fill_value = X_train_full[col].mode()[0]
                X_train_full[col] = X_train_full[col].fillna(fill_value)
    print(f"[OK] Missing values filled")
else:
    print(f"[OK] No missing values found")

# 4.2 Encode Categorical Variables
print("\n[2/3] Encoding categorical variables...")
X_train_encoded = pd.get_dummies(X_train_full, columns=categorical_cols, drop_first=True)
print(f"[OK] Categorical encoding complete")
print(f"  Original features: {X_train_full.shape[1]}")
print(f"  After encoding: {X_train_encoded.shape[1]} features")

# 4.3 Remove Low Variance Features
print("\n[3/3] Removing low variance features...")
pre_variance_features = X_train_encoded.shape[1]
selector_variance = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
X_train_encoded = pd.DataFrame(
    selector_variance.fit_transform(X_train_encoded),
    columns=X_train_encoded.columns[selector_variance.get_support()]
)
print(f"[OK] Low variance filtering complete")
print(f"  Before: {pre_variance_features} features")
print(f"  After: {X_train_encoded.shape[1]} features")

# ============================================================================
# SECTION 5: TARGET CORRELATION FILTERING
# ============================================================================

print("\n" + "=" * 80)
print("FILTERING FEATURES BY TARGET CORRELATION")
print("=" * 80)

X_train_processed = X_train_encoded.copy()
target_correlations = X_train_processed.corrwith(y_train).abs().sort_values(ascending=False)

print(f"\nCorrelation of all features with target (outcome):")
print(target_correlations)

# Visualize target correlations
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(target_correlations)), target_correlations.values)
ax.set_yticks(range(len(target_correlations)))
ax.set_yticklabels(target_correlations.index)
ax.set_xlabel('|Correlation with outcome|')
ax.set_title('Feature Correlation with Target (outcome)', fontweight='bold')
ax.axvline(x=TARGET_CORRELATION_THRESHOLD, color='red', linestyle='--', linewidth=2, 
           label=f'Threshold ({TARGET_CORRELATION_THRESHOLD})')
ax.legend()
plt.tight_layout()
plt.savefig('EDA_images/04_target_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] EDA_images/04_target_correlation.png")

if USE_TARGET_CORRELATION_FILTERING and TARGET_CORRELATION_THRESHOLD > 0:
    # Get features that meet correlation threshold
    features_by_correlation = target_correlations[target_correlations > TARGET_CORRELATION_THRESHOLD].index.tolist()
    
    # Add manual features to ensure they're always included
    manual_features_to_add = [f for f in MANUAL_FEATURES_TO_KEEP if f in X_train_processed.columns]
    features_to_keep = list(set(features_by_correlation + manual_features_to_add))
    
    X_train_processed = X_train_processed[features_to_keep]
    print(f"[OK] Target correlation filtering applied")
    print(f"  Threshold: {TARGET_CORRELATION_THRESHOLD}")
    print(f"  Features by correlation: {len(features_by_correlation)}")
    print(f"  Manual features added: {manual_features_to_add}")
    print(f"  Total features kept: {len(features_to_keep)}/{X_train_encoded.shape[1]}")
else:
    print(f"[DISABLED] Target correlation filtering disabled")

# ============================================================================
# SECTION 6: IDENTIFY HIGHLY CORRELATED FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("IDENTIFYING HIGHLY CORRELATED FEATURE GROUPS")
print("=" * 80)

corr_matrix = X_train_processed.corr().abs()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > CORRELATION_THRESHOLD:
            high_corr_pairs.append({
                'Feature1': corr_matrix.columns[i],
                'Feature2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    corr_pairs_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
    print(f"\nFound {len(high_corr_pairs)} feature pairs with correlation > {CORRELATION_THRESHOLD}:")
    print(corr_pairs_df.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_pairs_df_display = corr_pairs_df.head(10)
    pair_names = [f"{row['Feature1'][:15]} -- {row['Feature2'][:15]}" 
                  for _, row in corr_pairs_df_display.iterrows()]
    ax.barh(range(len(corr_pairs_df_display)), corr_pairs_df_display['Correlation'].values)
    ax.set_yticks(range(len(corr_pairs_df_display)))
    ax.set_yticklabels(pair_names, fontsize=9)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f'Top Highly Correlated Feature Pairs (>{CORRELATION_THRESHOLD})', fontweight='bold')
    ax.set_xlim([CORRELATION_THRESHOLD, 1.0])
    plt.tight_layout()
    plt.savefig('EDA_images/05_highly_correlated_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[SAVED] EDA_images/05_highly_correlated_pairs.png")
else:
    print(f"\nNo feature pairs found with correlation > {CORRELATION_THRESHOLD}")

# ============================================================================
# SECTION 7: FEATURE MERGING AND DROPPING
# ============================================================================

print("\n" + "=" * 80)
print("APPLYING FEATURE MERGING AND DROPPING")
print("=" * 80)

features_state = []
X_train_temp = X_train_processed.copy()

if FEATURES_TO_MERGE:
    for merged_name, feature_list in FEATURES_TO_MERGE.items():
        existing_features = [f for f in feature_list if f in X_train_temp.columns]
        if len(existing_features) > 1:
            X_train_temp[merged_name] = X_train_temp[existing_features].mean(axis=1)
            features_state.append(f"[MERGED] {existing_features} -> {merged_name}")
            X_train_temp = X_train_temp.drop(columns=existing_features)
            print(f"[OK] Created '{merged_name}' feature from {existing_features}")

if FEATURES_TO_DROP:
    existing_to_drop = [f for f in FEATURES_TO_DROP if f in X_train_temp.columns]
    if existing_to_drop:
        X_train_temp = X_train_temp.drop(columns=existing_to_drop)
        features_state.append(f"[DROPPED] {existing_to_drop}")
        print(f"[OK] Dropped {existing_to_drop}")

X_train_processed = X_train_temp.copy()
print(f"\nFeature count: {X_train_encoded.shape[1]} → {X_train_processed.shape[1]}")
print(f"Final features for selection ({X_train_processed.shape[1]} total)")

# ============================================================================
# SECTION 8: FEATURE SELECTION - MULTIPLE METHODS
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION USING MULTIPLE METHODS")
print("=" * 80)

# Method 1: F-Score
print("\n[1/4] F-Score Selection...")
selector_kbest = SelectKBest(f_regression, k=min(20, X_train_processed.shape[1]))
selector_kbest.fit(X_train_processed, y_train)
scores = selector_kbest.scores_

feature_importance_df = pd.DataFrame({
    'Feature': X_train_processed.columns,
    'F-Score': scores
}).sort_values('F-Score', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance_df.head(15)
ax.barh(range(len(top_features)), top_features['F-Score'].values)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.set_xlabel('F-Score')
ax.set_title('Top 15 Features by F-Score', fontweight='bold')
plt.tight_layout()
plt.savefig('EDA_images/06_fscore_selection.png', dpi=300, bbox_inches='tight')
plt.close()
print("      [OK] F-Score complete")
print(f"      Top 5: {feature_importance_df.head(5)['Feature'].tolist()}")
print("      [SAVED] EDA_images/06_fscore_selection.png")

# Method 2: Mutual Information
print("\n[2/4] Mutual Information Selection...")
mi_scores = mutual_info_regression(X_train_processed, y_train, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({
    'Feature': X_train_processed.columns,
    'MI-Score': mi_scores
}).sort_values('MI-Score', ascending=False)
print(f"      [OK] Mutual Information complete")
print(f"      Top 5: {mi_df.head(5)['Feature'].tolist()}")

# Method 3: Correlation
print("\n[3/4] Correlation Selection...")
correlation_scores = X_train_processed.corrwith(y_train).abs().sort_values(ascending=False)
print(f"      [OK] Correlation complete")
print(f"      Top 5: {correlation_scores.head(5).index.tolist()}")

# Method 4: RFE
print("\n[4/4] RFE Selection...")
rfe = RFE(Ridge(alpha=1.0), n_features_to_select=min(15, X_train_processed.shape[1]), step=1)
rfe.fit(X_train_processed, y_train)
rfe_features = X_train_processed.columns[rfe.support_].tolist()
print(f"      [OK] RFE complete - selected {len(rfe_features)} features")

# ============================================================================
# SECTION 9: CONSENSUS VOTING
# ============================================================================

print("\n" + "=" * 80)
print("CONSENSUS VOTING: COMBINE ALL METHODS")
print("=" * 80)

top_k = 15
top_fscore = set(feature_importance_df.head(top_k)['Feature'].tolist())
top_mi = set(mi_df.head(top_k)['Feature'].tolist())
top_corr = set(correlation_scores.head(top_k).index.tolist())
top_rfe = set(rfe_features)

consensus_features = {}
for feat in X_train_processed.columns:
    methods_count = sum([feat in top_fscore, feat in top_mi, feat in top_corr, feat in top_rfe])
    consensus_features[feat] = methods_count

consensus_df = pd.DataFrame(list(consensus_features.items()), 
                             columns=['Feature', 'Consensus_Count']).sort_values('Consensus_Count', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top_consensus = consensus_df[consensus_df['Consensus_Count'] > 0].head(20)
colors = ['#2ecc71' if x == 4 else '#f39c12' if x >= 2 else '#e74c3c' for x in top_consensus['Consensus_Count']]
ax.barh(range(len(top_consensus)), top_consensus['Consensus_Count'].values, color=colors)
ax.set_yticks(range(len(top_consensus)))
ax.set_yticklabels(top_consensus['Feature'].values)
ax.set_xlabel('Consensus Count (0-4 methods)')
ax.set_title('Feature Selection Consensus', fontweight='bold')
ax.set_xlim(0, 4.5)
plt.tight_layout()
plt.savefig('EDA_images/07_consensus_voting.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] EDA_images/07_consensus_voting.png")

print("\nCONSENSUS VOTING RESULTS")
print(f"Features selected by all 4 methods: {(consensus_df['Consensus_Count'] == 4).sum()}")
print(f"Features selected by 3+ methods: {(consensus_df['Consensus_Count'] >= 3).sum()}")
print(f"Features selected by 2+ methods: {(consensus_df['Consensus_Count'] >= 2).sum()}")
print(f"\nTop consensus features (4 votes):")
print(consensus_df[consensus_df['Consensus_Count'] == 4]['Feature'].tolist())

# ============================================================================
# SECTION 10: APPLY FEATURE SELECTION STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("APPLYING FEATURE SELECTION STRATEGY")
print("=" * 80)

consensus_4_features = consensus_df[consensus_df['Consensus_Count'] == 4]['Feature'].tolist()
consensus_2_features = consensus_df[consensus_df['Consensus_Count'] >= 2]['Feature'].tolist()

if FEATURE_SELECTION_OPTION.upper() == "A":
    selected_features = consensus_4_features
    strategy_name = "Most Conservative (All 4 methods agreed)"
elif FEATURE_SELECTION_OPTION.upper() == "B":
    selected_features = consensus_2_features
    strategy_name = "Balanced (2+ methods agreed)"
elif FEATURE_SELECTION_OPTION.upper() == "C":
    rank_df = pd.DataFrame(index=X_train_processed.columns)
    rank_df['Avg_Rank'] = rank_df.index.map(lambda f: sum([
        feature_importance_df[feature_importance_df['Feature'] == f].index[0] if f in feature_importance_df['Feature'].values else len(feature_importance_df),
        mi_df[mi_df['Feature'] == f].index[0] if f in mi_df['Feature'].values else len(mi_df),
    ]) / 2)
    rank_df = rank_df.sort_values('Avg_Rank')
    selected_features = rank_df.index.tolist()[:15]
    strategy_name = "Moderate (Top 15 by average rank)"
else:
    selected_features = X_train_processed.columns.tolist()
    strategy_name = "Aggressive (Keep all features)"

# Add manually specified features
manual_features_in_data = [f for f in MANUAL_FEATURES_TO_KEEP if f in X_train_processed.columns]
selected_features = list(set(selected_features + manual_features_in_data))

print(f"\nStrategy: Option {FEATURE_SELECTION_OPTION} - {strategy_name}")
print(f"Original features (after encoding): {X_train_encoded.shape[1]}")
print(f"After filtering/merging/dropping: {X_train_processed.shape[1]}")
print(f"\nManual Features Setup: {MANUAL_FEATURES_TO_KEEP}")
print(f"Manual Features Found: {manual_features_in_data}")
print(f"Manual Features Not Found: {[f for f in MANUAL_FEATURES_TO_KEEP if f not in manual_features_in_data]}")
print(f"\nTotal selected features: {len(selected_features)}")
print(f"Dropped features: {X_train_processed.shape[1] - len(selected_features)}")
print(f"\nSelected feature list ({len(selected_features)} features):")
for i, feat in enumerate(selected_features, 1):
    if feat in consensus_df['Feature'].values:
        consensus_count = consensus_df[consensus_df['Feature'] == feat]['Consensus_Count'].values[0]
        manual_label = " [MANUAL]" if feat in manual_features_in_data else ""
        print(f"  {i}. {feat} (consensus: {consensus_count}/4) {manual_label}")
    else:
        print(f"  {i}. {feat} (manual override)")

# ============================================================================
# SECTION 11: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

with open('selected_features.txt', 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

pd.DataFrame({'feature': selected_features}).to_csv('selected_features.csv', index=False)

print("\n[OK] Selected features saved to:")
print("  - selected_features.txt")
print("  - selected_features.csv")
print(f"\n[OK] All {len(selected_features)} features ready for modeling!")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n[OK] Generated plots saved to: EDA_images/")
print(f"  - 01_correlation_heatmap.png")
print(f"  - 02_target_distribution.png")
print(f"  - 03_feature_distributions.png")
print(f"  - 04_target_correlation.png")
print(f"  - 05_highly_correlated_pairs.png")
print(f"  - 06_fscore_selection.png")
print(f"  - 07_consensus_voting.png")
