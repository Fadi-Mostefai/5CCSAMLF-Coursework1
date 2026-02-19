import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import json

# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv') # This does not include true outcomes (obviously)

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# One-hot encode categorical variables
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

# Ensure test set has same columns as training set
for col in trn.columns:
    if col not in X_tst.columns:
        X_tst[col] = 0
X_tst = X_tst[trn.columns]

# Prepare data
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']

# Scale features
scaler = StandardScaler()
X_trn_scaled = scaler.fit_transform(X_trn)
X_tst_scaled = scaler.transform(X_tst)

# Load best hyperparameters
with open('best_model_params.json', 'r') as f:
    params_data = json.load(f)
    best_params = params_data['best_params']

# Train XGBoost model with tuned hyperparameters
model = xgb.XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    random_state=123
)
model.fit(X_trn_scaled, y_trn, verbose=False)

# Test set predictions
yhat_lm = model.predict(X_tst_scaled)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_KNUMBER.csv', index=False) # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes
tst = pd.read_csv('CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# How does the linear model do?
print(r2_fn(yhat_lm))




