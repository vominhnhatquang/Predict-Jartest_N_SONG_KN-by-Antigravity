"""
Model Training Script for DATA_FPT.csv
Water Quality Dataset - Regression Task
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys
import json

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("AI MODEL WEB INTEGRATION - MODEL TRAINING")
print("Dataset: DATA_FPT.csv (Water Quality)")
print("="*80)

# 1. Load Data
print("\n[1/10] Loading data...")
data_path = 'data/raw/DATA_FPT.csv'
df = pd.read_csv(data_path)

print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# 2. Data Exploration
print("\n[2/10] Exploring data...")
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum()
print(missing)
print(f"Total missing: {missing.sum()}")

print("\nBasic statistics:")
print(df.describe())

# 3. Prepare Features and Target
print("\n[3/10] Preparing features and target...")

# Assuming last column is target, rest are features
target_col = df.columns[-1]  # Last column as target
feature_cols = df.columns[:-1].tolist()  # Rest as features

print(f"\nTarget column: {target_col}")
print(f"Feature columns: {feature_cols}")

X = df[feature_cols]
y = df[target_col]

# Remove rows where target is missing
mask = y.notna()
X = X[mask]
y = y[mask]

print(f"\n[OK] After removing missing targets: {X.shape[0]} samples")

# 4. Train-Test Split
print("\n[4/10] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 5. Handle Missing Values
print("\n[5/10] Handling missing values...")
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print(f"[OK] Missing values imputed using mean strategy")
print(f"Training data shape: {X_train_imputed.shape}")

# 6. Feature Scaling
print("\n[6/10] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"[OK] Features scaled using StandardScaler")

# 7. Model Training and Comparison
print("\n[7/10] Training models...")
print("-"*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=RANDOM_STATE),
    'Lasso': Lasso(random_state=RANDOM_STATE),
    'ElasticNet': ElasticNet(random_state=RANDOM_STATE)
}

results = []
cv_results_data = []  # To store all CV scores for plotting

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store for plotting
        for score in cv_scores:
            cv_results_data.append({'Model': name, 'R2 Score': score})
            
    except Exception as e:
        print(f"  Warning: CV failed - {str(e)}")
        cv_mean = cv_std = 0
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate MAPE safely (avoid division by zero)
    # Adding a small epsilon to denominator or filtering out zero actuals
    mask = y_test != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan
        
    accuracy = 100 - mape
    
    results.append({
        'Model': name,
        'CV R2': cv_mean,
        'CV Std': cv_std,
        'Train R2': train_r2,
        'Test R2': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Test MAPE': mape,
        'Test Accuracy': accuracy
    })
    
    print(f"  CV R2: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"  Train R2: {train_r2:.4f}")
    print(f"  Test R2: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test MAPE: {mape:.2f}%")

print("\n" + "-"*80)

# 8. Select Best Model and Tune
print("\n[8/10] Selecting and tuning best model...")

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.round(4))

best_idx = results_df['Test R2'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\n[OK] Best model: {best_model_name}")

# Hyperparameter tuning for Ridge
print("\nTuning Ridge Regression...")
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    Ridge(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R2 score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# 9. Final Evaluation & Metric Saving
print("\n[9/10] Final evaluation and visualization...")

y_train_pred_final = best_model.predict(X_train_scaled)
y_test_pred_final = best_model.predict(X_test_scaled)

train_r2_final = r2_score(y_train, y_train_pred_final)
test_r2_final = r2_score(y_test, y_test_pred_final)
test_mae_final = mean_absolute_error(y_test, y_test_pred_final)
test_rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred_final))

# Final MAPE and Accuracy
mask = y_test != 0
if np.sum(mask) > 0:
    test_mape_final = np.mean(np.abs((y_test[mask] - y_test_pred_final[mask]) / y_test[mask])) * 100
else:
    test_mape_final = np.nan

test_accuracy_final = 100 - test_mape_final

print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE")
print("="*80)
print(f"\nModel: Ridge Regression (Optimized)")
print(f"Parameters: {grid_search.best_params_}")
print(f"\nTraining R2: {train_r2_final:.4f}")
print(f"Test R2: {test_r2_final:.4f}")
print(f"Test MAE: {test_mae_final:.4f}")
print(f"Test RMSE: {test_rmse_final:.4f}")
print(f"Test MAPE: {test_mape_final:.2f}%")
print(f"Test Accuracy: {test_accuracy_final:.2f}%")

diff = abs(train_r2_final - test_r2_final)
print(f"\nDifference (Train - Test R2): {diff:.4f}")

if diff > 0.1:
    print("[WARN] Possible overfitting detected")
else:
    print("[OK] Model generalizes well")

# Generate and Save Visualization
print("\n[VISUALIZATION] Generating Cross-Validation Comparison Plot...")
frontend_assets_dir = 'frontend/assets'
os.makedirs(frontend_assets_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
cv_df = pd.DataFrame(cv_results_data)
sns.boxplot(x='Model', y='R2 Score', data=cv_df, palette='viridis')
plt.title('Model Comparison - Cross Validation Scores (5 Folds)')
plt.ylabel('R2 Score')
plt.xlabel('Model')
plt.grid(True, linestyle='--', alpha=0.7)

plot_path = os.path.join(frontend_assets_dir, 'cv_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Plot saved to: {plot_path}")
plt.close()

# Prepare metrics data
metrics_data = {
    'best_model': best_model_name,
    'test_r2': round(test_r2_final, 4),
    'test_mae': round(test_mae_final, 4),
    'test_rmse': round(test_rmse_final, 4),
    'test_mape': round(test_mape_final, 2),
    'test_accuracy': round(test_accuracy_final, 2),
    'train_r2': round(train_r2_final, 4),
    'models_comparison': results_df.to_dict('records')
}

print("="*80)

# 10. Save Model, Preprocessing Objects and Metrics
print("\n[10/10] Saving model and preprocessing objects...")

model_dir = 'backend/model'
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_dir, 'trained_model.pkl')
joblib.dump(best_model, model_path)
print(f"[OK] Model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"[OK] Scaler saved: {scaler_path}")

# Save imputer
imputer_path = os.path.join(model_dir, 'imputer.pkl')
joblib.dump(imputer, imputer_path)
print(f"[OK] Imputer saved: {imputer_path}")

# Save feature names
feature_names = feature_cols
features_path = os.path.join(model_dir, 'feature_names.pkl')
joblib.dump(feature_names, features_path)
print(f"[OK] Feature names saved: {features_path}")

# Save validation report CSV
data_dir = 'data/processed'
os.makedirs(data_dir, exist_ok=True)
report_path = os.path.join(data_dir, 'model_validation_report.csv')
results_df.to_csv(report_path, index=False)
print(f"[OK] Validation report saved: {report_path}")

# Save Metrics JSON for API
metrics_json_path = os.path.join(model_dir, 'metrics.json')
with open(metrics_json_path, 'w') as f:
    json.dump(metrics_data, f, indent=4)
print(f"[OK] Metrics JSON saved: {metrics_json_path}")

# Print feature info
print("\n" + "="*80)
print("FEATURE INFORMATION")
print("="*80)
print(f"\nNumber of features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print(f"Target variable: {target_col}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Start the backend server (re-run start_backend.bat)")
print("2. Open frontend/index.html")
print("3. Check the 'Model Performance' section")
print("="*80)
