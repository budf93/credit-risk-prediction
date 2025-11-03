df_classif = pd.read_csv('loan_data_2007_2014.csv')
df_classif.info()

df_classif = df_classif.drop_duplicates()
df_classif.info()

import pandas as pd

# 1. Define the percentage threshold for KEEPING non-null data
# To KEEP columns with *at least* 20% non-nulls (i.e., drop those with > 80% nulls)
null_threshold = 0.80 
non_null_percent_to_keep = 1 - null_threshold  # This equals 0.20 (20%)

# 2. Calculate the required minimum number of non-null values
min_non_nulls = int(df_classif.shape[0] * non_null_percent_to_keep)

# 3. Drop columns that do NOT meet this threshold
df_classif = df_classif.dropna(
    axis=1,             # Operate on columns
    thresh=min_non_nulls # Only keep columns with at least this many non-null values
)

print(f"Original columns: {df_classif.shape[1]}")
print(f"Columns dropped _cladf_classifif < {non_null_percent_to_keep*100:.0f}% non-null.")

import numpy as np
import pandas as pd

# Load or create your DataFrame here (e.g., df_classif = pd.read_csv('data.csv'))

# Numerical Columns
numerical_cols = df_classif.select_dtypes(include=np.number).columns.tolist()

# Categorical Columns (including generic string/text 'object' types)
categorical_cols = df_classif.select_dtypes(include=['object', 'category']).columns.tolist()

print("--- Data Type Separation ---")
print(f"Total Columns: {len(df_classif.columns)}")
print(f"Numerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")

# List of high-cardinality columns to drop (excluding date and zip_code)
columns_to_drop = [
    'url',         # 466,285 unique values (Link/ID)
    'emp_title',   # 205,475 unique values (Too many unique job titles)
    'desc',        # 124,435 unique values (Free text/Noise)
    'title',        # 63,098 unique values (Free text/Redundant with 'purpose')
    'application_type',
    'pymnt_plan',
]

# Drop the columns from the DataFrame (use axis=1 for columns)
df_classif = df_classif.drop(columns=columns_to_drop, axis=1)

print(f"Dropped {len(columns_to_drop)} columns.")
print(f"New DataFrame shape: {df_classif.shape}")

import numpy as np
import pandas as pd

# Load or create your DataFrame here (e.g., df_classif = pd.read_csv('data.csv'))

# Numerical Columns
numerical_cols = df_classif.select_dtypes(include=np.number).columns.tolist()

# Categorical Columns (including generic string/text 'object' types)
categorical_cols = df_classif.select_dtypes(include=['object', 'category']).columns.tolist()

print("--- Data Type Separation ---")
print(f"Total Columns: {len(df_classif.columns)}")
print(f"Numerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")

# Calculate the median for all numerical columns in one go
numerical_medians = df_classif[numerical_cols].median()

# Apply the median imputation
df_classif[numerical_cols] = df_classif[numerical_cols].fillna(numerical_medians)

# Verification (optional): Check if nulls remain in numerical columns
# print("\nNull count after numerical imputation:")
# print(df_classif[numerical_cols].isnull().sum().nlargest(5))

# Calculate the mode for all categorical columns.
# Since .mode() returns a Series (as there can be multiple modes),
# we take the first element [0] for a single imputation value.
categorical_modes = {col: df_classif[col].mode()[0] for col in categorical_cols}

# Apply the mode imputation
df_classif[categorical_cols] = df_classif[categorical_cols].fillna(categorical_modes)

# Verification (optional): Check if nulls remain in categorical columns
# print("\nNull count after categorical imputation:")
# print(df_classif[categorical_cols].isnull().sum().nlargest(5))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Identify Numerical Columns ---
# # Assuming you have already imputed and cleaned your data
# numerical_cols = df_classif.select_dtypes(include=np.number).columns.tolist()

# --- 2. Apply Standardization (Z-Score) ---
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
df_classif[numerical_cols] = scaler.fit_transform(df_classif[numerical_cols])


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. NEW STATUS MAP (Implementing your specific rules) ---
status_map_new = {
    # Good Loans (0)
    'Fully Paid': 0, 
    'Does not meet the credit policy. Status:Fully Paid': 0, 
    'Current': 0, # USER MANDATE: Treating 'Current' as Good/Not Troubled

    # Bad/Troubled Loans (1)
    'Charged Off': 1, 
    'Default': 1, 
    'Late (31-120 days)': 1, 
    'Late (16-30 days)': 1, 
    'Does not meet the credit policy. Status:Charged Off': 1, 
    'In Grace Period': 1, # USER MANDATE: Treating 'In Grace Period' as Bad/Troubled
}

# --- 2. CREATE NEW TARGET COLUMN AND DEFINE FEATURE SETS ---

# Create the new binary target column
TARGET_COL_NEW = 'is_bad_loan'
df_classif[TARGET_COL_NEW] = df_classif['loan_status'].map(status_map_new).fillna(1).astype(int) 

# One-Hot Encoding (Low Cardinality: <= 20 unique values)
# NOTE: 'grade' is now a FEATURE, as risk grade is known at origination.
OHE_COLS = [
    'purpose', 'emp_length', 'grade', # 'grade' is moved here!
    'home_ownership', 'verification_status', 'term', 'initial_list_status'
]

# Label Encoding (High Cardinality: > 20 unique values)
LE_COLS = [
    'zip_code', 'earliest_cr_line', 'last_credit_pull_d', 
    'next_pymnt_d', 'last_pymnt_d', 'issue_d', 
    'addr_state', 'sub_grade'
]

# Define the target vector
y = df_classif[TARGET_COL_NEW]

# Drop the new target column AND the original 'loan_status' column (to prevent direct leakage)
COLS_TO_DROP_FROM_X = [TARGET_COL_NEW, 'loan_status']
X = df_classif.drop(columns=COLS_TO_DROP_FROM_X, errors='ignore').copy() 


# --- 3. APPLY FEATURE ENCODING TO X (The Features) ---

# A. One-Hot Encoding (Low Cardinality Columns)
X = pd.get_dummies(X, columns=OHE_COLS, drop_first=True)

# B. Label Encoding (High Cardinality Columns - Overwriting Original)
for col in LE_COLS:
    if col in X.columns:
        X[col], _ = pd.factorize(X[col])


# --- 4. TRAIN-TEST SPLIT (Binary Classification) ---

print("--- FEATURE & TARGET DEFINITION COMPLETE ---")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the data, stratifying by the target variable (y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("-" * 50)
print("TRAIN-TEST SPLIT COMPLETE (Binary Classification):")
print(f"Target distribution in y_train:\n{y_train.value_counts(normalize=True)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("-" * 50)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# --- NOTE: Assuming X_train, X_test, y_train, y_test are already defined ---
# For demonstration purposes, we assume X_train is a DataFrame and y_train is a numpy array.

# 1. Initialize and Train the Random Forest Model
# We use a base model with a small number of estimators for speed, 
# as the goal is only to rank feature importance.
print("Training Random Forest to determine Feature Importance...")
selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector_model.fit(X_train, y_train)

# 2. Define the Selection Threshold
# We use the median feature importance score as the threshold.
# Features with importance > median will be kept.
threshold = np.median(selector_model.feature_importances_)

print(f"Median Feature Importance Threshold: {threshold:.4f}")

# 3. Apply Feature Selection
# SelectFromModel creates a meta-transformer that selects features based on importance.
sfm = SelectFromModel(selector_model, threshold=threshold, prefit=True)

# Transform the training and testing sets to retain only the selected features
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# 4. Verification and Reporting (Optional, but highly recommended)
# Get the names of the selected features
selected_feature_indices = sfm.get_support(indices=True)
selected_features = X_train.columns[selected_feature_indices].tolist()

print("\n--- FEATURE SELECTION RESULTS ---")
print(f"Original feature count: {X_train.shape[1]}")
print(f"Selected feature count: {X_train_selected.shape[1]}")
print(f"Selected features: {selected_features}")
print("-" * 50)

# Update the X variables to the selected set
X_train = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)

print("X_train and X_test have been updated to include only the selected features.")

import numpy as np
import pandas as pd

# --- NOTE: Assuming X_train and X_test are your feature DataFrames ---
# X_train, X_test = ...

# 1. Calculate the Correlation Matrix (on the training data)
corr_matrix = X_train.corr().abs()

# 2. Identify Highly Correlated Features (using the upper triangle of the matrix)
# We set the threshold at 0.90
UPPER = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
features_to_drop = [column for column in UPPER.columns if any(UPPER[column] > 0.90)]

print("--- Highly Correlated Feature Removal ---")
print(f"Correlation Threshold: 0.90 (Absolute Value)")

if not features_to_drop:
    print("\n✅ No features found to be highly correlated above the 0.90 threshold.")
else:
    print(f"\n❌ {len(features_to_drop)} Features identified for removal:")
    for feature in features_to_drop:
        # Find the feature it was most correlated with for reporting
        most_correlated_with = corr_matrix[feature][(corr_matrix[feature] > 0.90) & (corr_matrix[feature] < 1.0)].idxmax()
        print(f"    - Dropping '{feature}' (Correlated with '{most_correlated_with}')")

    # 3. Drop the identified features from BOTH training and testing sets
    X_train = X_train.drop(columns=features_to_drop, errors='ignore')
    X_test = X_test.drop(columns=features_to_drop, errors='ignore')

    print("\n--- NEW FEATURE COUNTS ---")
    print(f"X_train new shape: {X_train.shape}")
    print(f"X_test new shape: {X_test.shape}")

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier # Used as base estimator for AdaBoost
from sklearn.linear_model import LogisticRegression # NEW
from sklearn.neural_network import MLPClassifier # NEW

# --- NOTE: Replace the following variables with your actual loaded data ---
# This is a placeholder section to ensure the code is complete.
# X_train, X_test, y_train, y_test = ... 
# --------------------------------------------------------------------------

# Define a single, consistent metric for evaluation (e.g., accuracy)
SCORING_METRIC = 'accuracy' 
N_ITER = 10  # Number of parameter settings that are sampled (reduce for speed)
CV_FOLDS = 3 # Number of cross-validation folds

# --- MODEL DEFINITIONS AND HYPERPARAMETER GRIDS ---

IMBALANCE_RATIO = 7.427 # Derived from 328762 / 44266

xgb_params = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'objective': ['binary:logistic'],  # CHANGED to Binary
    'eval_metric': ['logloss'],         # CHANGED to Binary
    'scale_pos_weight': [IMBALANCE_RATIO], # ADDED for Class Balancing
    'use_label_encoder': [False],
}

# 2. CatBoost Grid
# Note: CatBoost handles categorical features automatically, but here we assume X is fully numerical/encoded.
cbt_params = {
    'iterations': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128],
    'auto_class_weights': ['Balanced'], # ADDED for Class Balancing
    'loss_function': ['Logloss'],       # CHANGED to Binary
    'verbose': [0],
}

# 3. AdaBoost Grid
# AdaBoost typically uses a Decision Tree base estimator
ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'estimator__max_depth': [1, 2, 3],
    'estimator__class_weight': ['balanced'], # ADDED for Class Balancing
}

# Logistic Regression Grid
log_params = {
    # C is the inverse of regularization strength (smaller C = stronger regularization)
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga'], # Solvers compatible with l1/l2/elasticnet
    'class_weight': ['balanced'] # Imbalance Handling
}

# MLP Classifier Grid (Neural Network)
mlp_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], # Tries different network architectures
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01], # L2 regularization term
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [100, 200] # Reduced max_iter for faster tuning
}

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# --- REQUIRED VARIABLES (from previous steps) ---
# NOTE: Replace these placeholders with your actual data and parameter grid definitions
# X_train, X_test, y_train, y_test = ...
N_ITER = 10 
CV_FOLDS = 3 
SCORING_METRIC = 'accuracy' 

# Define the cross-validation strategy explicitly
# StratifiedKFold ensures each fold has the same proportion of loan grades as the whole set
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
# ------------------------------------------------------------------------------------

# Define the models list (ensuring AdaBoost base estimator is set)
model_list = [
    ('Logistic Regression', LogisticRegression(random_state=42, n_jobs=-1, solver='saga', max_iter=1000, penalty='l2'), log_params),
    ('MLP Classifier', MLPClassifier(random_state=42), mlp_params),
    ('XGBoost', XGBClassifier(random_state=42, n_jobs=-1, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False), xgb_params),
    ('CatBoost', CatBoostClassifier(random_state=42, thread_count=-1, loss_function='Logloss', verbose=0), cbt_params),
    ('AdaBoost', AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)), ada_params),
]

model_results = []
best_estimators = {}

for name, classifier, params in model_list:
    print(f"--- Starting {name} Tuning (Class Balanced) ---")

    # Fix AdaBoost naming issue and re-instantiate base estimator if needed
    if name == 'AdaBoost':
        classifier = AdaBoostClassifier(
            random_state=42,
            estimator=DecisionTreeClassifier(random_state=42)
        )

    # 1. Initialize RandomizedSearchCV with StratifiedKFold
    rand_search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=params,
        n_iter=N_ITER,
        cv=cv_strategy, 
        scoring=SCORING_METRIC,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # 2. Train and tune (ONLY on X_train/y_train)
    rand_search.fit(X_train, y_train)

    # 3. Store best model and predict on the FINAL, UNSEEN X_test
    best_model = rand_search.best_estimator_
    best_estimators[name] = best_model
    y_pred = best_model.predict(X_test)

    # 4. Compute and Save Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    model_results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'Best Params': rand_search.best_params_
    })

    # Detailed report per model
    print(f"\n✅ {name} Best Hyperparameters: {rand_search.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n" + "="*60 + "\n")

# 5. Create Comparison Table
results_df = pd.DataFrame(model_results).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

# Display cleanly
pd.set_option('display.max_colwidth', None)
print("\n📊 Model Performance Comparison (after Hyperparameter Tuning):")
print(results_df)

# Optional: save to CSV 
# results_df.to_csv("model_performance_comparison.csv", index=False)