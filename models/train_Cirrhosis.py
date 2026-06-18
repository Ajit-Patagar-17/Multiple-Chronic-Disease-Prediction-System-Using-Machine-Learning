import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
df = pd.read_csv("../data/Cirrhosis_final3.csv")  
print("Dataset shape:", df.shape)
print(df.head())

# Encode all object columns (especially 'Status')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Separate features and target
X = df.drop("Status", axis=1)
y = df["Status"]

# Save original feature names
feature_names = X.columns.tolist()

# Encode the target labels and save the encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store models and their accuracy
models = {}
accuracies = {}

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
accuracies['Random Forest'] = accuracy_score(y_test, y_pred_rf)
models['Random Forest'] = rf_model


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
accuracies['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
models['Logistic Regression'] = lr_model


# Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
accuracies['SVM'] = accuracy_score(y_test, y_pred_svm)
models['SVM'] = svm_model


# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracies['XGBoost'] = accuracy_score(y_test, y_pred_xgb)
models['XGBoost'] = xgb_model


# Tuned XGBoost
xgb_tuned_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_tuned_model.fit(X_train_scaled, y_train)
y_pred_xgb_tuned = xgb_tuned_model.predict(X_test_scaled)
accuracies['Tuned XGBoost'] = accuracy_score(y_test, y_pred_xgb_tuned)
models['Tuned XGBoost'] = xgb_tuned_model


# Print accuracies
print("\nModel Accuracies:")
for model_name, acc in accuracies.items():
    print(f"{model_name}: {acc:.4f}")


# Save the best performing model (example: Tuned XGBoost)
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

# Create output directory if not exists
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "Cirrhosis_best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(model_dir, "Cirrhosis_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(model_dir, "Cirrhosis_feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)
with open(os.path.join(model_dir, "Cirrhosis_label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n✅ Best model ({best_model_name}) saved successfully!")


# Save accuracies for final comparison graph
pickle.dump(accuracies, open(os.path.join(model_dir, "Cirrhosis_model_accuracies.pkl"), "wb"))
