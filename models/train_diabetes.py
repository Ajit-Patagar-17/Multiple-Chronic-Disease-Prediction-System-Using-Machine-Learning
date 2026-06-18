import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
df = pd.read_csv("../data/Diabetes.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Dictionary to store models and their accuracy
models = {}
accuracies = {}

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracies['Random Forest'] = accuracy_score(y_test, y_pred_rf)
models['Random Forest'] = rf_model

# -----------------------------
# Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracies['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
models['Logistic Regression'] = lr_model

# -----------------------------
# SVM
# -----------------------------
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracies['SVM'] = accuracy_score(y_test, y_pred_svm)
models['SVM'] = svm_model

# -----------------------------
# XGBoost
# -----------------------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracies['XGBoost'] = accuracy_score(y_test, y_pred_xgb)
models['XGBoost'] = xgb_model

# -----------------------------
# Tuned XGBoost
# -----------------------------
params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_distributions=params,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1
)
search.fit(X_train, y_train)
xgb_tuned_model = search.best_estimator_
y_pred_xgb_tuned = xgb_tuned_model.predict(X_test)
accuracies['Tuned XGBoost'] = accuracy_score(y_test, y_pred_xgb_tuned)
models['Tuned XGBoost'] = xgb_tuned_model

# -----------------------------
# Print accuracies
# -----------------------------
print("\nModel Accuracies:")
for model_name, acc in accuracies.items():
    print(f"{model_name}: {acc:.4f}")

# -----------------------------
# Save best model & all models
# -----------------------------
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "Diabetes_best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(model_dir, "Diabetes_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(model_dir, "Diabetes_features.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

# Save accuracies for final comparison graph
pickle.dump(accuracies, open(os.path.join(model_dir, "Diabetes_model_accuracies.pkl"), "wb"))

print(f"\n✅ Best model ({best_model_name}) and accuracies saved successfully!")
