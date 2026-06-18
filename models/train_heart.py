import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

df = pd.read_csv("../data/Heart Dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Improved Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    ),
    "SVM": SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model

    print("Accuracy:", acc)
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Final Comparison
print("\n===== Final Comparison =====")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

# Best model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy = {results[best_model_name]:.4f}")

# Save best model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Best model and scaler saved successfully!")
