"""
stability_classifier.py
Classify smart grid stability states using XGBoost + SMOTE + scaling.
Outputs both single split and cross-validation results to artifacts/stability_report.txt
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from preprocessing import load_stability

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def train_stability_classifier():
    df = load_stability()

    # Features and labels
    X = df.drop(columns=["stabf", "stab"])
    y = df["stabf"]

    # Encode labels (stable → 0, unstable → 1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance with SMOTE (training only)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Model pipeline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        ))
    ])

    # Train
    clf.fit(X_train_res, y_train_res)

    # Predict on hold-out test set
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    report_split = classification_report(y_test, preds, target_names=le.classes_)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    cv_acc_mean = np.mean(scores)
    cv_acc_std = np.std(scores)

    # Confusion matrix (hold-out)
    cm = confusion_matrix(y_test, preds)

    # Print to console
    print("🔹 Stability Classifier Results")
    print(f"Hold-out Accuracy: {acc:.4f}")
    print(f"Hold-out F1 (weighted): {f1:.4f}")
    print(report_split)
    print(f"5-fold CV Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print("Confusion Matrix (hold-out):")
    print(cm)

    # Save report for dashboard
    report_path = ART / "stability_report.txt"
    with open(report_path, "w") as f:
        f.write("Grid Stability Classifier\n")
        f.write(f"Hold-out Accuracy: {acc:.4f}\n")
        f.write(f"Hold-out F1 (weighted): {f1:.4f}\n\n")
        f.write(report_split + "\n")
        f.write(f"5-fold CV Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}\n\n")
        f.write("Confusion Matrix (hold-out):\n")
        f.write(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())
    print(f"📁 Report saved to {report_path}")

    return clf, le


if __name__ == "__main__":
    model, encoder = train_stability_classifier()
