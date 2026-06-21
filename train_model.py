"""
train_model.py

Trains a gesture classifier on the landmark CSV produced by extract_landmarks.py.
Tries an MLP (good default for landmark vectors) and saves the trained model +
label encoder to models/.

Usage:
    python train_model.py --csv data/landmarks.csv
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


def train(csv_path, model_out="models/gesture_model.pkl", encoder_out="models/label_encoder.pkl"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLP works well on normalized landmark vectors
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== MLP Classifier ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Also try Random Forest as a quick comparison / fallback
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("\n=== Random Forest (comparison) ===")
    print(classification_report(y_test, rf_pred, target_names=le.classes_))

    # Pick whichever scored better on test accuracy
    mlp_acc = (y_pred == y_test).mean()
    rf_acc = (rf_pred == y_test).mean()
    best_model, best_name = (model, "MLP") if mlp_acc >= rf_acc else (rf, "RandomForest")
    print(f"\nSelected best model: {best_name} (acc={max(mlp_acc, rf_acc):.4f})")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(best_model, model_out)
    joblib.dump(le, encoder_out)
    print(f"Saved model to {model_out}")
    print(f"Saved label encoder to {encoder_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/landmarks.csv", help="Path to landmarks CSV")
    parser.add_argument("--model_out", default="models/gesture_model.pkl")
    parser.add_argument("--encoder_out", default="models/label_encoder.pkl")
    args = parser.parse_args()

    train(args.csv, args.model_out, args.encoder_out)