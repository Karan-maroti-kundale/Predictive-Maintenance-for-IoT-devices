
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Try TensorFlow first; fall back to scikit-learn if TF is missing
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

BASE = Path(__file__).resolve().parent
CSV = BASE / "data" / "sensor_data.csv"

def load_xy():
    df = pd.read_csv(CSV)
    X = df[["temperature_c", "vibration_g", "current_a"]].values
    y = df["fail_soon"].values
    return X, y

def train_with_tensorflow(X_train, y_train, X_val, y_val):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=12, batch_size=64, validation_data=(X_val, y_val), verbose=0)
    return model

def main():
    X, y = load_xy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    if TF_AVAILABLE:
        print("Training with TensorFlow...")
        model = train_with_tensorflow(X_train_s, y_train, X_val_s, y_val)
        # Evaluate
        val_pred = (model.predict(X_val_s) > 0.5).astype(int).ravel()
        acc = accuracy_score(y_val, val_pred)
        print("Validation Accuracy:", round(acc, 4))
        print(classification_report(y_val, val_pred))
        # Save
        model.save(BASE / "model.h5")
    else:
        print("TensorFlow not found, using scikit-learn instead (LogisticRegression).")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
        val_pred = model.predict(X_val_s)
        acc = accuracy_score(y_val, val_pred)
        print("Validation Accuracy:", round(acc, 4))
        print(classification_report(y_val, val_pred))
        joblib.dump(model, BASE / "model_sklearn.pkl")

    joblib.dump(scaler, BASE / "scaler.pkl")
    print("Saved model and scaler. Done!")

if __name__ == "__main__":
    main()
