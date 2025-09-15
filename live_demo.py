import time, random
import numpy as np
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
scaler = joblib.load(BASE / "scaler.pkl")

# Try to load TensorFlow model; otherwise use scikit-learn fallback
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(BASE / "model.h5")
    USE_TF = True
except Exception:
    from joblib import load
    model = load(BASE / "model_sklearn.pkl")
    USE_TF = False

def predict_one(temp, vib, cur):
    x = scaler.transform([[temp, vib, cur]])
    if USE_TF:
        p = float(model.predict(x, verbose=0)[0][0])
    else:
        p = float(model.predict_proba(x)[0][1])
    return p

def gen_normal():
    return random.gauss(40, 2.2), max(0.01, random.gauss(0.20, 0.05)), random.gauss(1.0, 0.10)

def gen_alert():
    return random.gauss(60, 3.0), random.gauss(0.75, 0.12), random.gauss(1.45, 0.12)

print("Live demo started (20 readings).")
print("Columns: temperature °C | vibration g | current A | risk% | status")

for i in range(20):  # show only 20 readings
    if random.random() < 0.10:
        t, v, c = gen_alert()
    else:
        t, v, c = gen_normal()

    risk = predict_one(t, v, c)
    status = "ALERT 🚨" if risk >= 0.5 else "OK ✅"
    print(f"{t:6.2f} | {v:6.3f} | {c:5.3f} | {risk*100:5.1f}% | {status}")
    time.sleep(0.8)

print("Demo finished ✅")
