from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
scaler = joblib.load(BASE / "scaler.pkl")

# Try TensorFlow, else fallback
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(BASE / "model.h5")
    USE_TF = True
except Exception:
    model = joblib.load(BASE / "model_sklearn.pkl")
    USE_TF = False

app = Flask(__name__)

def predict_one(temp, vib, cur):
    x = scaler.transform([[temp, vib, cur]])
    if USE_TF:
        p = float(model.predict(x, verbose=0)[0][0])
    else:
        p = float(model.predict_proba(x)[0][1])
    return p

@app.route("/predict", methods=["GET"])
def predict():
    try:
        temp = float(request.args.get("temp"))
        vib  = float(request.args.get("vib"))
        cur  = float(request.args.get("cur"))
    except:
        return jsonify({"error": "Missing or invalid parameters"}), 400

    risk = predict_one(temp, vib, cur)
    status = "ALERT 🚨" if risk >= 0.5 else "OK ✅"
    return jsonify({
        "temperature": temp,
        "vibration": vib,
        "current": cur,
        "risk": risk,
        "status": status
    })

if __name__ == "__main__":
    app.run(debug=True)
