
# AI-Powered Predictive Maintenance (No Hardware!) — Kid-Friendly Project

**Goal:** Make a smart helper that watches a machine’s *temperature*, *vibration*, and *current*.
It warns us **before** the machine gets sick. We will **simulate** the machine (no sensors needed).

## What you will use
- **Google Colab** or your computer with **Python 3.10+**
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `flask`, `streamlit`
- Files in this folder (already prepared for you)

## Quick Start (the easy way)
1. **Install packages** (one time):
   ```bash
   pip install -r requirements.txt
   ```
2. **Make (or remake) the fake sensor data**:
   ```bash
   python generate_data.py
   ```
3. **Train the AI** (it learns from the data):
   ```bash
   python train_model.py
   ```
   After training, you’ll see a file `model.h5` (the brain) and `scaler.pkl` (a helper for numbers).
4. **Live console demo** (watch readings + warnings):
   ```bash
   python live_demo.py
   ```
5. **Simple dashboard (pretty screen)**:
   ```bash
   streamlit run dashboard_streamlit.py
   ```
6. **API server (for apps to ask the AI)**:
   ```bash
   python api_flask.py
   ```
   Test it from another terminal:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict         -H "Content-Type: application/json"         -d "{\"temperature_c\": 62, \"vibration_g\": 0.85, \"current_a\": 1.55}"
   ```

## Folder Map
- `data/sensor_data.csv` → the fake sensor readings.
- `generate_data.py` → makes new fake data.
- `train_model.py` → teaches the AI.
- `live_demo.py` → shows live predictions in text.
- `dashboard_streamlit.py` → live chart + status light.
- `api_flask.py` → tiny website for predictions (JSON in, answer out).

## How it works (like a story)
- The **machine** is a character. When it gets **hotter**, **shakier**, and **uses more current**, it is **not feeling well**.
- We show many examples to the computer. It learns patterns with **TensorFlow**.
- In live mode, new readings go to the AI. If risk is high → **ALERT** (fix before it breaks).

Have fun and make it yours! Try changing limits in `live_demo.py` to see more/less alerts.
