import time, random
import pandas as pd
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"

def gen_normal():
    return random.gauss(40, 2.2), max(0.01, random.gauss(0.20, 0.05)), random.gauss(1.0, 0.10)

def gen_alert():
    return random.gauss(60, 3.0), random.gauss(0.75, 0.12), random.gauss(1.45, 0.12)

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")

st.title("🛠️ Predictive Maintenance — Live Demo (via API)")
st.markdown("This page shows **fake** sensor readings sent to the **Flask API**, "
            "which runs the AI model and returns predictions.")

status_box = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

data = pd.DataFrame(columns=["temperature_c","vibration_g","current_a","risk","status"])

for i in range(50):  # shorter demo
    # Simulate sensor reading
    if random.random() < 0.10:
        t, v, c = gen_alert()
    else:
        t, v, c = gen_normal()

    # Call Flask API
    try:
        response = requests.get(API_URL, params={"temp": t, "vib": v, "cur": c})
        result = response.json()
    except Exception as e:
        st.error(f"⚠️ Could not reach API: {e}")
        break

    row = {
        "temperature_c": result["temperature"],
        "vibration_g": result["vibration"],
        "current_a": result["current"],
        "risk": result["risk"],
        "status": result["status"]
    }
    data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)

    # Status light
    if row["status"].startswith("ALERT"):
        status_box.error(f"ALERT 🚨 Risk {row['risk']*100:.1f}%", icon="⚠️")
    else:
        status_box.success(f"OK ✅ Risk {row['risk']*100:.1f}%", icon="✅")

    # Chart
    chart_placeholder.line_chart(data[["temperature_c","vibration_g","current_a","risk"]])

    # Table (last 10 rows)
    table_placeholder.dataframe(data.tail(10), use_container_width=True)

    time.sleep(0.6)

st.info("Demo finished. Press R to rerun.", icon="ℹ️")
