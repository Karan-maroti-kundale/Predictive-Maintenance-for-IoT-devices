
import numpy as np
import pandas as pd
from pathlib import Path

# Regenerate the synthetic dataset used for training and demos.
rng = np.random.default_rng(7)

def make_dataset(n_rows=12000, normal_ratio=0.65, start="2025-01-01 00:00:00"):
    n_normal = int(n_rows * normal_ratio)
    n_alert = n_rows - n_normal

    # Normal machine readings
    temp_ok = rng.normal(40, 2.2, n_normal)     # °C
    vib_ok  = rng.normal(0.20, 0.05, n_normal)  # g
    cur_ok  = rng.normal(1.00, 0.10, n_normal)  # A

    # "Fail soon" readings
    temp_bad = rng.normal(60, 3.5, n_alert)
    vib_bad  = rng.normal(0.70, 0.10, n_alert)
    cur_bad  = rng.normal(1.40, 0.15, n_alert)

    temp = np.concatenate([temp_ok, temp_bad])
    vib  = np.concatenate([vib_ok, vib_bad])
    cur  = np.concatenate([cur_ok, cur_bad])
    label = np.array([0]*n_normal + [1]*n_alert)

    # Shuffle and add a little noise
    idx = rng.permutation(n_rows)
    temp, vib, cur, label = temp[idx], vib[idx], cur[idx], label[idx]

    drift = np.linspace(0, 0.6, n_rows)
    temp = temp + rng.normal(0, 0.6, n_rows) + drift*0.1
    vib  = vib  + rng.normal(0, 0.02, n_rows)
    cur  = cur  + rng.normal(0, 0.05, n_rows)

    ts = pd.date_range(start=start, periods=n_rows, freq="s")
    df = pd.DataFrame({
        "timestamp": ts,
        "temperature_c": np.round(temp, 2),
        "vibration_g": np.round(vib, 3),
        "current_a": np.round(cur, 3),
        "fail_soon": label.astype(int)
    })
    return df

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    out_path = data_dir / "sensor_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
