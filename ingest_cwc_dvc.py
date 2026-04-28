import pandas as pd
import os
from datetime import datetime, timedelta


def generate_2021_replay_data():
    data_dir = './data/processed'
    os.makedirs(data_dir, exist_ok=True)

    print("Generating August 2021 Ghatal Flood replay dataset...")

    # Create a timeline from Aug 15 to Aug 17, hourly intervals
    start_time = datetime(2021, 8, 15, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(72)]

    # Simulate the Damodar Valley Corporation (DVC) dam release and Ghatal water levels
    data = []
    for t in timestamps:
        # Normal baseline
        panchet_discharge = 15000
        maithon_discharge = 10000
        ghatal_wl = 4.5  # Danger level is usually around 5.0m

        # The anomalous release starts Aug 16 afternoon
        if t >= datetime(2021, 8, 16, 12, 0):
            panchet_discharge = 85000 + (t.hour * 2000)  # Massive spike
            maithon_discharge = 60000 + (t.hour * 1000)

        # Water level downstream at Ghatal reacts with a delay
        if t >= datetime(2021, 8, 16, 18, 0):
            ghatal_wl = 5.2 + ((t - datetime(2021, 8, 16, 18, 0)).total_seconds() / 3600) * 0.15

        data.append({
            'timestamp': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'panchet_discharge_cusecs': panchet_discharge,
            'maithon_discharge_cusecs': maithon_discharge,
            'ghatal_water_level_m': round(ghatal_wl, 2)
        })

    df = pd.DataFrame(data)
    output_path = os.path.join(data_dir, 'dvc_cwc_aug2021_replay.csv')
    df.to_csv(output_path, index=False)

    print(f"Replay dataset saved to {output_path}")
    print("This file will act as our simulated live data stream for the demo.")


if __name__ == "__main__":
    generate_2021_replay_data()