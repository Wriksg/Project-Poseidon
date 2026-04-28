import zmq
import time
import pandas as pd
import json

# 1. Initialize the ZeroMQ Context and Publisher Socket
context = zmq.Context()
socket = context.socket(zmq.PUB)
# Bind to port 5555 on the local machine
socket.bind("tcp://*:5555")


def stream_historical_data(csv_filepath):
    print("ZeroMQ Enterprise Publisher Active.")
    print("Broadcasting on TCP Port 5555... Waiting 3 seconds for Radar to lock on.")

    # Critical: ZeroMQ needs a brief moment to bind before firing data
    time.sleep(3)

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}")
        return

    for index, row in df.iterrows():
        # Package the payload exactly as your FastAPI backend expects it
        payload = {
            "timestamp": row['timestamp'],
            "panchet_discharge_cusecs": float(row['panchet_discharge_cusecs']),
            "maithon_discharge_cusecs": float(row['maithon_discharge_cusecs']),
            "ghatal_water_level_m": float(row['ghatal_water_level_m'])
        }

        # Serialize to JSON and blast it over the TCP socket
        socket.send_string(json.dumps(payload))
        print(f"[{payload['timestamp']}] Fired via ZeroMQ | Panchet: {payload['panchet_discharge_cusecs']} cusecs")

        time.sleep(2)


if __name__ == "__main__":
    csv_file = "./data/processed/dvc_cwc_aug2021_replay.csv"
    stream_historical_data(csv_file)