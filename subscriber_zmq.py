import zmq
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000/predict"

# 1. Initialize the ZeroMQ Context and Subscriber Socket
context = zmq.Context()
socket = context.socket(zmq.SUB)

# Connect to the Publisher's port
socket.connect("tcp://127.0.0.1:5555")
# Subscribe to ALL incoming messages (empty string means catch everything)
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("ZeroMQ Radar Active. Listening on TCP Port 5555...\n")

while True:
    try:
        # 2. Catch the data the millisecond it arrives
        message = socket.recv_string()
        payload = json.loads(message)

        timestamp = payload.get('timestamp')
        cusecs = payload.get('panchet_discharge_cusecs')
        print(f"[X] Intercepted: [{timestamp}] Panchet at {cusecs} cusecs")

        # 3. Forward the data to your local AI model
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            danger = result['prediction']['danger_level']
            depth = result['prediction']['water_depth']
            print(f"    AI Prediction: {danger} | Depth: {depth}m")
        else:
            print(f"    FastAPI Error: Server returned {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("    [!] FastAPI is offline. Start your uvicorn server!")
    except KeyboardInterrupt:
        print("\nShutting down ZeroMQ Radar.")
        break