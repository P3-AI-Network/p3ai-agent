# This is a simple mqtt client for testing agnet communication from agnet_communication_example.py
# It connects to the same broker and subscribes to the same topic

import paho.mqtt.client as mqtt
import time

mqtt_messages = []
port = 1883

def on_message(client, userdata, msg):
    print(f"[Agent B] Got: {msg.payload.decode()}")
    # Store the message if you want to process it later
    mqtt_messages.append(msg.payload.decode())

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[Agent B] Connected to MQTT broker on localhost:{port}")
        # Subscribe with QoS 1
        client.subscribe("collaborate", qos=1)
    else:
        print(f"[Agent B] Failed to connect to MQTT broker, return code: {rc}")

def on_disconnect(client, userdata, rc, *args):
    if rc != 0:
        print(f"[Agent B] Unexpected disconnection, code: {rc}")
    else:
        print("[Agent B] Client disconnected successfully")

# Create client with unique ID and clean session
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, clean_session=True, client_id="agent_2")
client.on_message = on_message
client.on_connect = on_connect

# Connect with keep alive of 60 seconds
client.connect("localhost", port, keepalive=60)
client.loop_start()

# Wait a moment to ensure connection is established
time.sleep(1)

def send_loop():
    while True:
        msg = input("[Agent B] Say: ")
        if not msg:
            continue
        # Publish with QoS 1
        client.publish("collaborate", msg, qos=1)
        print(f"[Agent B] Message sent via port {port} on topic 'collaborate'")

print(f"[Agent B] Starting MQTT client on port {port}")
send_loop()