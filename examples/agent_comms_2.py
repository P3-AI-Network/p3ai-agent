# This is a simple mqtt client for testing agnet communication from agnet_communication_example.py
# It connects to the same broker and subscribes to the same topic

import paho.mqtt.client as mqtt
import time
from rich.console import Console

console = Console()

mqtt_messages = []
port = 1883

def on_message(client, userdata, msg):
    console.print(f"\nRecieved: {msg.payload.decode()}\n", style="bold green")
    # Store the message if you want to process it later
    mqtt_messages.append(msg.payload.decode())

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        console.print(f"Connected to MQTT broker on localhost:{port}", style="yellow")
        # Subscribe with QoS 1
        client.subscribe("agentb/inbox", qos=1)
    else:
        console.print(f"[Agent B] Failed to connect to MQTT broker, return code: {rc}", style="red")

def on_disconnect(client, userdata, rc, *args):
    if rc != 0:
        console.print(f"[Agent B] Unexpected disconnection, code: {rc}", style="red")
    else:
        console.print("[Agent B] Client disconnected successfully", style="yellow")

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
        print("\n")
        msg = input("[Agent B] Say: ")
        if not msg:
            continue
        # Publish with QoS 1
        client.publish("agenta/inbox", msg, qos=1)
        console.print(f"[Agent B] Sent: {msg}\n", style="bold blue")

print(f"[Agent B] Starting MQTT client on port {port}")
send_loop()