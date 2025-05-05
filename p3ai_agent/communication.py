import threading
import paho.mqtt.client as mqtt
from typing import List, Callable, Optional, Dict, Any
from langchain.tools import Tool
from pydantic import BaseModel, Field

class ConnectMQTTInput(BaseModel):
    url: str = Field(description="The URL of the MQTT broker to connect to, e.g., mqtt://localhost:1883")

class DisconnectMQTTInput(BaseModel):
    pass

class SendMQTTMessageInput(BaseModel):
    message: str = Field(description="The message to send to the MQTT broker.")

class ReadMQTTMessagesInput(BaseModel):
    pass

class SubscribeInput(BaseModel):
    topic: str = Field(..., description="The MQTT topic to subscribe to, e.g., 'sensor/data'.")

class UnsubscribeInput(BaseModel):
    topic: str = Field(..., description="The MQTT topic to unsubscribe from, e.g., 'sensor/data'.")

class MQTTAgentWrapper:
    """
    A wrapper class that provides MQTT functionality to any LangChain agent executor.
    Users only need to pass their agent executor that will use the MQTT tools provided by this wrapper.
    """
    
    def __init__(self, client_id: str = "mqtt_agent", default_topic: str = "collaborate"):
        """
        Initialize the MQTT wrapper.
        
        Args:
            client_id: The MQTT client ID to use
            default_topic: The default topic to subscribe to and publish on
        """
        self.mqtt_messages = []  # Store received messages
        self.connected = False  # Connection state
        self.agent_executor = None  # Will store the user's agent executor
        self.topic = default_topic  # Default topic
        self.client_id = client_id
        
        # Initialize MQTT client and setup callbacks
        self.mqtt_client = mqtt.Client(client_id=client_id)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        
        # Store the tools for easy access
        self.tools = self._create_tools()
    
    def _on_message(self, client, userdata, msg):
        """Callback when a message is received from the MQTT broker."""
        received_message = msg.payload.decode()
        print(f"[{self.client_id}] Received: {received_message}")
        self.mqtt_messages.append(received_message)

        # Automatically invoke the agent to respond if set
        if self.agent_executor:
            # Run the response generation in a separate thread to avoid blocking
            threading.Thread(target=self._auto_respond, args=(received_message,)).start()

    def _auto_respond(self, message):
        """Automatically respond to incoming message using the agent."""
        print(f"[{self.client_id}] Processing received message: {message}")
        if self.agent_executor:
            response = self.agent_executor.invoke({"input": f"Respond to this message: {message}"})
            print(f"[{self.client_id}] Auto-processed with response: {response['output']}")
            # Automatically send the response back
            self.send_message(response['output'])
        else:
            print(f"[{self.client_id}] Agent executor not set, cannot auto-respond")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when the MQTT client successfully connects."""
        if rc == 0:
            print(f"[{self.client_id}] Connected to MQTT broker.")
            client.subscribe(self.topic, qos=1)
            self.connected = True
        else:
            print(f"[{self.client_id}] Failed to connect to MQTT broker, return code: {rc}")
            self.connected = False

    def connect_mqtt(self, url: str) -> str:
        """Connect to the MQTT broker."""
        if self.connected:
            return "Already connected to MQTT broker."
            
        try:
            # Parse URL and connect
            parts = url.replace("mqtt://", "").split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 1883

            self.mqtt_client.connect(host, port)
            self.mqtt_client.loop_start()
            
            # Give a small delay to allow connection to establish
            import time
            time.sleep(0.5)
            
            print(f"[{self.client_id}] Connected to {host}:{port}")
            return f"Connected to MQTT broker at {host}:{port}"
        except Exception as e:
            print(f"[{self.client_id}] Error connecting to MQTT broker: {e}")
            self.connected = False
            return f"Error connecting to MQTT broker: {e}"

    def disconnect_mqtt(self) -> str:
        """Disconnect from the MQTT broker."""
        if not self.connected:
            print(f"[{self.client_id}] Not connected to MQTT broker.")
            return "Not connected to MQTT broker."

        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self.connected = False
        print(f"[{self.client_id}] Disconnected from MQTT broker.")
        return "Disconnected from MQTT broker."

    def send_message(self, message: str) -> str:
        """Send a message to the MQTT broker."""
        if not self.connected:
            print(f"[{self.client_id}] Not connected to MQTT broker.")
            return "Not connected to MQTT broker. Use connect_mqtt tool first."

        self.mqtt_client.publish(self.topic, message, qos=1)
        print(f"[{self.client_id}] Sent message to topic '{self.topic}': {message}")
        return f"Message sent to topic '{self.topic}': {message}"

    def read_messages(self) -> str:
        """Read messages received from the MQTT broker."""
        if not self.connected:
            return "Not connected to MQTT broker. Use connect_mqtt tool first."
            
        if not self.mqtt_messages:
            print(f"[{self.client_id}] No new messages.")
            return "No new messages."
        
        messages = "\n".join(self.mqtt_messages)
        # Make a copy of messages before clearing
        message_copy = messages
        self.mqtt_messages.clear()  # Clear messages after reading
        print(f"[{self.client_id}] Read messages: {message_copy}")
        return f"Messages received:\n{message_copy}"
    
    def _subscribe_topic(self, topic: str) -> str:
        """Subscribe to a specific topic."""
        if not self.connected:
            print(f"[{self.client_id}] Not connected to MQTT broker.")
            return

        self.mqtt_client.subscribe(topic, qos=1)
        return f"Subscribed to topic '{topic}'."
    
    def _unsubscribe_topic(self, topic: str) -> str:
        """Unsubscribe from a specific topic."""
        if not self.connected:
            print(f"[{self.client_id}] Not connected to MQTT broker.")
            return

        self.mqtt_client.unsubscribe(topic)
        return f"Unsubscribed from topic '{topic}'."
    
    def _create_tools(self) -> List[Tool]:

        """Create the MQTT tools using this instance."""

        connect_mqtt_tool = Tool(
            name="connect_mqtt",
            func=lambda url: self.connect_mqtt(url),
            description="""
            This tool allows the agent to connect to an MQTT broker.
            The URL should be in the format: mqtt://<host>:<port>.
            After connecting, you can use other tools to send and receive messages from the broker.
            Example: connect_mqtt("mqtt://localhost:1883")
            """,
            args_schema=ConnectMQTTInput,
            return_direct=False,
        )
        
        disconnect_mqtt_tool = Tool(
            name="disconnect_mqtt",
            func=lambda: self.disconnect_mqtt(),
            description="""
            This tool allows the agent to disconnect from the MQTT broker once the task is completed.
            You can safely use this tool when you're finished interacting with the broker.
            Example: disconnect_mqtt()
            """,
            args_schema=DisconnectMQTTInput,
            return_direct=False,
        )
        
        send_message_tool = Tool(
            name="send_mqtt_message",
            func=lambda message: self.send_message(message),
            description="""
            This tool allows the agent to send a message to the MQTT broker.
            The message will be published to the 'collaborate' topic.
            Example: send_message("Hello, this is Agent A!")
            """,
            args_schema=SendMQTTMessageInput,
            return_direct=False,
        )
        
        read_messages_tool = Tool(
            name="read_mqtt_messages",
            func=lambda: self.read_messages(),
            description="""
            This tool allows the agent to read messages received from the MQTT broker.
            The messages will be retrieved from the 'collaborate' topic.
            After reading, the messages are cleared.
            Example: read_messages()
            Note: This tool should be used after connecting to the broker.
            """,
            args_schema=ReadMQTTMessagesInput,
            return_direct=False,
        )

        subscribe_topic_tool = Tool(
            name="subscribe_to_mqtt_topic",
            func=lambda topic: self._subscribe_topic(topic),
            description="Subscribe to a specific MQTT topic to start receiving messages from it.",
            return_direct=False,
        )
        unsubscribe_topic_tool = Tool(
            name="unsubscribe_from_mqtt_topic",
            func=lambda topic: self._unsubscribe_topic(topic),
            description="Unsubscribe from a specific MQTT topic to stop receiving messages from it.",
            return_direct=False,
        )

        return [connect_mqtt_tool, disconnect_mqtt_tool, send_message_tool, read_messages_tool, subscribe_topic_tool, unsubscribe_topic_tool]

    def set_agent_executor(self, agent_executor) -> None:
        """Set the agent executor to use for processing messages."""
        self.agent_executor = agent_executor
        print(f"[{self.client_id}] Agent executor set for automatic responses.")

    def get_tools(self) -> List[Tool]:
        """Get the MQTT tools for use with the agent."""
        return self.tools