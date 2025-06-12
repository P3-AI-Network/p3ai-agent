import threading
import time
import json 
import logging
from typing import List, Callable, Optional, Dict, Any, Union
import uuid
import paho.mqtt.client as mqtt
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Configure logging with a more descriptive format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MQTTAgentCommunication")

class MQTTMessage:
    """
    Structured message format for agent communication via MQTT.
    
    This class provides a standardized way to format, serialize, and deserialize
    messages exchanged between agents, with support for conversation threading,
    message types, and metadata.
    """
    
    def __init__(
        self,
        content: str,
        sender_id: str,
        receiver_id: Optional[str] = None,
        message_type: str = "query",
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new MQTT message.
        
        Args:
            content: The main message content
            sender_id: Identifier for the message sender
            receiver_id: Identifier for the intended recipient (None for broadcasts)
            message_type: Type categorization ("query", "response", "broadcast", "system")
            message_id: Unique identifier for this message (auto-generated if None)
            conversation_id: ID grouping related messages (auto-generated if None)
            in_reply_to: ID of the message this is responding to (None if not a reply)
            metadata: Additional contextual information
        """
        self.content = content
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.message_id = message_id or str(uuid.uuid4())
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.in_reply_to = in_reply_to
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "content": self.content,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string for transmission."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MQTTMessage':
        """Create message object from dictionary data."""
        return cls(
            content=data.get("content", ""),
            sender_id=data.get("sender_id", "unknown"),
            receiver_id=data.get("receiver_id"),
            message_type=data.get("message_type", "query"),
            message_id=data.get("message_id"),
            conversation_id=data.get("conversation_id"),
            in_reply_to=data.get("in_reply_to"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MQTTMessage':
        """
        Create message object from JSON string.
        
        Handles both valid JSON and fallback for plain text messages.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message as JSON: {e}")

            return cls(
                content=json_str,
                sender_id="unknown",
                message_type="raw"
            )


class AgentCommunicationManager:
    """
    MQTT-based communication manager for LangChain agents.
    
    This class provides tools for LangChain agents to communicate via MQTT,
    enabling multi-agent collaboration through a publish-subscribe pattern.
    """
    
    def __init__(
        self, 
        agent_id: str,
        default_inbox_topic: Optional[str] = None,
        default_outbox_topic: Optional[str] = None,
        auto_reconnect: bool = True,
        message_history_limit: int = 100
    ):
        """
        Initialize the MQTT agent communication manager.
        
        Args:default_outbox_topic
            agent_id: Unique identifier for this agent
            default_inbox_topic: Topic to subscribe to by default
            default_outbox_topic: Topic to publish to by default
            auto_reconnect: Whether to attempt reconnection on failure
            message_history_limit: Maximum number of messages to keep in history
        """

        self.agent_id = agent_id
        self.inbox_topic = default_inbox_topic or f"{agent_id}/inbox"
        self.outbox_topic = default_outbox_topic or f"agents/collaboration"
        self.auto_reconnect = auto_reconnect
        self.message_history_limit = message_history_limit
        

        self.is_connected = False
        self.subscribed_topics = set()
        self.received_messages = []
        self.message_history = []
        self.pending_responses = {} 
        
        self.message_handlers = []
        

        self.mqtt_client = mqtt.Client(client_id=self.agent_id)
        self.mqtt_client.on_connect = self._handle_connect
        self.mqtt_client.on_message = self._handle_message
        self.mqtt_client.on_disconnect = self._handle_disconnect
        
    
    def _handle_message(self, client, userdata, mqtt_message: MQTTMessage):
        """Handle incoming MQTT messages and process them appropriately."""
        try:

            payload = mqtt_message.payload.decode('utf-8')
            topic = mqtt_message.topic
            
            logger.info(f"[{self.agent_id}] Received message on topic '{topic}'")
            

            try:
                message = MQTTMessage.from_json(payload)
                structured = True
            except Exception:

                message = MQTTMessage(
                    content=payload,
                    sender_id="unknown",
                    receiver_id=self.agent_id,
                    message_type="raw"
                )
                structured = False
            

            message_with_metadata = {
                "message": message,
                "topic": topic,
                "received_at": time.time(),
                "structured": structured
            }
            

            self.received_messages.append(message_with_metadata)
            self.message_history.append(message_with_metadata)
            

            if len(self.message_history) > self.message_history_limit:
                self.message_history = self.message_history[-self.message_history_limit:]
            

            if self.agent_executor:
                print("Preparing response...")
                threading.Thread(
                    target=self._generate_response,
                    args=(message, topic)
                ).start()
                

            for handler in self.message_handlers:
                try:
                    handler(message, topic)
                except Exception as e:
                    logger.error(f"Error in custom message handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing incoming message: {e}")


    def _handle_connect(self, client, userdata, flags, rc):
        """Handle successful connection to MQTT broker."""
        if rc == 0:
            self.is_connected = True
            logger.info(f"[{self.agent_id}] Connected to MQTT broker successfully")
            

            self._subscribe_to_topic(self.inbox_topic)
            logger.info(f"[{self.agent_id}] Listening for messages on {self.inbox_topic}")
            

            for topic in self.subscribed_topics:
                if topic != self.inbox_topic:
                    client.subscribe(topic, qos=1)
                    logger.info(f"[{self.agent_id}] Resubscribed to {topic}")
        else:
            self.is_connected = False
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized"
            }
            error_msg = error_messages.get(rc, f"Unknown error (code {rc})")
            logger.error(f"[{self.agent_id}] Failed to connect: {error_msg}")

    def _handle_disconnect(self, client, userdata, rc):
        """Handle disconnection from MQTT broker."""
        self.is_connected = False
        logger.warning(f"[{self.agent_id}] Disconnected from MQTT broker, code {rc}")
        

        if self.auto_reconnect:
            logger.info(f"[{self.agent_id}] Attempting to reconnect...")
            try:
                client.reconnect()
            except Exception as e:
                logger.error(f"[{self.agent_id}] Reconnect failed: {e}")

    def connect_to_broker(self, broker_url: str) -> str:
        """
        Connect to an MQTT broker.
        
        Args:
            broker_url: URL of the MQTT broker (format: mqtt://hostname:port)
            
        Returns:
            Status message about the connection attempt
        """
        if self.is_connected:
            return f"Already connected to MQTT broker as '{self.agent_id}'"
            
        try:

            if broker_url.startswith("mqtt://"):
                broker_url = broker_url[7:] 
                
            # Extract host and port
            if ":" in broker_url:
                host, port_str = broker_url.split(":")
                port = int(port_str)
            else:
                host = broker_url
                port = 1883  # Default MQTT port
            
            # Connect to the broker
            self.mqtt_client.connect(host, port)
            self.mqtt_client.loop_start()
            

            connection_timeout = 3  # seconds
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < connection_timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                return f"Connected to MQTT broker at {host}:{port} as '{self.agent_id}'"
            else:
                return f"Connection attempt to {host}:{port} timed out"
                
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error connecting to MQTT broker: {e}")
            return f"Failed to connect to MQTT broker: {str(e)}"

    def disconnect_from_broker(self) -> str:
        """
        Disconnect from the MQTT broker and clean up resources.
        
        Returns:
            Status message about the disconnection
        """
        if not self.is_connected:
            return "Not currently connected to any MQTT broker"

        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.is_connected = False
            self.received_messages.clear()
            logger.info(f"[{self.agent_id}] Disconnected from MQTT broker")
            return "Successfully disconnected from MQTT broker"
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error during disconnection: {e}")
            return f"Error during disconnection: {str(e)}"

    def send_message(self, message_content: str, message_type: str = "query", receiver_id: Optional[str] = None) -> str:
        """
        Send a message to the current outbox topic.
        
        Args:
            message_content: The main content of the message
            message_type: The type of message being sent
            receiver_id: Specific recipient ID (None for broadcast)
            
        Returns:
            Status message or error
        """
        if not self.is_connected:
            return "Not connected to MQTT broker. Use connect_to_broker first."
        
        try:
            # Create a structured message
            message = MQTTMessage(
                content=message_content,
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                message_type=message_type
            )
            
            # Convert to JSON and publish
            json_payload = message.to_json()
            result = self.mqtt_client.publish(self.outbox_topic, json_payload, qos=1)
            
            if result.rc == 0:
                logger.info(f"[{self.agent_id}] Message sent to '{self.outbox_topic}'")
                
                # Add to history
                self.message_history.append({
                    "message": message,
                    "topic": self.outbox_topic,
                    "sent_at": time.time(),
                    "direction": "outgoing"
                })
                
                # Maintain history limit
                if len(self.message_history) > self.message_history_limit:
                    self.message_history = self.message_history[-self.message_history_limit:]
                    
                return f"Message sent successfully to topic '{self.outbox_topic}'"
            else:
                error_msg = f"Failed to send message, error code: {result.rc}"
                logger.error(f"[{self.agent_id}] {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Error sending message: {str(e)}"
            logger.error(f"[{self.agent_id}] {error_msg}")
            return error_msg

    def read_messages(self) -> str:
        """
        Read and clear the current message queue.
        
        Returns:
            Formatted string of received messages
        """
        if not self.is_connected:
            return "Not connected to MQTT broker. Use connect_to_broker first."
            
        if not self.received_messages:
            return "No new messages in the queue."
        
        # Format messages for output
        formatted_messages = []
        for item in self.received_messages:
            message = item["message"]
            topic = item["topic"]
            
            formatted_msg = (
                f"Topic: {topic}\n"
                f"From: {message.sender_id}\n"
                f"Type: {message.message_type}\n"
                f"Content: {message.content}\n"
            )
            formatted_messages.append(formatted_msg)
        
        # Create a combined output
        output = "Messages received:\n\n" + "\n---\n".join(formatted_messages)
        
        # Clear the received messages queue but keep them in history
        self.received_messages.clear()
        
        return output
    
    def _subscribe_to_topic(self, topic_name: str) -> str:
        """
        Subscribe to a specific MQTT topic.
        
        Args:
            topic_name: The MQTT topic name to subscribe to
            
        Returns:
            Status message
        """
        if not self.is_connected:
            return "Not connected to MQTT broker. Use connect_to_broker first."

        try:
            result = self.mqtt_client.subscribe(topic_name, qos=1)
            if result[0] == 0:
                self.subscribed_topics.add(topic_name)
                logger.info(f"[{self.agent_id}] Subscribed to topic: {topic_name}")
                return f"Successfully subscribed to topic '{topic_name}'"
            else:
                return f"Failed to subscribe to topic '{topic_name}', error code: {result[0]}"
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error subscribing to topic: {e}")
            return f"Error subscribing to topic: {str(e)}"
    
    def _unsubscribe_from_topic(self, topic_name: str) -> str:
        """
        Unsubscribe from a specific MQTT topic.
        
        Args:
            topic_name: The MQTT topic name to unsubscribe from
            
        Returns:
            Status message
        """
        if not self.is_connected:
            return "Not connected to MQTT broker. Use connect_to_broker first."

        # Prevent unsubscribing from the primary inbox
        if topic_name == self.inbox_topic:
            return f"Cannot unsubscribe from primary inbox topic '{self.inbox_topic}'"
            
        try:
            result = self.mqtt_client.unsubscribe(topic_name)
            if result[0] == 0:
                if topic_name in self.subscribed_topics:
                    self.subscribed_topics.remove(topic_name)
                logger.info(f"[{self.agent_id}] Unsubscribed from topic: {topic_name}")
                return f"Successfully unsubscribed from topic '{topic_name}'"
            else:
                return f"Failed to unsubscribe from topic '{topic_name}', error code: {result[0]}"
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error unsubscribing from topic: {e}")
            return f"Error unsubscribing from topic: {str(e)}"
    
    def _change_outbox_topic(self, topic_name: str) -> str:
        """
        Change the default topic for outgoing messages.
        
        Args:
            topic_name: The new MQTT topic name for publishing
            
        Returns:
            Status message
        """
        previous_topic = self.outbox_topic
        self.outbox_topic = topic_name
        logger.info(f"[{self.agent_id}] Changed outbox topic from '{previous_topic}' to '{topic_name}'")
        return f"Changed outbox topic to '{topic_name}'"
    
    def add_message_handler(self, handler_function: Callable) -> None:
        """
        Add a custom message handler function.
        
        Args:
            handler_function: Function to call when messages arrive
                              Should accept (message, topic) parameters
        """
        self.message_handlers.append(handler_function)
        logger.info(f"[{self.agent_id}] Added custom message handler")
        
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get the current connection status and statistics.
        
        Returns:
            Dictionary with connection information
        """
        return {
            "agent_id": self.agent_id,
            "is_connected": self.is_connected,
            "inbox_topic": self.inbox_topic,
            "outbox_topic": self.outbox_topic,
            "subscribed_topics": list(self.subscribed_topics),
            "pending_messages": len(self.received_messages),
            "message_history_count": len(self.message_history)
        }

    def get_message_history(self, limit: int = None, filter_by_topic: str = None) -> List[Dict[str, Any]]:
        """
        Get the message history with optional filtering.
        
        Args:
            limit: Maximum number of messages to return (None for all)
            filter_by_topic: Only return messages from this topic
            
        Returns:
            List of message history entries
        """
        history = self.message_history
        
        # Apply topic filter if specified
        if filter_by_topic:
            history = [msg for msg in history if msg["topic"] == filter_by_topic]
            
        # Apply limit if specified
        if limit is not None:
            history = history[-limit:]
            
        return history
