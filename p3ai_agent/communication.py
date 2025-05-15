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


class MQTTBrokerConnectionInput(BaseModel):
    broker_url: str = Field(
        description="The URL of the MQTT broker (format: mqtt://hostname:port)"
    )

class MQTTMessageInput(BaseModel):
    message_content: str = Field(
        description="The content of the message to send"
    )
    message_type: str = Field(
        default="query", 
        description="Message type (query, response, broadcast, system)"
    )
    receiver_id: Optional[str] = Field(
        default=None,
        description="Specific recipient ID (leave empty for broadcast)"
    )
    
    @property
    def message(self):
        """Alias for message_content to maintain compatibility with existing code"""
        return self.message_content
    
class MQTTTopicInput(BaseModel):
    topic_name: str = Field(
        description="The MQTT topic name (e.g., 'agents/collaboration')"
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
        

        self.agent_executor = None
        self.message_handlers = []
        

        self.mqtt_client = mqtt.Client(client_id=self.agent_id)
        self.mqtt_client.on_connect = self._handle_connect
        self.mqtt_client.on_message = self._handle_message
        self.mqtt_client.on_disconnect = self._handle_disconnect
        

        self.available_tools = self._create_agent_tools()
        
        logger.info(f"Agent '{self.agent_id}' communication manager initialized")
    
    def _handle_message(self, client, userdata, mqtt_message):
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

    def _generate_response(self, incoming_message, source_topic):
        """Generate an automatic response using the configured agent executor."""
        try:
            if not self.agent_executor:
                logger.warning("No agent executor configured for auto-response")
                return
                

            sender = incoming_message.sender_id
            content = incoming_message.content
            msg_type = incoming_message.message_type
            msg_id = incoming_message.message_id
            

            prompt = (
                f"A new message has arrived from {sender} on topic {source_topic}:\n\n"
                f"CONTENT: {content}\n"
                f"MESSAGE TYPE: {msg_type}\n\n"
                f"Consider the context and formulate an appropriate response.\n"
                f"Use the send_mqtt_message tool to reply if needed.\n\n"
                f"Remember to check if the outbox topic is correctly set to reach the sender."
            )
            

            try:

                self.agent_executor.invoke({
                    "input": prompt,
                    "message_context": {
                        "original_message": incoming_message.to_dict(),
                        "source_topic": source_topic,
                        "in_reply_to": msg_id
                    }
                })
            except Exception as api_error:

                if "expected an object, but got a string instead" in str(api_error):
                    logger.warning("Detected OpenAI message format error, attempting with formatted content")

                    self.agent_executor.invoke({
                        "input": {"type": "text", "text": prompt},
                        "message_context": {
                            "original_message": incoming_message.to_dict(),
                            "source_topic": source_topic,
                            "in_reply_to": msg_id
                        }
                    })
                else:

                    raise
            
        except Exception as e:
            logger.error(f"Error generating automatic response: {e}")

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
    
    def _create_agent_tools(self) -> List[StructuredTool]:
        """
        Create LangChain tools for agent communication.
        
        These tools allow LangChain agents to interact with the MQTT network.
        
        Returns:
            List of StructuredTool objects
        """
        # Connection management tools
        connect_tool = StructuredTool.from_function(
            name="connect_to_mqtt_broker",
            func=lambda broker_url: self.connect_to_broker(broker_url),
            description="""
            Connect to an MQTT broker to enable communication with other agents.
            The broker_url should be in the format: mqtt://hostname:port
            Example: connect_to_mqtt_broker("mqtt://localhost:1883")
            """,
            args_schema=MQTTBrokerConnectionInput,
            return_direct=False,
        )
        
        disconnect_tool = StructuredTool.from_function(
            name="disconnect_from_mqtt_broker",
            func=lambda: self.disconnect_from_broker(),
            description="""
            Disconnect from the current MQTT broker and clean up resources.
            Use this when you're finished communicating with other agents.
            Example: disconnect_from_mqtt_broker()
            """,
            return_direct=False,
        )
        
        # Messaging tools
        send_message_tool = StructuredTool.from_function(
            name="send_mqtt_message",
            func=lambda message_content, message_type="query", receiver_id=None: 
                self.send_message(message_content, message_type, receiver_id),
            description="""
            Send a message to other agents via the current outbox topic.
            Specify the message content, type, and optional specific recipient.
            Example: send_mqtt_message("Hello, this is Agent X!", "query", "agent_y")
            """,
            args_schema=MQTTMessageInput,
            return_direct=True,  # Changed to True for better agent feedback
        )
        
        read_messages_tool = StructuredTool.from_function(
            name="read_mqtt_messages",
            func=lambda: self.read_messages(),
            description="""
            Read all messages received from subscribed topics.
            This will clear the message queue after reading.
            Example: read_mqtt_messages()
            """,
            return_direct=False,
        )
        
        # Topic management tools
        subscribe_tool = StructuredTool.from_function(
            name="subscribe_to_mqtt_topic",
            func=lambda topic_name: self._subscribe_to_topic(topic_name),
            description="""
            Subscribe to receive messages from a specific MQTT topic.
            Example: subscribe_to_mqtt_topic("agents/announcements")
            """,
            args_schema=MQTTTopicInput,
            return_direct=False,
        )
        
        unsubscribe_tool = StructuredTool.from_function(
            name="unsubscribe_from_mqtt_topic",
            func=lambda topic_name: self._unsubscribe_from_topic(topic_name),
            description="""
            Stop receiving messages from a specific MQTT topic.
            Cannot unsubscribe from your primary inbox topic.
            Example: unsubscribe_from_mqtt_topic("agents/announcements")
            """,
            args_schema=MQTTTopicInput,
            return_direct=False,
        )
        
        change_outbox_tool = StructuredTool.from_function(
            name="change_mqtt_outbox_topic",
            func=lambda topic_name: self._change_outbox_topic(topic_name),
            description="""
            Change the topic where your outgoing messages are published.
            Use this when you want to communicate with a specific agent or group.
            Example: change_mqtt_outbox_topic("agent_y/inbox")
            """,
            args_schema=MQTTTopicInput,
            return_direct=False,
        )
                
        return [
            connect_tool,
            disconnect_tool, 
            send_message_tool, 
            read_messages_tool, 
            subscribe_tool, 
            unsubscribe_tool, 
            change_outbox_tool
        ]

    def set_agent_executor(self, agent_executor) -> None:
        """
        Set the agent executor for automatic message processing.
        
        Args:
            agent_executor: LangChain agent executor to handle messages
        """
        self.agent_executor = agent_executor
        logger.info(f"[{self.agent_id}] Agent executor configured for automatic responses")

    def add_message_handler(self, handler_function: Callable) -> None:
        """
        Add a custom message handler function.
        
        Args:
            handler_function: Function to call when messages arrive
                              Should accept (message, topic) parameters
        """
        self.message_handlers.append(handler_function)
        logger.info(f"[{self.agent_id}] Added custom message handler")

    def get_available_tools(self) -> List[StructuredTool]:
        """
        Get all available communication tools for use with a LangChain agent.
        
        Returns:
            List of StructuredTool objects
        """
        return self.available_tools
        
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

# Example usage
if __name__ == "__main__":
    # Create an agent communication manager
    agent_comms = AgentCommunicationManager(agent_id="agent_a")
    
    # Connect to a broker
    status = agent_comms.connect_to_broker("mqtt://localhost:1883")
    print(status)
    
    # Subscribe to another agent's topic
    agent_comms._subscribe_to_topic("agent_b/inbox")
    
    # Change outbox topic to communicate with agent_b
    agent_comms._change_outbox_topic("agent_b/inbox")
    
    # Send a test message
    agent_comms.send_message("Hello from agent_a! How are you?")
    
    # Example of custom error handling for OpenAI message format issues 
    def safe_message_handler(message, topic):
        """Handle messages with additional OpenAI format error protection"""
        try:
            print(f"Received message: {message.content} on {topic}")
            
            # If using with LangChain + OpenAI, properly format the message
            if agent_comms.agent_executor:
                try:
                    # First try normal format
                    agent_comms.agent_executor.invoke({
                        "input": f"Respond to: {message.content}"
                    })
                except Exception as e:
                    if "expected an object, but got a string instead" in str(e):
                        # Try with properly formatted content for multimodal models
                        agent_comms.agent_executor.invoke({
                            "input": {"type": "text", "text": f"Respond to: {message.content}"}
                        })
                    else:
                        raise
        except Exception as e:
            print(f"Error in message handler: {e}")
        
    # Add our safe message handler
    agent_comms.add_message_handler(safe_message_handler)
    
    # Keep the example running to demonstrate
    try:
        print("Agent is running. Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent...")
    finally:
        agent_comms.disconnect_from_broker()