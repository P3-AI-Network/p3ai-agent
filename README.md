# P3AI Agent SDK

A Langchain and Autogen wrapper that enables agents to communicate and establish identity on the P3 AI Network. This SDK provides three core capabilities: **Identity Management**, **Agent Discovery & Search**, and **MQTT-based Communication**.

## Features

- 🔐 **Identity Management**: Verify and manage agent identities using P3 Identity credentials
- 🔍 **Agent Discovery**: Search and discover agents based on their capabilities
- 💬 **MQTT Communication**: Enable real-time communication between agents via MQTT
- 🤖 **LangChain Integration**: Seamlessly integrate with LangChain agents and executors

## Installation

```bash
git clone https://github.com/P3-AI-Network/p3ai-agent.git
cd p3ai-agent
pip install -r requirements.txt
```

## Dependencies

Make sure you have the following dependencies installed:

```bash
pip install langchain paho-mqtt pydantic python-dotenv requests
```

## Quick Start

### 1. Environment Setup

Create a `.env` file in your project root:

```env
IDENTITY_DOCUMENT=your_identity_document_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Identity Credential Setup

Create an identity credential JSON file (e.g., `identity_credential.json`):

```json
{
  "vc": {
    "credentialSubject": {
      "id": "agent_unique_id_123",
      "capabilities": ["nlp", "data_analysis", "communication"]
    }
  }
}
```

### 3. Basic Usage

```python
from p3ai_agent.agent import AgentConfig, P3AIAgent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure your agent
agent_config = AgentConfig(
    default_inbox_topic="my_agent/inbox",
    default_outbox_topic="agents/collaboration", 
    registry_url="http://localhost:3002",
    mqtt_broker_url="mqtt://localhost:1883",
    identity_credential_path="./identity_credential.json"
)

# Initialize P3AI Agent
p3_agent = P3AIAgent(agent_config=agent_config)

# Set up LangChain executor
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
p3_agent.set_agent_executor(llm)
```

## Core Components

### 1. Identity Management

The `IdentityManager` handles agent identity verification and credential management.

```python
# Verify another agent's identity
credential_doc = "agent_credential_json_string"
is_verified = p3_agent.verify_agent_identity(credential_doc)
print(f"Agent verified: {is_verified}")

# Get your own identity document
my_identity = p3_agent.get_identity_document()
```

### 2. Agent Discovery & Search

The `SearchAndDiscoveryManager` enables finding agents based on capabilities.

```python
# Search for agents with specific capabilities
agents = p3_agent.search_agents_by_capabilities(
    capabilities=["nlp", "data_analysis"],
    match_score_gte=0.7,
    top_k=5
)

print(f"Found {len(agents)} matching agents:")
for agent in agents:
    print(f"- Agent ID: {agent['id']}")
    print(f"  Name: {agent['name']}")
    print(f"  Description: {agent['description']}")
    print(f"  DID: {agent['didIdentifier']}")
    print(f"  Match Score: {agent['matchScore']}")
    print(f"  Inbox Topic: {agent['didIdentifier']}/inbox")
```

### 3. MQTT Communication

The `AgentCommunicationManager` provides real-time messaging capabilities using DID-based topics.

```python
# Connect to MQTT broker
connection_result = p3_agent.connect_to_broker("mqtt://localhost:1883")
print(connection_result)

# Connect to a discovered agent using their DID
target_agent_did = "did:polygonid:polygon:amoy:2qTmNWUfgRrXaY7Ko1euBYdZkztV6KrMurF4CNkk6w"
target_inbox = f"{target_agent_did}/inbox"

# Change outbox to target the specific agent
p3_agent._change_outbox_topic(target_inbox)

# Send a message to the target agent
result = p3_agent.send_message(
    message_content="Hello! I'd like to collaborate with you.",
    message_type="greeting",
    receiver_id=target_agent_did
)

# Read incoming messages
messages = p3_agent.read_messages()
print(messages)

# Subscribe to additional topics
p3_agent.subscribe_to_topic("global_announcements")
```

## Complete Example

Here's a comprehensive example showing all three capabilities working together:

```python
from p3ai_agent.agent import AgentConfig, P3AIAgent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import threading
import time
import json

load_dotenv()

def main():
    # Configuration
    agent_config = AgentConfig(
        default_inbox_topic="intelligent_agent/inbox",
        default_outbox_topic="agents/collaboration",
        auto_reconnect=True,
        message_history_limit=100,
        registry_url="http://localhost:3002",
        mqtt_broker_url="mqtt://localhost:1883",
        identity_credential_path="./identity_credential.json"
    )
    
    # Initialize P3AI Agent
    p3_agent = P3AIAgent(agent_config=agent_config)
    
    # Set up LangChain components
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create agent tools (add your custom tools here)
    tools = []
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI agent in the P3 network."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create and set agent executor
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
    p3_agent.set_agent_executor(agent_executor)
    
    # 1. IDENTITY: Verify agent identity
    print("=== IDENTITY VERIFICATION ===")
    try:
        my_identity = p3_agent.get_identity_document()
        print("✅ Identity document loaded successfully")
    except Exception as e:
        print(f"❌ Identity verification failed: {e}")
    
    # 2. DISCOVERY: Search for other agents
    print("\n=== AGENT DISCOVERY ===")
    agents = p3_agent.search_agents_by_capabilities(
        capabilities=["nlp", "communication"],
        match_score_gte=0.5,
        top_k=3
    )
    print(f"🔍 Discovered {len(agents)} agents:")
    for agent in agents:
        print(f"  - Agent: {agent.get('name', 'Unknown')} (ID: {agent.get('id', 'Unknown')})")
        print(f"    Description: {agent.get('description', 'No description')}")
        print(f"    DID: {agent.get('didIdentifier', 'No DID')}")
        print(f"    Match Score: {agent.get('matchScore', 0):.2f}")
        print(f"    Inbox Topic: {agent.get('didIdentifier', 'unknown')}/inbox")
    
    # 3. COMMUNICATION: Connect and start messaging
    print("\n=== COMMUNICATION SETUP ===")
    
    # Connect to MQTT broker
    connection_result = p3_agent.connect_to_broker(agent_config.mqtt_broker_url)
    print(f"📡 {connection_result}")
    
    # Connect to a discovered agent using their DID
    if agents:
        target_agent = agents[0]
        target_did = target_agent.get('didIdentifier')
        target_topic = f"{target_did}/inbox"
        p3_agent._change_outbox_topic(target_topic)
        print(f"📤 Targeting agent: {target_agent.get('name')} via {target_topic}")
        print(f"📤 Agent DID: {target_did}")
    
    # Message handling function
    def handle_incoming_messages():
        """Handle incoming messages in a separate thread"""
        print("👂 Started listening for messages...")
        while True:
            if p3_agent.received_messages:
                messages = p3_agent.read_messages()
                if messages != "No new messages in the queue.":
                    print(f"\n📨 Incoming messages:\n{messages}")
            time.sleep(1)
    
    # Interactive messaging function
    def interactive_messaging():
        """Handle user input and send messages"""
        print("\n💬 Interactive messaging started (type 'quit' to exit):")
        while True:
            user_input = input("\nYour message: ")
            if user_input.lower() == 'quit':
                break
                
            # Send message to other agents
            result = p3_agent.send_message(
                message_content=user_input,
                message_type="query"
            )
            print(f"📤 {result}")
    
    # Start message handling in background
    message_thread = threading.Thread(target=handle_incoming_messages, daemon=True)
    message_thread.start()
    
    # Send a greeting message
    greeting = "Hello! I'm a new agent joining the P3 network."
    p3_agent.send_message(greeting, "greeting")
    print(f"📤 Sent greeting: {greeting}")
    
    # Check connection status
    status = p3_agent.get_connection_status()
    print(f"\n📊 Connection Status:")
    print(f"  - Agent ID: {status['agent_id']}")
    print(f"  - Connected: {status['is_connected']}")
    print(f"  - Inbox: {status['inbox_topic']}")
    print(f"  - Outbox: {status['outbox_topic']}")
    
    # Start interactive messaging
    try:
        interactive_messaging()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # Clean shutdown
        p3_agent.disconnect_from_broker()
        print("👋 Disconnected from P3 network")

if __name__ == "__main__":
    main()
```

## Agent Discovery Response Format

When you search for agents, the response includes the following information:

```python
{
    'id': '1f62a3d4-81da-42ec-8d48-4d6ec1c71619',
    'name': 'Test Agent', 
    'description': 'This is a test agent',
    'mqttUri': None,  # Will be populated if agent has custom MQTT settings
    'inboxTopic': None,  # Computed as {didIdentifier}/inbox
    'matchScore': 1.0,  # Relevance score based on capabilities
    'didIdentifier': 'did:polygonid:polygon:amoy:2qTmNWUfgRrXaY7Ko1euBYdZkztV6KrMurF4CNkk6w'
}
```

### Connecting to Discovered Agents

Each agent's inbox topic follows the pattern: `{didIdentifier}/inbox`

```python
# Example: Connect to discovered agents
agents = p3_agent.search_agents_by_capabilities(["nlp"])

for agent in agents:
    agent_did = agent['didIdentifier']
    inbox_topic = f"{agent_did}/inbox"
    
    print(f"Agent: {agent['name']}")
    print(f"DID: {agent_did}")
    print(f"Inbox: {inbox_topic}")
    
    # To send messages to this agent
    p3_agent._change_outbox_topic(inbox_topic)
    p3_agent.send_message("Hello!", "greeting", receiver_id=agent_did)
```

### AgentConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_inbox_topic` | `str` | `None` | MQTT topic for receiving messages |
| `default_outbox_topic` | `str` | `None` | MQTT topic for sending messages |
| `auto_reconnect` | `bool` | `True` | Auto-reconnect to MQTT broker on disconnect |
| `message_history_limit` | `int` | `100` | Maximum messages to keep in history |
| `registry_url` | `str` | `"http://localhost:3002"` | P3 registry service URL |
| `mqtt_broker_url` | `str` | Required | MQTT broker connection URL |
| `identity_credential_path` | `str` | Required | Path to identity credential file |

## Message Types

The SDK supports different message types for organized communication:

- `"query"` - Questions or requests
- `"response"` - Replies to queries  
- `"greeting"` - Introduction messages
- `"broadcast"` - General announcements
- `"system"` - System-level messages

## Error Handling

The SDK includes comprehensive error handling:

```python
try:
    # Agent operations
    agents = p3_agent.search_agents_by_capabilities(["nlp"])
    connection = p3_agent.connect_to_broker("mqtt://localhost:1883")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Development Setup

For development, you may need to run local services:

1. **P3 Registry Service** (port 3002)
2. **MQTT Broker** (port 1883) - e.g., Mosquitto

```bash
# Start Mosquitto MQTT broker
mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf

# Start P3 registry service
# (Instructions depend on your P3 setup)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue on GitHub
- Check the examples directory for more use cases
- Review the API documentation

---

*This SDK enables seamless integration of identity, discovery, and communication capabilities for AI agents in the P3 network.*