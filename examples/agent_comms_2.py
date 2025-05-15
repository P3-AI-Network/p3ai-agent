from p3ai_agent.communication import AgentCommunicationManager
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
import time 

if __name__ == "__main__":
    
    agent_comms = AgentCommunicationManager(agent_id="agent_b")
    tools = agent_comms.get_available_tools()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an AI assistant that can interact with MQTT brokers."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    agent_comms.set_agent_executor(agent_executor)

    # Connect to a broker
    status = agent_comms.connect_to_broker("mqtt://localhost:1883")

    
    # Subscribe to another agent's topic
    agent_comms._subscribe_to_topic("agent_b/inbox")
    
    # Change outbox topic to communicate with agent_b
    agent_comms._change_outbox_topic("agent_a/inbox")
    
    # Send a test message
    # agent_comms.send_message("Hello from agent_a! How are you?")
    
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
    

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"Agent Response: {result['output']}")
