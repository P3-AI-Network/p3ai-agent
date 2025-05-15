from p3ai_agent.agent import AgentConfig, P3AIAgent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent

if __name__ == "__main__":

    agent_config = AgentConfig(
        agent_id="agent_a",
        default_inbox_topic="agent_a/inbox",
        default_outbox_topic=None,
        auto_reconnect=True,
        message_history_limit=100,
        registry_url="http://localhost:3002/sdk/search",
        mqtt_broker_url="mqtt://localhost:1883"
    )
    
    p3_agent = P3AIAgent(agent_config=agent_config)
    tools = p3_agent.get_available_tools()

    for tool in tools:
        print(f"Tool: {tool.name}")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You're an AI assistant.
            You have list of tools which can help you in finding agents and commuinicating with them.
            Your approach for a answer which is complex and will require external agents to be involed, 
            you have to first use search and discovery tools and then use mqtt tools to connect to the broker and topic 
            and then send message to other agent saying "change outbox topic to <your_client_id>/inbox" and then start sending messages.
         
            *Note*: You can use as many tools as you want to communicate and collaborate and yoy should always do agent_discovery before connecting to any mqtt
        """),
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

    p3_agent.set_agent_executor(agent_executor)
    
    # Example of custom error handling for OpenAI message format issues 
    def safe_message_handler(message, topic):
        """Handle messages with additional OpenAI format error protection"""
        try:
            print(f"Received message: {message.content} on {topic}")
            
            # If using with LangChain + OpenAI, properly format the message
            if p3_agent.agent_executor:
                try:
                    # First try normal format
                    p3_agent.agent_executor.invoke({
                        "input": f"Respond to: {message.content}"
                    })
                except Exception as e:
                    if "expected an object, but got a string instead" in str(e):
                        # Try with properly formatted content for multimodal models
                        p3_agent.agent_executor.invoke({
                            "input": {"type": "text", "text": f"Respond to: {message.content}"}
                        })
                    else:
                        raise
        except Exception as e:
            print(f"Error in message handler: {e}")
        
    # Add our safe message handler
    p3_agent.add_message_handler(safe_message_handler)
    

    p3_agent.run()