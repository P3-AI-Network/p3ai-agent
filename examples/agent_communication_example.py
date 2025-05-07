from p3ai_agent.communication import MQTTAgentWrapper
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from rich.console import Console

console = Console()

if __name__ == "__main__":

    
    mqtt_wrapper = MQTTAgentWrapper(client_id="agenta")
    
    tools = mqtt_wrapper.get_tools()
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an AI assistant that can interact with MQTT brokers.
        You can connect to brokers, send messages, and read received messages.
        Always report the results of your actions and maintain a clear understanding of the connection state.
        When multiple steps are required, execute them in sequence without asking for confirmation between steps."""),
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
    
    # Set the agent executor in the MQTT wrapper
    mqtt_wrapper.set_agent_executor(agent_executor)
    

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke({"input": user_input})
        console.print(f"Agent Response: {result['output']}", style="bold green")
