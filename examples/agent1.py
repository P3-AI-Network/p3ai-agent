from p3ai_agent.agent import AgentConfig, P3AIAgent
from p3ai_agent.communication import MQTTMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
import os
from time import sleep

load_dotenv()



if __name__ == "__main__":

    # Create agent config


    """
    default_outbox_topic:
        <agent_id>/inbox is used to connect to other agents topic and communicate with it
    auto_reconnect:
        auto run the connection logic if disconnection happens
    message_history:
        store <limit> number of past messages for better context
    registry_url:
        P3 AI agent registry url
    mqtt_broker_url:
        default mqtt broker url on which you will be listening on
    identity_credential_path:
        file path of credential document of the agent downloaded from the P3 AI dashboard 
    secret_seed:
        Seed string of agent downloaded from the P3 AI dashboard
    """
    agent_config = AgentConfig(
        default_outbox_topic=None,
        auto_reconnect=True,
        message_history_limit=100,
        registry_url="https://registry.p3ai.network",
        mqtt_broker_url="mqtt://registry.p3ai.network:1883",
        identity_credential_path = "/Users/swapnilshinde/Desktop/p3ai/p3ai-agent/examples/identity_credential1.json",
        secret_seed = os.environ["AGENT1_SEED"]
    )


    # Init p3 agent sdk wrapper
    p3_agent = P3AIAgent(agent_config=agent_config)
    
    # Created a langchain agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    search_tool = TavilySearchResults(max_results=3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    system_prompt = """You are a helpful AI agent. Use search when the user asks anything about current events, facts, or the web."""
    agent_executor = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # Supports tool calling
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_prompt}
    )

    p3_agent.set_agent_executor(agent_executor)


    def message_handler(message: MQTTMessage, topic: str):
        agent_response = p3_agent.agent_executor.invoke({"input": message.content})
        agent_output = agent_response["output"]
        p3_agent.send_message(agent_output)

    p3_agent.add_message_handler(message_handler)


    # Main loop
    while True:
        message = input("Message (Exit for exit): ")

        if message == "Exit":
            break
        
        p3_agent.send_message(message)
    