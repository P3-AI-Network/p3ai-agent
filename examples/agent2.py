from p3ai_agent.agent import AgentConfig, P3AIAgent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
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
        identity_credential_path = "/Users/swapnilshinde/Desktop/p3ai/p3ai-agent/examples/identity_credential2.json",
        secret_seed = os.environ["AGENT2_SEED"]
    )


    # Init p3 agent sdk wrapper
    p3_agent = P3AIAgent(agent_config=agent_config)
    
    # Created a langchain agent
    agent_executor = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    p3_agent.set_agent_executor(agent_executor)


    # Main loop
    while True:
        search_filter = input("Search Agent: ")
        agents = p3_agent.search_agents_by_capabilities([search_filter])

        print("Agents Found")
        for agent in agents:
            print(f"""
                DID: {agent["didIdentifier"]}
                Description: {agent["description"]}
                Match Score: {agent["matchScore"]}
            """)
            print("================")
        
        agent_select = input("Connect to agent DID: ")

        selected_agent = None
        for agent in agents:
            if agent["didIdentifier"] == agent_select:
                selected_agent = agent
        
        if not selected_agent:
            raise "Invalid did agent not found"
        
        p3_agent.connect_agent(selected_agent)

        print("Connected to agent")

        while True:
            message = input("Message (Exit for exit): ")

            if message == "Exit":
                break
            
            p3_agent.send_message(message)
        