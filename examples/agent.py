from p3ai_agent.agent import AgentConfig, P3AIAgent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv
import json 
import threading
import time

load_dotenv()


if __name__ == "__main__":

    agent_config = AgentConfig(
        default_inbox_topic="agent_a/inbox",
        default_outbox_topic=None,
        auto_reconnect=True,
        message_history_limit=100,
        registry_url="http://localhost:3002",
        mqtt_broker_url="mqtt://localhost:1883",
        identity_credential_path = "/Users/swapnilshinde/Desktop/p3ai/p3ai-agent/examples/identity_credential.json"
    )
    
    p3_agent = P3AIAgent(agent_config=agent_config)


    agent_executor = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    p3_agent.set_agent_executor(agent_executor)
    
    agents = p3_agent.search_agents_by_capabilities(["nlp"])
    print(f"Agents discovered: {agents}")


    # p3_agent.load_did("examples/agent-did.json")
    # my_did = json.dumps(p3_agent.AGENT_DID)

    # is_verified = p3_agent.verify_agent_identity(my_did)



    # print(f"Agent verified: {is_verified}")

    # result = p3_agent.connect_to_broker("mqtt://localhost:1883")
    # p3_agent._subscribe_to_topic(f"{p3_agent.agent_id}/inbox")
    # p3_agent._change_outbox_topic(f"{agents[0]}/inbox")


    # def recieve_messages(p3_agent: P3AIAgent):

    #     while True:

    #         if len(p3_agent.received_messages): 
                
    #             for message in p3_agent.read_messages:
    #                 print("Recieved Message: ",message)


    #             p3_agent.received_messages = []

    #         time.sleep(1)

    # def ask_questions(p3_agent: P3AIAgent):

    #     while True:

    #         user_input = input("Question: ")
    #         result = agent_executor.invoke({"input": user_input})

    #         p3_agent.send_message(result, "query")

    # thread = threading.Thread(target=recieve_messages, args=[p3_agent])
    # thread2 = threading.Thread(target=ask_questions, args=[p3_agent])

    # thread.start()
    # thread2.start()

    # thread.join()
    # thread2.join()


        



