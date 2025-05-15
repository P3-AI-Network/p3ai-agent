from p3ai_agent.identity import IdentityManager
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
import time 
    

if __name__ == "__main__":
    agent_identity = IdentityManager(sdk_url="http://localhost:3002/sdk/search")
    tools = agent_identity.get_available_tools()

    my_did = agent_identity.get_my_did()
    print(f"My DID: {my_did}")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an AI assistant, you have to use the tools given for every user question. You can verify any identity document by using the given tools."""),
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

    agent_identity.set_agent_executor(agent_executor)


    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"Agent Response: {result['output']}")
