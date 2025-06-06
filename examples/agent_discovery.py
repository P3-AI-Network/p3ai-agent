from p3ai_agent.search import SearchAndDiscoveryManager
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
import time 
    

if __name__ == "__main__":
    agent_search = SearchAndDiscoveryManager()
    tools = agent_search.get_available_tools()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an AI assistant, you have to use the tools given for every user question. to search for approaiate capabilities needed, make capabilities less than 2"""),
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

    agent_search.set_agent_executor(agent_executor)


    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"Agent Response: {result['output']}")
