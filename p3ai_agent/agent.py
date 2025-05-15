from p3ai_agent.communication import AgentCommunicationManager
from p3ai_agent.identity import IdentityManager
from p3ai_agent.search import SearchAndDiscoveryManager

from pydantic import BaseModel
from typing import Optional, List

from langchain.tools import StructuredTool

class AgentConfig(BaseModel):
    agent_id: str
    default_inbox_topic: Optional[str] = None
    default_outbox_topic: Optional[str] = None
    auto_reconnect: bool = True
    message_history_limit: int = 100
    registry_url: str = "http://localhost:3002/sdk/search"
    mqtt_broker_url: str

class P3AIAgent(AgentCommunicationManager, SearchAndDiscoveryManager):

    def __init__(self, agent_config: AgentConfig): 
        
        AgentCommunicationManager.__init__(
            self,
            agent_id=agent_config.agent_id,
            default_inbox_topic=agent_config.default_inbox_topic,
            default_outbox_topic=agent_config.default_outbox_topic,
            auto_reconnect=agent_config.auto_reconnect,
            message_history_limit=agent_config.message_history_limit
        )

        AgentCommunicationManager.connect_to_broker(self,agent_config.mqtt_broker_url)

        SearchAndDiscoveryManager.__init__(
            self,
            registry_url=agent_config.registry_url
        )

        self.agent_executor = None
        self.agent_config = agent_config    

    def get_available_tools(self, additional_tools: List[StructuredTool] = []) -> List[StructuredTool]:
        """
        Get the available tools for the agent, including additional tools and those from the communication and search managers.
        """
        
        return ( 
            additional_tools + 
            AgentCommunicationManager.get_available_tools(self) + 
            SearchAndDiscoveryManager.get_available_tools(self)
        )

    def set_agent_executor(self, agent_executor) -> None:
        
        AgentCommunicationManager.set_agent_executor(self, agent_executor)
        SearchAndDiscoveryManager.set_agent_executor(self, agent_executor)
        self.agent_executor = agent_executor

    def run(self):
        
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
            result = self.agent_executor.invoke({"input": user_input})
            print(f"Agent Response: {result['output']}")
