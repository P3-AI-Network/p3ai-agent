# Agent Discovery and Search Protocol Module for P3AI
import logging
import requests

from typing import List
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SearchAndDiscovery")

class DiscoverAgentsInput(BaseModel):
    """
    Input schema for the discover_agents function.
    This class defines the
    input parameters for the discover_agents function.
    """

    capabilities: List[str] = Field(
        default=[],
        description="List of capabilities to search for. Example: ['story-teller']"
    )

class SearchAndDiscoveryManager:
    """
    This class implements the search and discovery protocol for P3AI agents.
    It allows agents to discover each other and share information about their capabilities.
    """

    def __init__(self, registry_url: str = "http://localhost:3002/sdk/search"):

        self.agents = []
        self.registry_url = registry_url
        self.agent_executor = None


    def discover_agents(self, capabilities: List[str] = []):
        """
        Discover all registered agents in the system based on their capabilities.
        """

        logger.info("Discovering agents...")


        resp = requests.post(self.registry_url, json={"userProvidedCapabilities": capabilities})
        if resp.status_code == 201:
            agents = resp.json()
            logger.info(f"Discovered {len(agents)} agents.")
            return agents
        else:
            logger.error(f"Failed to discover agents: {resp.status_code} - {resp.text}")
            return []
        
    
    def get_available_tools(self) -> List[StructuredTool]:
        """
        Get the tools for agent discovery.  
        This method is used to retrieve the tools that can be used for agent discovery.
        Returns:
            List of StructuredTool objects
        """

        agent_discovery_tool = StructuredTool.from_function(
            func=lambda capabilities: self.discover_agents(capabilities),
            name="agent_discovery",
            description="""
                Discover agents based on their capabilities.
                capabilities: List of capabilities to search for.
                Returns a list of agents that match the capabilities.
                Example: discover_agents(["story-teller"])
                Returns: List of agents that match the capabilities.
            """,
            args_schema=DiscoverAgentsInput,
            return_direct=False
        )
        return [agent_discovery_tool]
    

    def set_agent_executor(self, agent_executor) -> None:
        """
        Set the agent executor for automatic message processing.
        
        Args:
            agent_executor: LangChain agent executor to handle messages
        """
        self.agent_executor = agent_executor
        logger.info("Agent executor configured for automatic responses")
    