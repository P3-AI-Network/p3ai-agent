import json
from p3ai_agent.search import SearchAndDiscoveryManager
from p3ai_agent.identity import IdentityManager
from p3ai_agent.communication import AgentCommunicationManager
from pydantic import BaseModel
from typing import Optional


class AgentConfig(BaseModel):
    auto_reconnect: bool = True
    message_history_limit: int = 100
    registry_url: str = "http://localhost:3002/sdk/search"
    mqtt_broker_url: str
    identity_credential_path: str
    identity_credential: Optional[dict] = None
    default_inbox_topic: Optional[str] = None
    default_outbox_topic: Optional[str] = None

class P3AIAgent(SearchAndDiscoveryManager, IdentityManager, AgentCommunicationManager):
    
    identity_credential: Optional[dict] = None

    def __init__(self, agent_config: AgentConfig): 

        try:
            with open(agent_config.identity_credential_path, "r") as f:
                self.identity_credential = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Identity credential file not found at {agent_config.identity_credential_path}")
        
        IdentityManager.__init__(self,agent_config.registry_url)

        SearchAndDiscoveryManager.__init__(
            self,
            registry_url=agent_config.registry_url
        )

        AgentCommunicationManager.__init__(
            self,
            self.identity_credential["vc"]["credentialSubject"]["id"],
            default_inbox_topic=agent_config.default_inbox_topic,
            default_outbox_topic=agent_config.default_outbox_topic,
            auto_reconnect=True,
            message_history_limit=agent_config.message_history_limit
        )

        self.agent_executor = None
        self.agent_config = agent_config    

    def set_agent_executor(self, agent_executor):
        """Set the agent executor for the agent."""

        self.agent_executor = agent_executor