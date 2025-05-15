import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List, Union 
from pydantic import BaseModel, Field

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.tools import StructuredTool
from langchain.agents.conversational.base import ConversationalAgent
from langchain.schema import AgentAction, AgentFinish


class VerifyAgentTool(BaseModel):
    agent_did_document: str = Field(
        description="The credential document to verify",
        examples=["{\"id\": \"85f96d67-15de-11f0-938d-0242ac130005\", \"proofTypes\": [\"P3AI\"...."]
    )


class GetMyDIDDocumentTool(BaseModel):
    pass


class IdentityManager:
    """
    This class manages the identity verification process for P3AI agents.
    It interacts with the P3 Identity SDK to verify agent identities.
    """
    
    def __init__(self, sdk_url: str = None):
        """
        Initialize the P3 Identity SDK by loading environment variables
        and setting up necessary attributes.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get identity document from environment variables
        self.IDENTITY_DOCUMENT = os.environ.get("IDENTITY_DOCUMENT")
        
        # Get DID from environment variables
        self.AGENT_DID = os.environ.get("AGENT_DID")
        
        # Get SDK API endpoint from environment variables with a default fallback
        self.SDK_API_URL = os.environ.get("P3_AGENT_REGISTRY_URL", sdk_url)
                
                
        self.agent_executor = None


    
    def verify_agent(self, credential_document: str) -> Dict[str, Any]:
        """
        Verify an agent's identity credential document by calling the SDK API.
        
        Args:
            credential_document (str): The credential document to verify.
        
        Returns:
            Dict[str, Any]: The response from the verification API
        
        Raises:
            ValueError: If no credential document is provided
            RuntimeError: If the API call fails
        """
        # Validate that we have a credential document to verify
        if not credential_document:
            raise ValueError("No credential document provided for verification")
        
        try:
            # Prepare the request payload
            payload = {
                "credDocumentJson": credential_document
            }
            
            # Set up headers
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json"
            }
            
            # Make the API call
            response = requests.post(
                self.SDK_API_URL,
                headers=headers,
                json=payload
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            # Return the JSON response
            return response.json()
            
        except requests.RequestException as e:
            # Handle API request failures
            raise RuntimeError(f"Failed to verify identity: {str(e)}")
        except json.JSONDecodeError:
            # Handle invalid JSON responses
            raise RuntimeError("Received invalid response from verification service")
    
    def get_identity_document(self) -> str:
        """
        Get the identity document of the current agent.
        
        Returns:
            str: The identity document
            
        Raises:
            ValueError: If no identity document is available
        """
        if not self.IDENTITY_DOCUMENT:
            raise ValueError("No identity document available for this agent")
        
        return self.IDENTITY_DOCUMENT
    
    def get_my_did(self) -> str:
        """
        Get the DID (Decentralized Identifier) of the current agent.
        
        Returns:
            str: The agent's DID
            
        Raises:
            ValueError: If no DID is available
        """
        if not self.AGENT_DID:
            raise ValueError("No DID available for this agent")
        
        return self.AGENT_DID
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse the output of the language model to determine the next action.
        
        Args:
            text (str): The output from the language model
            
        Returns:
            Union[AgentAction, AgentFinish]: The next action or the final answer
        """
        # Check if the output indicates an identity verification request
        if "verify_identity" in text.lower():
            # Extract the credential document to verify
            try:
                # Simple extraction logic - could be enhanced based on actual format
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    credential_document = text[start_idx:end_idx]
                    return AgentAction(
                        tool="verify_identity",
                        tool_input=credential_document,
                        log=text
                    )
            except Exception:
                pass
        
        # Default to the standard ConversationalAgent parser
        return ConversationalAgent.output_parser.parse(text)


    def get_available_tools(self, additional_tools: List[StructuredTool] = []) -> List[StructuredTool]:
        """
        Get all available tools including P3 identity tools.
        
        Args:
            additional_tools (List[Tool], optional): Additional tools to include
            
        Returns:
            List[Tool]: All available tools
        """

        verify_agent_tool = StructuredTool.from_function(
            name="verify_agent",
            func=lambda agent_did_document: self.verify_agent(agent_did_document),
            description="""
            Verify the identity of another agent using their credential document.
            The credential document should be in JSON format.
            example:  "{\n      \"id\": \"85f96d67-15de-11f0-938d-0242ac130005\",\n      \"proofTypes\": [\n....
            """,
            args_schema=VerifyAgentTool,
            return_direct=False,
        )

        get_my_did_tool = StructuredTool.from_function(
            name="get_my_did",
            func=lambda : self.get_my_did(),
            description="""
            get my did document.
            """,
            args_schema=GetMyDIDDocumentTool,
            return_direct=False,
        )
        
        return additional_tools + [verify_agent_tool, get_my_did_tool]
    
    def set_agent_executor(self, agent_executor: AgentExecutor) -> None:
        """
        Set the agent executor for automatic message processing.
        
        Args:
            agent_executor (AgentExecutor): The agent executor to set
        """
        self.agent_executor = agent_executor