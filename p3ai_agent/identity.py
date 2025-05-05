import os
import json
import requests
from typing import Dict, Any, List, Optional, Union, Callable
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.base import ConversationalAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory

class P3IdentitySDK:
    """
    SDK for P3 Identity services that provides decentralized identity
    verification capabilities for AI agents.
    """
    
    def __init__(self):
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
        self.SDK_API_URL = os.environ.get("P3_AGENT_REGISTRY_URL", "http://localhost:3002/sdk")
    
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
    
    def get_agent_did(self) -> str:
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


class P3IdentityOutputParser(AgentOutputParser):
    """
    Output parser that adds identity verification capability to LangChain agents.
    """
    
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


class P3IdentityAgent:
    """
    A wrapper around LangChain agents that adds P3 identity capabilities.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Tool] = None,
        memory: Optional[ConversationBufferMemory] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        verbose: bool = False,
        agent_prefix: str = "You are an AI assistant with a verified decentralized identity."
    ):
        """
        Initialize a P3 Identity-enabled agent.
        
        Args:
            llm (BaseLanguageModel): The language model to use
            tools (List[Tool], optional): List of tools the agent can use
            memory (ConversationBufferMemory, optional): Memory for the agent
            callback_manager (BaseCallbackManager, optional): Callback manager
            verbose (bool): Whether to display verbose output
            agent_prefix (str): Prefix for the agent prompt
        """
        # Initialize P3 Identity SDK
        self.identity_sdk = P3IdentitySDK()
        
        # Set up memory
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up tools
        self.tools = tools or []
        
        # Add identity verification tool
        self.tools.append(
            Tool(
                name="verify_identity",
                func=self._verify_identity,
                description="Verify the identity of another agent using their credential document"
            )
        )
        
        # Add other identity-related tools
        self.tools.append(
            Tool(
                name="get_my_did",
                func=self._get_my_did,
                description="Get your own DID (Decentralized Identifier)"
            )
        )
        
        # Define the prompt template with identity information
        try:
            agent_did = self.identity_sdk.get_agent_did()
        except ValueError:
            agent_did = "Not available"
        
        identity_suffix = f"""

            Your DID (Decentralized Identifier) is: {agent_did}

            You can verify the identity of other agents using the verify_identity tool.
            When you need to verify another agent, use the verify_identity tool with their credential document.

            TOOLS:
            ------
            You have access to the following tools:
            {{tools}}

            To use a tool, please use the following format:
            ```
            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of [{{tool_names}}]
            Action Input: the input to the tool
            Observation: the result of the action
            ```

            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
            ```
            Thought: Do I need to use a tool? No
            AI: [your response here]
            ```
        """
        
        # Create the prompt template
        prompt = ConversationalAgent.create_prompt(
            tools=self.tools,
            prefix=agent_prefix,
            suffix=identity_suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        
        # Create the agent
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt
        )
        
        # Create the output parser
        output_parser = P3IdentityOutputParser()
        
        # Create the agent
        agent = ConversationalAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in self.tools],
            output_parser=output_parser
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            callback_manager=callback_manager,
            verbose=verbose
        )
    
    def _verify_identity(self, credential_document: str) -> str:
        """
        Verify the identity of another agent using their credential document.
        
        Args:
            credential_document (str): The credential document to verify
            
        Returns:
            str: The verification result as a string
        """
        try:
            result = self.identity_sdk.verify_agent(credential_document)
            
            # Format the result as a helpful message
            if result.get("verified", False):
                agent_did = self._extract_did_from_credential(credential_document)
                return f"Identity verified successfully. Agent DID: {agent_did}"
            else:
                return "Identity verification failed. The credential document is not valid."
                
        except Exception as e:
            return f"Failed to verify identity: {str(e)}"
    
    def _get_my_did(self) -> str:
        """
        Get the agent's own DID.
        
        Returns:
            str: The agent's DID
        """
        try:
            return self.identity_sdk.get_agent_did()
        except ValueError:
            return "No DID available for this agent."
    
    def _extract_did_from_credential(self, credential_document: str) -> str:
        """
        Extract the DID from a credential document.
        
        Args:
            credential_document (str): The credential document
            
        Returns:
            str: The extracted DID or a message indicating it couldn't be extracted
        """
        try:
            cred_json = json.loads(credential_document)
            # Try to find the subject ID which is typically the DID
            if "vc" in cred_json and "credentialSubject" in cred_json["vc"]:
                return cred_json["vc"]["credentialSubject"].get("id", "Unknown DID")
            return "Could not extract DID from credential"
        except Exception:
            return "Could not extract DID from credential"
    
    def run(self, input_text: str) -> str:
        """
        Run the agent with the given input.
        
        Args:
            input_text (str): The input text from the user
            
        Returns:
            str: The agent's response
        """
        return self.agent_executor.run(input=input_text)


# Example usage
def create_p3_identity_agent(
    llm: BaseLanguageModel,
    additional_tools: List[Tool] = None,
    verbose: bool = False
) -> P3IdentityAgent:
    """
    Create a P3 Identity-enabled agent with the given language model and tools.
    
    Args:
        llm (BaseLanguageModel): The language model to use
        additional_tools (List[Tool], optional): Additional tools for the agent
        verbose (bool): Whether to display verbose output
        
    Returns:
        P3IdentityAgent: The configured agent
    """
    tools = additional_tools or []
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create and return the agent
    return P3IdentityAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        verbose=verbose
    )


# Example of how to use the P3 Identity Agent
if __name__ == "__main__":
    # Import LangChain's ChatOpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import Tool
    
    # Example function for a custom tool
    def get_current_weather(location):
        """Get the current weather in a given location"""
        return f"The weather in {location} is sunny and 75 degrees."
    
    # Create additional tools
    weather_tool = Tool(
        name="get_weather",
        func=get_current_weather,
        description="Get the current weather in a specific location."
    )
    
    # Initialize the language model
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4",  # or any other available model
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create the P3 identity agent
    agent = create_p3_identity_agent(
        llm=llm,
        additional_tools=[weather_tool],
        verbose=True
    )
    
    # Run the agent with a sample query
    response = agent.run("Hello! Can you tell me your DID and help me understand what P3 identity verification is?")
    print(response)