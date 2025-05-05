import zmq
import asyncio
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
import didkit
import json
import os

class Agent:
    
    account = None
    context = None
    connect_to_agent = None
    server = None
    client = None
    did_document = None

    def __init__(self, port, connect_to_agent, private_key=None, load_did = False) -> None:
        self.context = zmq.Context()
        self.port = port
        self.connect_to_agent = connect_to_agent

        # Server uses REP socket
        self.server = self.context.socket(zmq.REP)

        # Client uses REQ socket
        self.client = self.context.socket(zmq.REQ)

        self.create_ether_account(private_key)

        if (load_did):
            self.load_ethereum_did()
        else:
            self.create_ethereum_did(self.account.key.hex())


    # Generate new Ethereum account
    def create_ether_account(self,private_key):
        
        if (private_key == None):
            account: Account = Account.create()
            self.account = account
            print(f"Agent Account created with Private Key {account.key.hex()}")
        
        else:
            account = Account.from_key(private_key)
            self.account = account

    

    def create_ethereum_did(self, private_key):

        address = self.account.address
        
        # Connect to Optimism network
        w3 = Web3(Web3.HTTPProvider('https://mainnet.optimism.io'))  # Or use testnet URL for testing
        
        # Create DID document
        did = f"did:ethr:optimism:{address}"
        
        # Create verification method
        verification_method = {
            "id": f"{did}#controller",
            "type": "EcdsaSecp256k1RecoveryMethod2020",
            "controller": did,
            "blockchainAccountId": f"eip155:10:{address}"  # 10 is Optimism's chain ID
        }
        
        # Create DID Document
        did_document = {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/suites/secp256k1recovery-2020/v2"
            ],
            "id": did,
            "verificationMethod": [verification_method],
            "authentication": [verification_method["id"]],
            "assertionMethod": [verification_method["id"]]
        }
        
        # Sign the DID document
        message = json.dumps(did_document)
        message_hash = encode_defunct(text=message)
        signed = w3.eth.account.sign_message(message_hash, private_key=private_key)

        # Store DID on file for future use
        self.write_store(json.dumps(did_document), "did_document.txt")

        return {
            "did": did,
            "did_document": did_document,
            "private_key": private_key,
            "address": address,
            "signature": signed.signature.hex()
        }
    

    def load_ethereum_did(self):
        did = self.read_store("did_document.txt")
        self.did_document = did

    def read_store(self, file_name: str):
        file_path = os.path.join(os.getcwd(), "agent_data" ,file_name)
        with open(file_path, 'r') as file:

            data = file.read().strip()
            if (data == ""):
                print("No Did Data found")
                return
            
            return json.loads(data)
        


    def write_store(self, content: str, file_name: str):
        file_path = os.path.join(os.getcwd(), "agent_data" ,file_name)

        with open(file_path, 'w') as file:
            file.write(content)
        
        print(f"Data stored Successfully on {file_path}")


    def log(message, level): 
        #  Create/User predefined agent logging system 
        print(message)

    async def server_run(self):
        """Server listens for incoming messages and sends responses."""
        self.server.bind(f"tcp://*:{self.port}")
        print(f"Agent running as server on port {self.port}")

        while True:
            message = await asyncio.to_thread(self.server.recv)
            print(f"Received message: {message.decode('utf-8')}")

            # Simulate processing time
            await asyncio.sleep(1)

            # Reply to client
            reply = "Server Reply: Message Received"
            self.server.send(reply.encode('utf-8'))

    async def client_run(self, target_port):
        """Client connects to another agent and sends a message."""
        print(f"Client connecting to port {target_port}")
        self.client.connect(f"tcp://localhost:{target_port}")

        while True:
            # Send a message
            message = "Hello from Client"
            self.client.send(message.encode('utf-8'))
            print(f"Sent: {message}")

            # Wait for the server's reply
            reply = await asyncio.to_thread(self.client.recv)
            print(f"Received reply: {reply.decode('utf-8')}")

            # Wait before sending the next message
            await asyncio.sleep(2)

    async def run(self):
        """Run both client and server concurrently."""
        await asyncio.gather(
            self.server_run(),
            self.client_run(5556)  # Assuming there’s another agent running on port 5556
        )