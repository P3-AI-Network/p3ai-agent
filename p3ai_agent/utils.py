import base64
import hashlib
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend


def derive_private_key_from_seed(seed_phrase):
    """
    Derive private key from seed phrase
    
    Args:
        seed_phrase (str): Base64 encoded seed phrase
        
    Returns:
        bytes: Private key (32 bytes)
    """
    seed_bytes = base64.b64decode(seed_phrase)
    return hashlib.sha256(seed_bytes).digest()

def derive_public_key_from_private(private_key_bytes):
    """
    Derive public key from private key
    
    Args:
        private_key_bytes (bytes): Private key (32 bytes)
        
    Returns:
        bytes: Public key in uncompressed format
    """

    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, 'big'),
        ec.SECP256K1(),
        default_backend()
    )
    

    public_key = private_key.public_key()
    public_key_numbers = public_key.public_numbers()
    

    x_bytes = public_key_numbers.x.to_bytes(32, 'big')
    y_bytes = public_key_numbers.y.to_bytes(32, 'big')
    
    return b'\x04' + x_bytes + y_bytes

def extract_public_key_from_did(did_document):
    """
    Extract public key from DID document coordinates
    
    Args:
        did_document (dict): DID document containing credentialSubject with x,y coordinates
        
    Returns:
        bytes: Public key in uncompressed format
    """
    try:
        x = int(did_document['credentialSubject']['x'])
        y = int(did_document['credentialSubject']['y'])
        

        x_bytes = x.to_bytes(32, 'big')
        y_bytes = y.to_bytes(32, 'big')
        
        return b'\x04' + x_bytes + y_bytes
        
    except Exception as e:
        raise ValueError(f"Failed to extract public key from DID document: {e}")

def encrypt_message(message, identity_credential_connected_agent):
    """
    Encrypt message using ECIES (Elliptic Curve Integrated Encryption Scheme)
    
    Args:
        message (str): Plain text message to encrypt
        identity_credential_connected_agent (dict): Recipient's DID document for encryption
        
    Returns:
        dict: Encrypted message with metadata containing ephemeral_public_key, iv, encrypted_data, and algorithm
    """
    try:

        recipient_public_key_bytes = extract_public_key_from_did(identity_credential_connected_agent)
        
        ephemeral_private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
        ephemeral_public_key = ephemeral_private_key.public_key()
        
        recipient_x = int.from_bytes(recipient_public_key_bytes[1:33], 'big')
        recipient_y = int.from_bytes(recipient_public_key_bytes[33:65], 'big')
        
        try:
            recipient_ec_public_key = ec.EllipticCurvePublicNumbers(
                recipient_x, recipient_y, ec.SECP256K1()
            ).public_key(default_backend())
        except ValueError as e:
            coordinate_data = recipient_x.to_bytes(32, 'big') + recipient_y.to_bytes(32, 'big')
            derived_private_key_int = int.from_bytes(
                hashlib.sha256(coordinate_data).digest(), 'big'
            )
            
            secp256k1_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            derived_private_key_int = derived_private_key_int % secp256k1_order
            if derived_private_key_int == 0:
                derived_private_key_int = 1
            
            temp_private_key = ec.derive_private_key(
                derived_private_key_int, ec.SECP256K1(), default_backend()
            )
            recipient_ec_public_key = temp_private_key.public_key()
        
        shared_secret = ephemeral_private_key.exchange(
            ec.ECDH(), recipient_ec_public_key
        )
        
        encryption_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'encryption',
            backend=default_backend()
        ).derive(shared_secret)
        
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        message_bytes = message.encode('utf-8')
        padding_length = 16 - (len(message_bytes) % 16)
        padded_message = message_bytes + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_message) + encryptor.finalize()
        
        ephemeral_public_numbers = ephemeral_public_key.public_numbers()
        ephemeral_x = ephemeral_public_numbers.x.to_bytes(32, 'big')
        ephemeral_y = ephemeral_public_numbers.y.to_bytes(32, 'big')
        ephemeral_public_key_bytes = b'\x04' + ephemeral_x + ephemeral_y
        
        return {
            'ephemeral_public_key': base64.b64encode(ephemeral_public_key_bytes).decode(),
            'iv': base64.b64encode(iv).decode(),
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'algorithm': 'ECIES-AES256-CBC'
        }
        
    except Exception as e:
        raise ValueError(f"Encryption failed: {e}")

def decrypt_message(encrypted_message, secret_seed, identity_credential):
    """
    Decrypt message using recipient's seed phrase and identity credential
    
    Args:
        encrypted_message (dict): Encrypted message from encrypt_message() containing ephemeral_public_key, iv, and encrypted_data
        secret_seed (str): Base64 encoded seed phrase for private key derivation
        identity_credential (dict): Recipient's DID document for key validation and fallback derivation
        
    Returns:
        str: Decrypted plain text message
    """
    try:
        recipient_private_key_bytes = derive_private_key_from_seed(secret_seed)
        
        derived_public_key = derive_public_key_from_private(recipient_private_key_bytes)
        did_public_key = extract_public_key_from_did(identity_credential)
        
        if derived_public_key != did_public_key:
            x = int(identity_credential['credentialSubject']['x'])
            y = int(identity_credential['credentialSubject']['y'])
            coordinate_data = x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
            recipient_private_key_bytes = hashlib.sha256(coordinate_data).digest()
            
            secp256k1_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            private_key_int = int.from_bytes(recipient_private_key_bytes, 'big')
            private_key_int = private_key_int % secp256k1_order
            if private_key_int == 0:
                private_key_int = 1
            recipient_private_key_bytes = private_key_int.to_bytes(32, 'big')
        
        recipient_private_key = ec.derive_private_key(
            int.from_bytes(recipient_private_key_bytes, 'big'),
            ec.SECP256K1(),
            default_backend()
        )
        
        ephemeral_public_key_bytes = base64.b64decode(encrypted_message['ephemeral_public_key'])
        ephemeral_x = int.from_bytes(ephemeral_public_key_bytes[1:33], 'big')
        ephemeral_y = int.from_bytes(ephemeral_public_key_bytes[33:65], 'big')
        ephemeral_public_key = ec.EllipticCurvePublicNumbers(
            ephemeral_x, ephemeral_y, ec.SECP256K1()
        ).public_key(default_backend())
        
        shared_secret = recipient_private_key.exchange(ec.ECDH(), ephemeral_public_key)
        
        decryption_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'encryption',
            backend=default_backend()
        ).derive(shared_secret)
        
        iv = base64.b64decode(encrypted_message['iv'])
        encrypted_data = base64.b64decode(encrypted_message['encrypted_data'])
        
        cipher = Cipher(
            algorithms.AES(decryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        padded_message = decryptor.update(encrypted_data) + decryptor.finalize()
        
        padding_length = padded_message[-1]
        message_bytes = padded_message[:-padding_length]
        
        return message_bytes.decode('utf-8')
        
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")