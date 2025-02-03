"""
Utility module for standardized hashing functions across AutoSynth.

Features:
- Stable content hashing
- Configurable hash length
- Support for different hash algorithms
"""

import hashlib
import json
from typing import Any, Union

def stable_hash(
    content: Union[str, dict, list],
    length: int = 32,
    algorithm: str = "md5"
) -> str:
    """
    Generate a stable hash for content.
    
    Args:
        content: Content to hash (string or JSON-serializable object)
        length: Length of hash to return (max 32 for md5, 64 for sha256)
        algorithm: Hash algorithm to use ('md5' or 'sha256')
        
    Returns:
        Stable hash string of specified length
        
    Raises:
        ValueError: If length is invalid or algorithm not supported
    """
    if algorithm not in ("md5", "sha256"):
        raise ValueError("Algorithm must be 'md5' or 'sha256'")
        
    max_length = 64 if algorithm == "sha256" else 32
    if not 1 <= length <= max_length:
        raise ValueError(f"Length must be between 1 and {max_length}")
    
    # Convert content to string if needed
    if isinstance(content, (dict, list)):
        content = json.dumps(content, sort_keys=True)
    elif not isinstance(content, str):
        content = str(content)
    
    # Generate hash
    hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()[:length]

def content_hash(content: str, length: int = 32) -> str:
    """
    Generate stable hash for document content.
    Uses MD5 for speed since this is for caching, not security.
    """
    return stable_hash(content, length=length)

def query_hash(query: str, length: int = 16) -> str:
    """
    Generate stable hash for search queries.
    Uses shorter length since collisions are less likely with queries.
    """
    return stable_hash(query, length=length)

def url_hash(url: str, length: int = 16) -> str:
    """
    Generate stable hash for URLs.
    Uses shorter length for more manageable filenames.
    """
    return stable_hash(url, length=length)

def fingerprint_hash(content: str, length: int = 32) -> str:
    """
    Generate stable hash for document fingerprinting.
    Uses full MD5 length for better collision resistance.
    """
    return stable_hash(content, length=length) 