from typing import Dict, Type
from .base import BaseProvider

PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {}

def register_provider(name: str, provider_cls: Type[BaseProvider]):
    """
    Register a provider class under a given name.
    This allows dynamic instantiation in search.py without hardcoding imports.
    """
    PROVIDER_REGISTRY[name] = provider_cls