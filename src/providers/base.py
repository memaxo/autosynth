from abc import ABC, abstractmethod
from typing import List, Dict

class BaseProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    def results(self, query: str, num_results: int = 10) -> List[Dict]:
        pass