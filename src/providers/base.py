from abc import ABC, abstractmethod
from typing import List, Dict

class BaseProvider(ABC):
    """
    Abstract base class for search providers.

    All providers must implement the results method with the following signature:

        def results(self, query: str, num_results: int = 10) -> List[Dict]:
            '''
            Execute a search for the given query and return a list of results,
            where each result is a dictionary containing keys such as "link", "title", and "snippet".
            '''

    Providers inheriting from ProviderMixin must ensure they call _acquire_rate_limit() before making API calls
    and use _handle_error() in exception blocks to standardize error handling.
    Note: ProviderMixin is defined in the module 'provider_mixin.py' and provides common functionality for rate limiting and error handling.
    """

    @abstractmethod
    def results(self, query: str, num_results: int = 10) -> List[Dict]:
        pass