"""
Brave API provider implementation for AutoSynth using LangChain.

This provider wraps LangChain's BraveSearch tool to fetch search results.
It follows LangChain best practices:
  • Modular provider design via extending the BaseProvider interface
  • Comprehensive type hints and docstrings
  • Environment variable support for API keys
  • Configurable search parameters
  • Error handling and logging for reliability

Before using, ensure you have installed the langchain-community package:
    pip install --upgrade langchain-community
"""

import logging
import os
import asyncio
from typing import List, Dict, Optional

from .base import BaseProvider
from . import register_provider
from langchain_core.rate_limiters import InMemoryRateLimiter

# Import the BraveSearch tool from LangChain community
try:
    from langchain_community.tools import BraveSearch
except ImportError:
    raise ImportError("Please install langchain-community via: pip install --upgrade langchain-community")

logger = logging.getLogger(__name__)


class BraveSearchAPIWrapper(BaseProvider):
    """
    Provider wrapper for querying the Brave Search API through LangChain.
    
    Attributes:
        api_key (str): API key for Brave Search. If not provided, loads from BRAVE_API_KEY env var.
        search_kwargs (dict): Optional configuration parameters for search queries.
    """

    def __init__(self, api_key: Optional[str] = None, search_kwargs: Optional[Dict] = None):
        """
        Initialize the BraveSearchAPIWrapper.

        Args:
            api_key (Optional[str]): Your Brave Search API key. If not provided, loads from BRAVE_API_KEY env var.
            search_kwargs (Optional[Dict]): Additional parameters for search configuration.
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A valid Brave Search API key is required. Set it via parameter or BRAVE_API_KEY environment variable."
            )
        self.search_kwargs = search_kwargs or {}
        
        # Initialize the BraveSearch tool
        self.tool = BraveSearch.from_api_key(
            api_key=self.api_key,
            search_kwargs=self.search_kwargs
        )

        # Initialize rate limiter for this provider
        self.rate_limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)

    async def results(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Retrieve search results from the Brave Search API (async).

        This method uses LangChain's BraveSearch tool to perform the search
        and formats results into a standardized list of dictionaries with
        keys: "link", "title", and "snippet".

        Args:
            query (str): Search query string.
            num_results (int, optional): Number of desired results. Defaults to 10.

        Returns:
            List[Dict]: A list of search result dictionaries.

        Raises:
            Exception: Any error during the API call will be logged and re-raised.
        """
        try:
            # Update search parameters with num_results
            self.tool.search_kwargs.update({"count": num_results})
            
            # Rate limiting
            await self.rate_limiter.acquire()

            # Execute the search query using the tool in a thread
            raw_results = await asyncio.to_thread(self.tool.run, query)

            # Parse the results - BraveSearch returns a string containing JSON
            import json
            if isinstance(raw_results, str):
                results_list = json.loads(raw_results)
            else:
                results_list = raw_results
                
            # Process and format the results
            results = []
            for item in results_list:
                result = {
                    "link": item.get("link", ""),
                    "title": item.get("title", query),
                    "snippet": item.get("snippet", "")
                }
                results.append(result)
                
            return results

        except Exception as e:
            logger.error(f"BraveSearch API error for query '{query}': {e}")
            raise

# Register Brave provider
register_provider("brave", BraveSearchAPIWrapper)

# Example usage:
# if __name__ == "__main__":
#     import os
#     logging.basicConfig(level=logging.INFO)
#     api_key = os.getenv("BRAVE_API_KEY", "your_brave_api_key_here")
#     brave_provider = BraveSearchAPIWrapper(api_key=api_key)
#     search_results = brave_provider.results(
#         "LangChain API integration",
#         num_results=5
#     )
#     for res in search_results:
#         print(f"{res['title']}: {res['link']}")