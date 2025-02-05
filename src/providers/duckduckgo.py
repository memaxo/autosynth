"""
DuckDuckGo API provider implementation for AutoSynth using LangChain.

This provider wraps LangChain's DuckDuckGoSearchResults tool to fetch search results.
It follows LangChain best practices:
  • Modular provider design via extending the BaseProvider interface
  • Comprehensive type hints and docstrings
  • Configurable search parameters including region and time filters
  • Support for different output formats and search backends
  • Error handling and logging for reliability

Before using, ensure you have installed the required packages:
    pip install --upgrade duckduckgo-search langchain-community
"""

import logging
import asyncio
from typing import List, Dict, Optional, Literal

from .base import BaseProvider
from .provider_mixin import ProviderMixin
from . import register_provider
from langchain_core.rate_limiters import InMemoryRateLimiter

# Import the DuckDuckGo tools from LangChain community
try:
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper as ExternalDuckDuckGoSearchAPIWrapper
except ImportError:
    raise ImportError(
        "Please install required packages via: "
        "pip install --upgrade duckduckgo-search langchain-community"
    )

logger = logging.getLogger(__name__)


class DuckDuckGoProvider(ProviderMixin, BaseProvider):
    """
    Provider wrapper for accessing the DuckDuckGo Search API through LangChain.

    Attributes:
        region (str): Region code for search results (e.g., "us-en", "uk-en").
        time (str): Time filter for results ("d" for day, "w" for week, etc.).
        max_results (int): Maximum number of results to return.
        backend (str): Search backend to use ("text" or "news").
        output_format (str): Format for results ("list", "json", or "raw").
    """

    def __init__(
        self,
        region: str = "us-en",
        time: Optional[str] = None,
        backend: Literal["text", "news"] = "text",
        output_format: Literal["list", "json", "raw"] = "list",
        safesearch: Literal["on", "moderate", "off"] = "moderate",
        max_results: int = 10,
        rate_limiter: Optional[InMemoryRateLimiter] = None
    ):
        """
        Initialize the DuckDuckGoSearchAPIWrapper.

        Args:
            region (str): Region code for localized results. Defaults to "us-en".
            time (Optional[str]): Time filter ("d" for day, "w" for week, etc.).
            backend (str): Search type - "text" or "news". Defaults to "text".
            output_format (str): Result format - "list", "json", or "raw".
            safesearch (str): SafeSearch setting - "on", "moderate", or "off".
            max_results (int): Maximum number of results. Defaults to 10.
        """
        # Initialize the DuckDuckGo API wrapper with configuration
        self.api_wrapper = ExternalDuckDuckGoSearchAPIWrapper(
            region=region,
            time=time,
            max_results=max_results,
            safesearch=safesearch
        )

        # Initialize the search tool
        self.tool = DuckDuckGoSearchResults(
            api_wrapper=self.api_wrapper,
            backend=backend,
            output_format=output_format
        )

        # Initialize rate limiter
        self.rate_limiter = rate_limiter or InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)

    async def results(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Retrieve search results from the DuckDuckGo Search API (async).

        This method uses LangChain's DuckDuckGoSearchResults tool to perform the search
        and formats results into a standardized list of dictionaries with keys:
        "link", "title", and "snippet". For news results, also includes "date" and "source".

        Args:
            query (str): The search query string.
            num_results (int, optional): Number of desired results. Defaults to 10.

        Returns:
            List[Dict]: A list of search result dictionaries.

        Raises:
            Exception: Any error during the API call will be logged and re-raised.
        """
        try:
            # Update max_results for this query
            self.api_wrapper.max_results = num_results

            # Rate limiting
            await self._acquire_rate_limit()

            # Execute search with the tool in a thread
            response = await asyncio.to_thread(self.tool.invoke, query)

            # Process the results based on output format
            if isinstance(response, str):
                import json
                # Handle raw string response
                if self.tool.output_format == "json":
                    results_list = json.loads(response)
                else:
                    # Parse comma-separated string format
                    results_list = [
                        dict(item.split(": ", 1) for item in result.split(", "))
                        for result in response.split("\n")
                        if result.strip()
                    ]
            else:
                results_list = response

            # Process and format the results
            results = []
            for item in results_list:
                result = {
                    "link": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")
                }
                
                # Add news-specific fields if present
                if "date" in item:
                    result["date"] = item["date"]
                if "source" in item:
                    result["source"] = item["source"]
                    
                results.append(result)

            return results

        except Exception as e:
            self._handle_error(e, query)

# Register DuckDuckGo provider
register_provider("duckduckgo", DuckDuckGoProvider)

# Example usage:
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     ddg_provider = DuckDuckGoSearchAPIWrapper(
#         region="us-en",
#         time="d",  # Last 24 hours
#         backend="news",  # Use news backend
#         output_format="list",
#         max_results=5
#     )
#     search_results = ddg_provider.results(
#         "Latest tech news",
#         num_results=5
#     )
#     for res in search_results:
#         print(f"{res['title']}: {res['link']}")
#         if "date" in res:
#             print(f"Date: {res['date']}")
#         if "source" in res:
#             print(f"Source: {res['source']}\n")