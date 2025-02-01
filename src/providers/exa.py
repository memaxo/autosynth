"""
Exa API provider implementation for AutoSynth using LangChain.

This provider wraps the Exa API client to fetch search results with advanced filtering capabilities.
It follows LangChain best practices:
  • Modular provider design via extending the BaseProvider interface
  • Comprehensive type hints and docstrings
  • Environment variable support for API keys
  • Configurable search parameters with domain and date filtering
  • Error handling and logging for reliability

Before using, ensure you have installed the langchain-exa package:
    pip install --upgrade langchain-exa
"""

import logging
import os
import asyncio
from typing import List, Dict, Optional

from .base import BaseProvider
from . import register_provider
from langchain_core.rate_limiters import InMemoryRateLimiter

# Import the Exa client from the official package
try:
    from exa_py import Exa
except ImportError:
    raise ImportError("Please install langchain-exa via: pip install --upgrade langchain-exa")

logger = logging.getLogger(__name__)


class ExaAPIWrapper(BaseProvider):
    """
    Provider wrapper for querying the Exa Search API with advanced filtering.

    Attributes:
        api_key (str): API key for Exa search. If not provided, loads from EXA_API_KEY env var.
        search_kwargs (dict): Optional configuration parameters for search queries.
    """

    def __init__(self, api_key: Optional[str] = None, search_kwargs: Optional[Dict] = None):
        """
        Initialize the Exa API wrapper.

        Args:
            api_key (Optional[str]): Your Exa API key. If not provided, loads from EXA_API_KEY env var.
            search_kwargs (Optional[Dict]): Additional parameters for search configuration.
        """
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A valid Exa API key is required. Set it via parameter or EXA_API_KEY environment variable."
            )
        self.search_kwargs = search_kwargs or {}

        # Initialize rate limiter
        self.rate_limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)

    async def results(
        self,
        query: str,
        num_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve search results from the Exa API with advanced filtering options (async).

        This method initializes an Exa client and performs a search with optional filters
        for domains, dates, and text content. Results are formatted into a standardized
        list of dictionaries containing "link", "title", and "snippet" keys.

        Args:
            query (str): The search query string.
            num_results (int): Number of results to retrieve (defaults to 10).
            include_domains (Optional[List[str]]): Restrict search to these domains.
            exclude_domains (Optional[List[str]]): Exclude these domains from search.
            start_published_date (Optional[str]): Include docs published after this date (YYYY-MM-DD).
            end_published_date (Optional[str]): Include docs published before this date (YYYY-MM-DD).
            include_text (Optional[List[str]]): Only include results with these phrases.
            exclude_text (Optional[List[str]]): Exclude results with these phrases.

        Returns:
            List[Dict]: A list of search result dictionaries.

        Raises:
            Exception: Any error during the API call will be logged and re-raised.
        """
        try:
            # Merge search parameters with filters
            params = self.search_kwargs.copy()
            params.update({
                "num_results": num_results,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "start_published_date": start_published_date,
                "end_published_date": end_published_date,
                "include_text": include_text,
                "exclude_text": exclude_text,
                "use_autoprompt": True,
                "text": True,
                "highlights": True
            })

            # Initialize the Exa client
            await self.rate_limiter.acquire()

            # Run Exa client in a thread
            def _run_exa_search():
                client = Exa(api_key=self.api_key)
                return client.search_and_contents(query, **params)

            response = await asyncio.to_thread(_run_exa_search)

            # Process and format the results
            results = []
            for item in response:
                # Extract common fields from response
                result = {
                    "link": item.get("url", ""),
                    "title": item.get("title", query),
                    "snippet": item.get("text", "")[:200] if item.get("text") else "",
                    "published_date": item.get("published_date"),
                    "author": item.get("author"),
                    "highlights": item.get("highlights", [])
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Exa API error for query '{query}': {e}")
            raise

# Register Exa provider
register_provider("exa", ExaAPIWrapper)

# Example usage:
# if __name__ == "__main__":
#     import os
#     logging.basicConfig(level=logging.INFO)
#     api_key = os.getenv("EXA_API_KEY", "your_exa_api_key_here")
#     exa_provider = ExaAPIWrapper(api_key=api_key)
#     search_results = exa_provider.results(
#         "LangChain API integration",
#         num_results=5,
#         include_domains=["blog.langchain.dev"],
#         start_published_date="2023-01-01"
#     )
#     for res in search_results:
#         print(f"{res['title']}: {res['link']}")