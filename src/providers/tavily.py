"""
Tavily API provider implementation for AutoSynth using LangChain.

This provider wraps LangChain's TavilySearchResults tool to fetch search results.
It follows LangChain best practices:
  • Modular provider design via extending the BaseProvider interface
  • Comprehensive type hints and docstrings
  • Environment variable support for API keys
  • Advanced search configuration options
  • Error handling and logging for reliability

Before using, ensure you have installed the required packages:
    pip install --upgrade "langchain-community>=0.2.11" tavily-python
"""

import logging
import os
import asyncio
from typing import List, Dict, Optional

from .base import BaseProvider
from .provider_mixin import ProviderMixin
from . import register_provider
from langchain_core.rate_limiters import InMemoryRateLimiter

# Import the TavilySearchResults tool from LangChain community
try:
    from langchain_community.tools import TavilySearchResults
except ImportError:
    raise ImportError(
        "Please install langchain-community and tavily-python via: "
        "pip install --upgrade 'langchain-community>=0.2.11' tavily-python"
    )

logger = logging.getLogger(__name__)


class TavilySearchAPIWrapper(ProviderMixin, BaseProvider):
    """
    Provider wrapper for accessing the Tavily Search API through LangChain.

    Attributes:
        api_key (str): API key for Tavily Search. If not provided, loads from TAVILY_API_KEY env var.
        search_kwargs (dict): Optional configuration parameters for search queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "advanced",
        include_answer: bool = True,
        include_raw_content: bool = True,
        include_images: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_kwargs: Optional[Dict] = None,
        rate_limiter: Optional[InMemoryRateLimiter] = None
    ):
        """
        Initialize the TavilySearchAPIWrapper.

        Args:
            api_key (Optional[str]): Your Tavily API key. If not provided, loads from TAVILY_API_KEY env var.
            search_depth (str): Search depth - "basic" or "advanced". Defaults to "advanced".
            include_answer (bool): Whether to include AI-generated answer. Defaults to True.
            include_raw_content (bool): Whether to include raw content. Defaults to True.
            include_images (bool): Whether to include image results. Defaults to False.
            include_domains (Optional[List[str]]): List of domains to include in search.
            exclude_domains (Optional[List[str]]): List of domains to exclude from search.
            search_kwargs (Optional[Dict]): Additional search parameters.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A valid Tavily API key is required. Set it via parameter or TAVILY_API_KEY environment variable."
            )

        # Initialize the TavilySearchResults tool with configuration
        self.tool = TavilySearchResults(
            api_key=self.api_key,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            max_results=10,  # Will be updated in results() method
            **search_kwargs or {}
        )

        # Initialize rate limiter
        self.rate_limiter = rate_limiter or InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)

    async def results(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Retrieve search results from the Tavily Search API (async).

        This method uses LangChain's TavilySearchResults tool to perform the search
        and formats results into a standardized list of dictionaries with keys:
        "link", "title", and "snippet".

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
            self.tool.max_results = num_results

            # Rate limiting
            await self._acquire_rate_limit()

            # Execute search with the tool in a thread
            response = await asyncio.to_thread(self.tool.invoke, {"query": query})

            # Extract results from the tool's response
            if hasattr(response, "artifact") and response.artifact:
                raw_results = response.artifact.get("results", [])
            else:
                # Handle string response case
                import json
                raw_results = json.loads(response.content) if isinstance(response.content, str) else response.content

            # Process and format the results
            results = []
            for item in raw_results:
                result = {
                    "link": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:200] if item.get("content") else "",
                    "answer": item.get("answer", ""),
                    "images": item.get("images", [])
                }
                results.append(result)

            return results

        except Exception as e:
            self._handle_error(e, query)

# Register Tavily provider
register_provider("tavily", TavilySearchAPIWrapper)

# Example usage:
# if __name__ == "__main__":
#     import os
#     logging.basicConfig(level=logging.INFO)
#     api_key = os.getenv("TAVILY_API_KEY", "your_tavily_api_key_here")
#     tavily_provider = TavilySearchAPIWrapper(
#         api_key=api_key,
#         search_depth="advanced",
#         include_answer=True,
#         include_raw_content=True,
#         include_images=True
#     )
#     search_results = tavily_provider.results(
#         "What happened at the last Wimbledon?",
#         num_results=5
#     )
#     for res in search_results:
#         print(f"{res['title']}: {res['link']}")
#         if res.get("answer"):
#             print(f"Answer: {res['answer']}\n")