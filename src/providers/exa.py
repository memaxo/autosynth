"""
Exa API provider implementation for AutoSynth using LangChain.

This provider wraps the Exa API client to fetch search results with advanced filtering capabilities.
It follows LangChain best practices:
  • Modular provider design via extending the BaseProvider interface
  • Comprehensive type hints and docstrings
  • Environment variable support for API keys
  • Configurable search parameters with domain and date filtering
  • Error handling and logging for reliability
  • Neural search and hybrid mode support
  • Connection pooling for better performance
  • Batch processing capabilities

Before using, ensure you have installed the langchain-exa package:
    pip install --upgrade langchain-exa
"""

import logging
import os
import asyncio
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from aiohttp import ClientSession, TCPConnector

from .base import BaseProvider
from .provider_mixin import ProviderMixin
from . import register_provider
from langchain_core.rate_limiters import InMemoryRateLimiter

# Import the Exa client from the official package
try:
    from exa_py import Exa
except ImportError:
    raise ImportError("Please install langchain-exa via: pip install --upgrade langchain-exa")

logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """Search parameters configuration."""
    num_results: int = 10
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    start_published_date: Optional[str] = None
    end_published_date: Optional[str] = None
    start_crawl_date: Optional[str] = None
    end_crawl_date: Optional[str] = None
    include_text: Optional[List[str]] = None
    exclude_text: Optional[List[str]] = None
    neural: bool = True
    hybrid: bool = True
    num_passages: int = 3
    max_words: int = 100
    use_autoprompt: bool = True
    category: Optional[str] = None
    search_type: str = "auto"
    contents: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and set default content options."""
        if self.contents is None:
            self.contents = {
                "text": True,
                "highlights": {
                    "numSentences": 1,
                    "highlightsPerUrl": 1
                },
                "summary": True,
                "livecrawl": "always",
                "livecrawlTimeout": 1000,
                "extras": {
                    "links": True,
                    "imageLinks": True
                }
            }

        # Validate category if provided
        valid_categories = {
            "company", "research paper", "news", "pdf", "github", 
            "tweet", "personal site", "linkedin profile", "financial report"
        }
        if self.category and self.category not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")

        # Validate search type
        valid_types = {"keyword", "neural", "auto"}
        if self.search_type not in valid_types:
            raise ValueError(f"Invalid search type. Must be one of: {valid_types}")

        # Validate date formats
        for date_field in [
            self.start_published_date, self.end_published_date,
            self.start_crawl_date, self.end_crawl_date
        ]:
            if date_field and not self._is_valid_iso_date(date_field):
                raise ValueError(
                    f"Invalid date format. Must be ISO 8601 format (e.g., 2023-01-01T00:00:00.000Z)"
                )

    @staticmethod
    def _is_valid_iso_date(date_str: str) -> bool:
        """Validate ISO 8601 date format."""
        try:
            from datetime import datetime
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False


class ExaAPIError(Exception):
    """Custom exception for Exa API errors."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class ExaAPIWrapper(ProviderMixin, BaseProvider):
    """
    Enhanced provider wrapper for querying the Exa Search API with advanced filtering.

    Attributes:
        api_key (str): API key for Exa search. If not provided, loads from EXA_API_KEY env var.
        search_kwargs (dict): Optional configuration parameters for search queries.
        _session (Optional[ClientSession]): Aiohttp session for connection pooling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_kwargs: Optional[Dict] = None,
        max_connections: int = 10,
        rate_limiter: Optional[InMemoryRateLimiter] = None
    ):
        """
        Initialize the Exa API wrapper.

        Args:
            api_key (Optional[str]): Your Exa API key. If not provided, loads from EXA_API_KEY env var.
            search_kwargs (Optional[Dict]): Additional parameters for search configuration.
            max_connections (int): Maximum number of concurrent connections.
        """
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A valid Exa API key is required. Set it via parameter or EXA_API_KEY environment variable."
            )
        self.search_kwargs = search_kwargs or {}
        self._session: Optional[ClientSession] = None
        self._max_connections = max_connections

        # Initialize rate limiter
        self.rate_limiter = rate_limiter or InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)

    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=TCPConnector(limit=self._max_connections)
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_search_params(self, params: SearchParams) -> Dict[str, Any]:
        """Build search parameters dictionary."""
        search_params = self.search_kwargs.copy()
        search_params.update({
            "query": None,  # Will be set in _execute_search
            "numResults": params.num_results,
            "includeDomains": params.include_domains,
            "excludeDomains": params.exclude_domains,
            "startPublishedDate": params.start_published_date,
            "endPublishedDate": params.end_published_date,
            "startCrawlDate": params.start_crawl_date,
            "endCrawlDate": params.end_crawl_date,
            "includeText": params.include_text,
            "excludeText": params.exclude_text,
            "type": params.search_type,
            "useAutoprompt": params.use_autoprompt,
            "category": params.category,
            "contents": params.contents
        })
        return search_params

    async def _execute_search(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> List[Dict]:
        """Execute search with error handling."""
        try:
            await self._acquire_rate_limit()
            session = await self._get_session()

            # Set query in params
            params["query"] = query

            async def _run_exa_search():
                client = Exa(api_key=self.api_key, session=session)
                return await client.search(params)

            response = await _run_exa_search()

            # Process and format the results
            results = []
            for item in response.get("results", []):
                result = {
                    "link": item.get("url", ""),
                    "title": item.get("title", query),
                    "snippet": item.get("text", "")[:params.get("contents", {}).get("max_words", 200)] if item.get("text") else "",
                    "published_date": item.get("publishedDate"),
                    "author": item.get("author"),
                    "highlights": item.get("highlights", []),
                    "highlight_scores": item.get("highlightScores", []),
                    "summary": item.get("summary", ""),
                    "score": item.get("score", 0.0),
                    "image": item.get("image"),
                    "favicon": item.get("favicon"),
                    "subpages": item.get("subpages", []),
                    "extras": item.get("extras", {})
                }
                results.append(result)

            return results

        except Exception as e:
            self._handle_error(e, query)

    async def results(
        self,
        query: str,
        num_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        category: Optional[str] = None,
        search_type: str = "auto",
        use_autoprompt: bool = True,
        contents: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Retrieve search results from the Exa API with advanced filtering options (async).

        This method initializes an Exa client and performs a search with optional filters
        for domains, dates, and text content. Results are formatted into a standardized
        list of dictionaries.

        Args:
            query (str): The search query string.
            num_results (int): Number of results to retrieve (defaults to 10).
            include_domains (Optional[List[str]]): Restrict search to these domains.
            exclude_domains (Optional[List[str]]): Exclude these domains from search.
            start_published_date (Optional[str]): Include docs published after this date (ISO 8601).
            end_published_date (Optional[str]): Include docs published before this date (ISO 8601).
            start_crawl_date (Optional[str]): Include docs crawled after this date (ISO 8601).
            end_crawl_date (Optional[str]): Include docs crawled before this date (ISO 8601).
            include_text (Optional[List[str]]): Only include results with these phrases.
            exclude_text (Optional[List[str]]): Exclude results with these phrases.
            category (Optional[str]): Data category to focus on (e.g., "research paper", "news").
            search_type (str): Type of search - "keyword", "neural", or "auto" (default).
            use_autoprompt (bool): Whether to use Exa's query enhancement (default True).
            contents (Optional[Dict]): Configuration for result contents and features.

        Returns:
            List[Dict]: A list of search result dictionaries with comprehensive metadata.

        Raises:
            ExaAPIError: If there's an error during the API call.
            ValueError: If invalid parameters are provided.
        """
        params = SearchParams(
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            start_crawl_date=start_crawl_date,
            end_crawl_date=end_crawl_date,
            include_text=include_text,
            exclude_text=exclude_text,
            category=category,
            search_type=search_type,
            use_autoprompt=use_autoprompt,
            contents=contents
        )
        search_params = self._build_search_params(params)
        return await self._execute_search(query, search_params)

    async def batch_results(
        self,
        queries: List[str],
        batch_size: int = 5,
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple queries in batches.

        Args:
            queries (List[str]): List of search queries to process.
            batch_size (int): Number of queries to process in parallel (defaults to 5).
            **kwargs: Additional search parameters passed to results().

        Returns:
            List[Dict]: Combined list of search results from all queries.

        Raises:
            ExaAPIError: If there's an error during any API call.
        """
        all_results = []
        try:
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                tasks = [self.results(q, **kwargs) for q in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in the batch
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch query {i+j}: {str(result)}")
                    else:
                        all_results.extend(result)

            return all_results

        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            logger.error(error_msg)
            raise ExaAPIError(error_msg, original_error=e)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Register Exa provider
register_provider("exa", ExaAPIWrapper)

# Example usage:
# async def main():
#     async with ExaAPIWrapper() as exa_provider:
#         # Single search
#         results = await exa_provider.results(
#             "LangChain API integration",
#             num_results=5,
#             include_domains=["blog.langchain.dev"],
#             neural=True,
#             hybrid=True
#         )
#         
#         # Batch search
#         queries = [
#             "LangChain integration",
#             "Vector databases",
#             "Neural search"
#         ]
#         batch_results = await exa_provider.batch_results(
#             queries,
#             batch_size=3,
#             num_results=5
#         )