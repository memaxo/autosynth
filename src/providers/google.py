"""
Google Custom Search API provider implementation for AutoSynth using LangChain.

This provider wraps LangChain's GoogleSearchAPIWrapper to fetch search results with academic focus.
It follows LangChain best practices:
  • Modular provider design via extending the BaseProvider interface
  • Comprehensive type hints and docstrings
  • Environment variable support for API keys
  • Academic domain filtering and ranking
  • Error handling and logging for reliability
  • Caching for cost optimization
  • Rate limiting for API quotas

Before using, ensure you have installed the required package:
    pip install --upgrade langchain-google-community
"""

import logging
import os
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .base import BaseProvider
from .provider_mixin import ProviderMixin
from . import register_provider
from langchain_core.rate_limiters import InMemoryRateLimiter

# Import the Google Search wrapper from LangChain
try:
    from langchain_google_community import GoogleSearchAPIWrapper as ExternalGoogleSearchAPIWrapper
except ImportError:
    raise ImportError(
        "Please install langchain-google-community via: "
        "pip install --upgrade langchain-google-community"
    )

logger = logging.getLogger(__name__)

# Academic domains for filtering and boosting
ACADEMIC_DOMAINS = [
    ".edu",
    ".ac.",
    "scholar.google",
    "arxiv.org",
    "researchgate.net",
    "academia.edu",
    "sciencedirect.com",
    "springer.com",
    "ieee.org",
    "acm.org",
    "jstor.org",
    "nature.com",
    "science.org",
]

@dataclass
class SearchParams:
    """Search parameters configuration."""
    num_results: int = 10
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    academic_only: bool = True
    language: str = "en"
    safe_search: str = "off"
    file_type: Optional[str] = None  # e.g., "pdf" for academic papers

    def __post_init__(self):
        """Set up academic domains if academic_only is True."""
        if self.academic_only and not self.include_domains:
            self.include_domains = ACADEMIC_DOMAINS


class GoogleSearchError(Exception):
    """Custom exception for Google Search API errors."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class GoogleSearchAPIWrapper(ProviderMixin, BaseProvider):
    """
    Enhanced provider wrapper for Google Custom Search API with academic focus.

    Attributes:
        api_key (str): API key for Google Custom Search. If not provided, loads from GOOGLE_API_KEY env var.
        cse_id (str): Custom Search Engine ID. If not provided, loads from GOOGLE_CSE_ID env var.
        search_kwargs (dict): Optional configuration parameters for search queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        search_kwargs: Optional[Dict] = None,
        rate_limiter: Optional[InMemoryRateLimiter] = None
    ):
        """
        Initialize the Google Search API wrapper.

        Args:
            api_key (Optional[str]): Your Google API key. If not provided, loads from GOOGLE_API_KEY env var.
            cse_id (Optional[str]): Your Custom Search Engine ID. If not provided, loads from GOOGLE_CSE_ID env var.
            search_kwargs (Optional[Dict]): Additional parameters for search configuration.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        
        if not self.api_key:
            raise ValueError(
                "A valid Google API key is required. Set it via parameter or GOOGLE_API_KEY environment variable."
            )
        if not self.cse_id:
            raise ValueError(
                "A valid Custom Search Engine ID is required. Set it via parameter or GOOGLE_CSE_ID environment variable."
            )

        # Initialize the Google Search wrapper
        self.search_wrapper = ExternalGoogleSearchAPIWrapper(
            google_api_key=self.api_key,
            google_cse_id=self.cse_id,
            k=10  # Will be updated in results() method
        )

        self.search_kwargs = search_kwargs or {}

        # Initialize rate limiter (100 queries per day free tier)
        self.rate_limiter = rate_limiter or InMemoryRateLimiter(
            requests_per_second=0.1,  # Max ~8,640 requests per day
            max_bucket_size=5
        )

    def _build_search_params(self, params: SearchParams) -> Dict[str, Any]:
        """Build search parameters dictionary."""
        search_params = self.search_kwargs.copy()
        
        # Build site restriction string for included domains
        if params.include_domains:
            site_restrict = " OR ".join(f"site:{domain}" for domain in params.include_domains)
            search_params["siterestrict"] = site_restrict

        # Add other parameters
        search_params.update({
            "num": params.num_results,
            "lr": f"lang_{params.language}",
            "safe": params.safe_search
        })

        # Add file type restriction if specified
        if params.file_type:
            search_params["filetype"] = params.file_type

        return search_params

    def _filter_results(self, results: List[Dict], params: SearchParams) -> List[Dict]:
        """Filter and process search results."""
        filtered = []
        
        for item in results:
            # Skip excluded domains
            if params.exclude_domains and any(
                domain in item.get("link", "").lower()
                for domain in params.exclude_domains
            ):
                continue

            # Format result
            result = {
                "link": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "file_type": item.get("fileFormat", ""),
                "is_academic": any(
                    domain in item.get("link", "").lower()
                    for domain in ACADEMIC_DOMAINS
                )
            }
            filtered.append(result)

        return filtered

    async def results(
        self,
        query: str,
        num_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        academic_only: bool = True,
        language: str = "en",
        safe_search: str = "off",
        file_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve search results from Google Custom Search API with academic focus.

        This method uses LangChain's GoogleSearchAPIWrapper to perform the search
        and formats results into a standardized list of dictionaries.

        Args:
            query (str): The search query string.
            num_results (int): Number of results to retrieve (defaults to 10).
            include_domains (Optional[List[str]]): Restrict search to these domains.
            exclude_domains (Optional[List[str]]): Exclude these domains from search.
            academic_only (bool): Whether to restrict to academic domains (defaults to True).
            language (str): Language code for results (defaults to "en").
            safe_search (str): SafeSearch setting ("off", "medium", "high").
            file_type (Optional[str]): Restrict to specific file type (e.g., "pdf").

        Returns:
            List[Dict]: A list of search result dictionaries.

        Raises:
            GoogleSearchError: If there's an error during the API call.
        """
        try:
            # Update max results
            self.search_wrapper.k = num_results

            # Build search parameters
            params = SearchParams(
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                academic_only=academic_only,
                language=language,
                safe_search=safe_search,
                file_type=file_type
            )
            search_params = self._build_search_params(params)

            # Rate limiting
            await self._acquire_rate_limit()

            # Execute search in thread
            raw_results = await asyncio.to_thread(
                self.search_wrapper.results,
                query,
                **search_params
            )

            # Filter and process results
            results = self._filter_results(raw_results, params)
            
            return results

        except Exception as e:
            self._handle_error(e, query)


# Register Google Search provider
register_provider("google", GoogleSearchAPIWrapper)

# Example usage:
# async def main():
#     google_provider = GoogleSearchAPIWrapper()
#     results = await google_provider.results(
#         "machine learning research papers",
#         num_results=5,
#         academic_only=True,
#         file_type="pdf"
#     )
#     for res in results:
#         print(f"{res['title']}: {res['link']}")
#         print(f"Academic: {res['is_academic']}\n") 