"""
Document collection and processing module for AutoSynth.

This module handles:
- Document collection from various sources (web, PDF, text, GitHub)
- Content processing and chunking
- Rate limiting and caching
- Concurrent document loading
"""

import asyncio
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import aiohttp
from bs4 import BeautifulSoup
from langchain.document_loaders import (
    WebBaseLoader, BSHTMLLoader, PyPDFLoader,
    TextLoader, GitLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_core.rate_limiters import InMemoryRateLimiter
from simhash import Simhash

from .github import RepoCollector

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RATE_LIMIT = 1.0
DEFAULT_MAX_WORKERS = 5
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MAX_CONCURRENCY = 5

# Configure global rate limiter
GLOBAL_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=DEFAULT_RATE_LIMIT,
    check_every_n_seconds=0.1,
    max_bucket_size=5
)

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for collected documents."""
    source_url: str
    collection_time: float
    token_count: int
    fingerprint: int
    source_type: str = "web"
    additional_meta: Optional[Dict] = field(default=None)

class DocumentLoadError(Exception):
    """Raised when document loading fails."""
    pass

class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass

class Collector:
    """Document collection and processing manager."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_workers: int = DEFAULT_MAX_WORKERS,
        simhash_threshold: int = 3,  # Number of differing bits to consider similar
        session: Optional[aiohttp.ClientSession] = None,
        text_splitter: Optional['RecursiveCharacterTextSplitter'] = None
    ):
        """
        Initialize collector with configuration.
        
        Args:
            cache_dir: Directory for caching documents
            rate_limit: Requests per second limit
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            max_workers: Maximum concurrent workers
            simhash_threshold: Number of differing bits to consider documents similar
        """
        self.session = session
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.default_rate_limit = rate_limit
        self.simhash_threshold = simhash_threshold
        
        # Initialize text splitter with configuration
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Track processed URLs and fingerprints
        self.processed_urls: Set[str] = set()
        self.document_fingerprints: Set[int] = set()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def __aenter__(self) -> 'Collector':
        """Async context manager entry."""
        await self.setup()
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Async context manager exit."""
        await self.cleanup()
        
    async def setup(self):
        """Initialize HTTP session and cache directory."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
        
    async def _check_rate_limit(self):
        """
        Check and wait for rate limit.
        
        Raises:
            RateLimitError: If rate limit check fails
        """
        try:
            await GLOBAL_RATE_LIMITER.acquire()
        except Exception as e:
            raise RateLimitError(f"Rate limit check failed: {str(e)}")
            
    async def _check_cache(self, url: str) -> Optional[List[Document]]:
        """
        Check if URL content is cached.
        """
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{hash(url)}_repo.pkl"
        if cache_file.exists():
            try:
                return await asyncio.to_thread(self._sync_load_cache, cache_file)
            except Exception as e:
                logger.warning(f"Cache read failed for {url}: {e}")
                return None
        return None
        
    def _sync_load_cache(self, cache_file: Path) -> List[Document]:
        import pickle
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    async def _cache_documents(self, url: str, docs: List[Document]):
        """
        Cache documents for URL.
        """
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{hash(url)}_repo.pkl"
        try:
            await asyncio.to_thread(self._sync_save_cache, cache_file, docs)
        except Exception as e:
            logger.warning(f"Failed to write cache for {url}: {e}")
            
    def _sync_save_cache(self, cache_file: Path, docs: List[Document]):
        """Synchronously save documents to cache."""
        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump(docs, f)
            
    def _get_loader(self, url: str) -> Union[AsyncHtmlLoader, WebBaseLoader, PyPDFLoader, TextLoader, RepoCollector]:
        """
        Get appropriate document loader based on URL.
        
        Args:
            url: URL to get loader for
            
        Returns:
            Document loader instance
        """
        if url.endswith('.pdf'):
            return PyPDFLoader(url)
        elif url.endswith('.txt'):
            return TextLoader(url)
        elif 'github.com' in url and not url.endswith('.git'):
            return RepoCollector(
                github_token=os.getenv('GITHUB_TOKEN'),
                cache_dir=self.cache_dir
            )
        else:
            return AsyncHtmlLoader(web_path=url)
            
    async def _load_with_timeout(
        self,
        loader: Union[AsyncHtmlLoader, WebBaseLoader, PyPDFLoader, TextLoader, RepoCollector],
        timeout: int
    ) -> List[Document]:
        """
        Load documents with timeout.
        
        Args:
            loader: Document loader instance
            timeout: Timeout in seconds
            
        Returns:
            List of loaded documents
            
        Raises:
            DocumentLoadError: If loading fails
            TimeoutError: If loading times out
        """
        try:
            if hasattr(loader, "aload"):
                return await asyncio.wait_for(loader.aload(), timeout=timeout)
            else:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    loader.load
                )
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Document loading timed out after {timeout}s")
        except Exception as e:
            raise DocumentLoadError(f"Failed to load document: {str(e)}")
            
    def _is_similar_simhash(self, fingerprint: int) -> bool:
        """
        Check if a document's Simhash fingerprint is similar to any existing ones.
        Uses hamming distance for comparison.
        
        Args:
            fingerprint: Simhash fingerprint to check
            
        Returns:
            True if similar document exists, False otherwise
        """
        for existing in self.document_fingerprints:
            # Compare hamming distance of fingerprints
            if bin(fingerprint ^ existing).count('1') <= self.simhash_threshold:
                return True
        return False

    def _process_document(self, doc: Document, url: str) -> Optional[Document]:
        """
        Process and enrich document with metadata.
        Includes first-pass duplicate detection using Simhash.
        
        Args:
            doc: Document to process
            url: Source URL
            
        Returns:
            Processed document if unique, None if duplicate
        """
        # Compute simhash fingerprint
        fingerprint = Simhash(doc.page_content).value
        
        # Quick duplicate check using Simhash
        if self._is_similar_simhash(fingerprint):
            logger.debug(f"Simhash detected duplicate content from {url}")
            return None
            
        # Add fingerprint to tracked set
        self.document_fingerprints.add(fingerprint)
        
        # Create metadata
        metadata = DocumentMetadata(
            source_url=url,
            collection_time=time.time(),
            token_count=len(doc.page_content.split()),
            fingerprint=fingerprint,
            source_type="web",
            additional_meta=doc.metadata
        )
        
        # Update document
        doc.metadata.update(vars(metadata))
        return doc
            
    async def collect(
        self,
        url: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        force_reload: bool = False
    ) -> List[Document]:
        """
        Collect and process documents from URL.
        
        Args:
            url: URL to collect from
            max_tokens: Maximum tokens per document
            timeout: Timeout in seconds
            force_reload: Force reload even if cached
            
        Returns:
            List of processed documents
            
        Raises:
            DocumentLoadError: If collection fails
        """
        if url in self.processed_urls and not force_reload:
            logger.info(f"Skipping already processed URL: {url}")
            return []
            
        try:
            # Check rate limiting
            await self._check_rate_limit()
            
            # Check cache
            if not force_reload:
                if cached := await self._check_cache(url):
                    logger.info(f"Using cached content for {url}")
                    return cached
                    
            # Get appropriate loader
            loader = self._get_loader(url)
            
            # Load and process documents
            if isinstance(loader, RepoCollector):
                docs = loader.collect(url)
            else:
                docs = await self._load_with_timeout(loader, timeout)
            
            # Split into chunks asynchronously
            splits = await asyncio.to_thread(self.text_splitter.split_documents, docs)
            
            # Filter and process documents
            processed_docs = []
            for doc in splits:
                if len(doc.page_content.split()) <= max_tokens:
                    processed_doc = self._process_document(doc, url)
                    if processed_doc:
                        processed_docs.append(processed_doc)
                    
            # Cache results
            await self._cache_documents(url, processed_docs)
            
            # Mark as processed
            self.processed_urls.add(url)
            
            logger.info(f"Collected {len(processed_docs)} documents from {url}")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Failed to collect from {url}: {str(e)}")
            raise DocumentLoadError(f"Failed to collect from {url}: {str(e)}")
            
    async def collect_batch(
        self,
        urls: List[str],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    ) -> List[Document]:
        """
        Collect documents from multiple URLs concurrently with shared concurrency limit.
        
        Args:
            urls: List of URLs to collect from
            max_tokens: Maximum tokens per document
            timeout: Timeout in seconds
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of collected documents
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _collect_url(url: str) -> List[Document]:
            async with semaphore:
                try:
                    return await self.collect(url, max_tokens, timeout)
                except Exception as e:
                    logger.error(f"Failed to collect from {url}: {str(e)}")
                    return []

        tasks = [_collect_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs = []
        for r in results:
            if isinstance(r, list):
                all_docs.extend(r)
            else:
                logger.error(f"Batch collection error: {str(r)}")

        return all_docs

    @asynccontextmanager
    async def _rate_limited_session(self):
        """Context manager for rate-limited session."""
        try:
            await self._check_rate_limit()
            yield self.session
        finally:
            if self.session:
                await self._check_rate_limit()

    async def collect_with_session(
        self,
        url: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        force_reload: bool = False
    ) -> List[Document]:
        """Collect documents using existing session."""
        async with self._rate_limited_session():
            return await self.collect(url, max_tokens, timeout, force_reload) 