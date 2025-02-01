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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

import aiohttp
import bs4
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
    additional_meta: Dict = None

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
        max_workers: int = DEFAULT_MAX_WORKERS
    ):
        """
        Initialize collector with configuration.
        
        Args:
            cache_dir: Directory for caching documents
            rate_limit: Requests per second limit
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            max_workers: Maximum concurrent workers
        """
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = cache_dir
        self.default_rate_limit = rate_limit
        
        # Initialize text splitter with configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Track processed URLs and initialize thread pool
        self.processed_urls: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def setup(self):
        """Initialize HTTP session and cache directory."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        self.executor.shutdown(wait=True)
            
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
        
    async def _check_rate_limit(self, url: str):
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
        
        Args:
            url: URL to check cache for
            
        Returns:
            Cached documents if available, None otherwise
        """
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{hash(url)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed for {url}: {str(e)}")
                if cache_file.exists():
                    cache_file.unlink()  # Remove corrupted cache
                return None
                
        return None
        
    async def _cache_documents(self, url: str, docs: List[Document]):
        """
        Cache documents for URL.
        
        Args:
            url: Source URL
            docs: Documents to cache
        """
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{hash(url)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(docs, f)
        except Exception as e:
            logger.warning(f"Cache write failed for {url}: {str(e)}")
            if cache_file.exists():
                cache_file.unlink()
            
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
            
    def _process_document(self, doc: Document, url: str) -> Document:
        """
        Process and enrich document with metadata.
        
        Args:
            doc: Document to process
            url: Source URL
            
        Returns:
            Processed document
        """
        # Compute simhash fingerprint
        fingerprint = Simhash(doc.page_content).value
        
        # Create metadata
        metadata = DocumentMetadata(
            source_url=url,
            collection_time=time.time(),
            token_count=len(doc.page_content.split()),
            fingerprint=fingerprint,
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
            await self._check_rate_limit(url)
            
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
            
            # Split into chunks
            splits = self.text_splitter.split_documents(docs)
            
            # Filter and process documents
            processed_docs = []
            for doc in splits:
                if len(doc.page_content.split()) <= max_tokens:
                    processed_doc = self._process_document(doc, url)
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
        Collect documents from multiple URLs concurrently.
        
        Args:
            urls: List of URLs to collect from
            max_tokens: Maximum tokens per document
            timeout: Timeout in seconds
            max_concurrency: Maximum concurrent requests
            
        Returns:
            List of all collected documents
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def collect_with_semaphore(url: str) -> List[Document]:
            async with semaphore:
                try:
                    return await self.collect(url, max_tokens, timeout)
                except Exception as e:
                    logger.error(f"Failed to collect from {url}: {str(e)}")
                    return []
                    
        # Collect from all URLs concurrently
        tasks = [collect_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all documents
        all_docs = []
        for docs in results:
            if isinstance(docs, list):
                all_docs.extend(docs)
            else:
                logger.error(f"Batch collection error: {str(docs)}")
            
        return all_docs 