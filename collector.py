"""
Collector module for AutoSynth - handles document collection and initial processing
"""

import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse
import aiohttp
import bs4
from langchain.document_loaders import (
    WebBaseLoader, BSHTMLLoader, PyPDFLoader,
    TextLoader, GitLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RateLimiter:
    """Rate limiting for domain-specific requests"""
    
    def __init__(self, rate: float, per: float = 60.0):
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make request"""
        async with self.lock:
            now = time.time()
            
            # Replenish tokens
            time_passed = now - self.last_update
            self.tokens = min(
                self.rate,
                self.tokens + time_passed * (self.rate / self.per)
            )
            self.last_update = now
            
            # Check if we can make a request
            if self.tokens >= 1:
                self.tokens -= 1
                return True
                
            # Calculate wait time
            wait_time = (1 - self.tokens) * (self.per / self.rate)
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_update = time.time()
            return True

class DocumentLoadError(Exception):
    """Raised when document loading fails"""
    pass

class Collector:
    """Handles document collection and initial processing"""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: float = 1.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.cache_dir = cache_dir
        self.default_rate_limit = rate_limit
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Track processed URLs
        self.processed_urls: Set[str] = set()
        
    async def setup(self):
        """Initialize HTTP session and cache directory"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc
        
    async def _check_rate_limit(self, url: str):
        """Check and wait for rate limit"""
        domain = self._get_domain(url)
        if domain not in self.rate_limiters:
            self.rate_limiters[domain] = RateLimiter(self.default_rate_limit)
            
        await self.rate_limiters[domain].acquire()
        
    async def _check_cache(self, url: str) -> Optional[List[Document]]:
        """Check if URL content is cached"""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{hash(url)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Cache read failed for {url}: {str(e)}")
                return None
                
        return None
        
    async def _cache_documents(self, url: str, docs: List[Document]):
        """Cache documents for URL"""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{hash(url)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(docs, f)
        except Exception as e:
            self.logger.warning(f"Cache write failed for {url}: {str(e)}")
            
    def _get_loader(self, url: str):
        """Get appropriate document loader based on URL"""
        if url.endswith('.pdf'):
            return PyPDFLoader(url)
        elif url.endswith('.txt'):
            return TextLoader(url)
        elif 'github.com' in url:
            return GitLoader(
                clone_url=url,
                branch="main",
                file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt"))
            )
        else:
            # Default to WebBaseLoader with BS4 config
            return WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("content", "article", "main", "post")
                    )
                )
            )
            
    async def _load_with_timeout(self, loader, timeout: int) -> List[Document]:
        """Load documents with timeout"""
        try:
            # Most LangChain loaders are synchronous, run in thread pool
            docs = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return docs
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Document loading timed out after {timeout}s")
        except Exception as e:
            raise DocumentLoadError(f"Failed to load document: {str(e)}")
            
    async def collect(
        self,
        url: str,
        max_tokens: int = 2000,
        timeout: int = 30,
        force_reload: bool = False
    ) -> List[Document]:
        """
        Collect and process documents from URL
        
        Args:
            url: URL to collect from
            max_tokens: Maximum tokens per document
            timeout: Timeout in seconds
            force_reload: Force reload even if cached
            
        Returns:
            List of processed documents
        """
        # Skip if already processed
        if url in self.processed_urls and not force_reload:
            self.logger.info(f"Skipping already processed URL: {url}")
            return []
            
        # Check rate limiting
        await self._check_rate_limit(url)
        
        # Check cache
        if not force_reload:
            if cached := await self._check_cache(url):
                self.logger.info(f"Using cached content for {url}")
                return cached
                
        # Get appropriate loader
        loader = self._get_loader(url)
        
        try:
            # Load documents
            docs = await self._load_with_timeout(loader, timeout)
            
            # Split into chunks
            splits = self.text_splitter.split_documents(docs)
            
            # Filter by token count
            filtered_splits = []
            for split in splits:
                # Simple token count approximation
                token_count = len(split.page_content.split())
                if token_count <= max_tokens:
                    filtered_splits.append(split)
                    
            # Update metadata
            for doc in filtered_splits:
                doc.metadata.update({
                    "source_url": url,
                    "collection_time": time.time(),
                    "token_count": len(doc.page_content.split())
                })
                
            # Cache results
            await self._cache_documents(url, filtered_splits)
            
            # Mark as processed
            self.processed_urls.add(url)
            
            self.logger.info(
                f"Collected {len(filtered_splits)} documents from {url}"
            )
            return filtered_splits
            
        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {str(e)}")
            raise
            
    async def collect_batch(
        self,
        urls: List[str],
        max_tokens: int = 2000,
        timeout: int = 30,
        max_concurrency: int = 5
    ) -> List[Document]:
        """
        Collect documents from multiple URLs concurrently
        
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
                    self.logger.error(f"Failed to collect from {url}: {str(e)}")
                    return []
                    
        # Collect from all URLs concurrently
        tasks = [collect_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Combine all documents
        all_docs = []
        for docs in results:
            all_docs.extend(docs)
            
        return all_docs 