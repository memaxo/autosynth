"""
Search module for AutoSynth that combines provider orchestration and result ranking.

Features:
- Multi-provider search orchestration
- Semantic ranking with embedding-based similarity
- Milvus vector store for result caching
- Integrated performance monitoring
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

from providers import PROVIDER_REGISTRY
from ranking.embeddings import EmbeddingGenerator
from ranking.scoring import calculate_final_score
from milvus_store import MilvusStore
from .monitor import Monitor

logger = logging.getLogger(__name__)


class Search:
    """
    Unified search system that orchestrates multiple search providers and ranks results
    using semantic similarity. Includes integrated monitoring and caching.
    """

    def __init__(
        self,
        providers: Optional[Dict] = None,
        concurrency: int = 10,
        monitor: Optional[Monitor] = None
    ):
        """
        Initialize the search system.

        Args:
            providers: Dict mapping provider names to provider instances.
                      If None, initializes all providers from registry.
            concurrency: Maximum number of concurrent search tasks.
            monitor: Monitor instance for metrics tracking.
        """
        self.providers = providers or {
            name: cls() for name, cls in PROVIDER_REGISTRY.items()
        }
        
        self.semaphore = asyncio.Semaphore(concurrency)
        self.monitor = monitor or Monitor()
        self.embedding_generator = EmbeddingGenerator()
        self.milvus_store = MilvusStore()
        self.milvus_store.create_index()

    async def _search_provider(self, provider: Any, query: str, num_results: int) -> List[Dict]:
        """
        Execute search for a single provider with metrics tracking.
        
        Args:
            provider: Provider instance to search with
            query: Search query string
            num_results: Number of results to request
            
        Returns:
            List of search results, empty list if provider fails
        """
        provider_name = provider.__class__.__name__
        start_time = time.time()
        
        try:
            async with self.semaphore:
                results = await asyncio.to_thread(
                    provider.results,
                    query=query,
                    num_results=num_results
                )
                
                duration = time.time() - start_time
                self.monitor.track_event(
                    category="search",
                    action="provider_search",
                    label=provider_name,
                    value={
                        "success": True,
                        "duration": duration,
                        "results_count": len(results)
                    }
                )
                return results
                
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.track_event(
                category="search",
                action="provider_error",
                label=provider_name,
                value={
                    "success": False,
                    "duration": duration,
                    "error": str(e)
                }
            )
            logger.error(f"Provider {provider_name} failed: {str(e)}")
            return []

    async def search(self, query: str, selected_providers: Optional[List[str]] = None, num_results: int = 10) -> List[Dict]:
        """
        Search across selected providers and rank results.
        
        Args:
            query: Search query string
            selected_providers: List of provider names to use. If None, uses all providers.
            num_results: Number of results to return
            
        Returns:
            List of ranked search results, sorted by relevance score
        """
        search_start = time.time()
        
        providers_to_use = [
            self.providers[name] for name in (selected_providers or self.providers.keys())
            if name in self.providers
        ]
        
        if not providers_to_use:
            return []

        # Execute searches in parallel
        tasks = [
            self._search_provider(provider, query, num_results * 2)
            for provider in providers_to_use
        ]
        
        all_results = []
        try:
            results = await asyncio.gather(*tasks)
            for provider_results in results:
                all_results.extend(provider_results)
                
        except Exception as e:
            self.monitor.track_event(
                category="search",
                action="search_error",
                label="gather_failed",
                value={"error": str(e)}
            )
            logger.error(f"Search failed: {str(e)}")
            return []

        self.monitor.track_event(
            category="search",
            action="search_complete",
            value={
                "duration": time.time() - search_start,
                "total_results": len(all_results),
                "provider_count": len(providers_to_use)
            }
        )

        if all_results:
            try:
                ranked_results = await self._rank_results(all_results, query)
                return ranked_results[:num_results]
            except Exception as e:
                self.monitor.track_event(
                    category="search",
                    action="ranking_error",
                    value={"error": str(e)}
                )
                logger.error(f"Ranking failed: {str(e)}")
                return all_results[:num_results]
        
        return []

    async def _rank_results(self, results: List[Dict], topic: str) -> List[Dict]:
        """
        Rank results using semantic similarity and other ranking factors.
        
        Args:
            results: List of search results to rank
            topic: Original search topic/query
            
        Returns:
            List of results sorted by calculated relevance score
        """
        if not results:
            return []

        rank_start = time.time()

        # Calculate semantic similarity scores
        texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in results]
        topic_emb = await self.embedding_generator.embed_texts([topic])
        text_embs = await self.embedding_generator.embed_texts(texts)
        similarities = (text_embs * topic_emb[0]).sum(axis=1)
        
        # Score and rank results
        topic_terms = set(topic.lower().split())
        
        for i, result in enumerate(results):
            # Combine multiple ranking signals
            result["semantic_score"] = float(similarities[i])
            result["domain_score"] = 90 if "github.com" in urlparse(result.get("link", "")).netloc else 50
            
            text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
            result["keyword_score"] = sum(term in text for term in topic_terms) / len(topic_terms) if topic_terms else 0
            
            result["score"] = calculate_final_score(
                semantic_score=result["semantic_score"],
                domain_score=result["domain_score"],
                keyword_score=result["keyword_score"],
                freshness=0.5
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        self.monitor.track_event(
            category="search",
            action="ranking_complete",
            value={
                "duration": time.time() - rank_start,
                "results_count": len(results)
            }
        )

        return results 