"""
Optimized search module for AutoSynth with Google-first provider orchestration
and academic-aware ranking.
"""

import asyncio
import logging
import time
import os
import re
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse
from pathlib import Path
import json

from .providers import PROVIDER_REGISTRY
from .db import AutoSynthDB
from .utils.hashing import query_hash
from .validator import DocumentValidator, ValidationMetrics

logger = logging.getLogger(__name__)

# Provider configuration
ENABLED_PROVIDERS = os.getenv("ENABLED_PROVIDERS", "google").split(",")
DEFAULT_PROVIDER = "google"
FALLBACK_PROVIDERS = ["google"]  # Providers to try if primary fails

# Search configuration
MAX_RESULTS_PER_PROVIDER = 10
MAX_CACHE_AGE_HOURS = 24
DEFAULT_NUM_RESULTS = 10
POSITION_WEIGHT = 2.0  # Weight for original Google ranking
ACADEMIC_BOOST = 1.25  # Multiplier for academic sources
MIN_QUALITY_SCORE = 0.7  # Minimum quality score for results

# Cache configuration
CACHE_DIR = Path("~/.autosynth/cache").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Use validator's academic patterns
ACADEMIC_DOMAINS = DocumentValidator.ACADEMIC_DOMAINS
ACADEMIC_PATTERNS = DocumentValidator.ACADEMIC_PATTERNS

class Search:
    """
    Optimized search system with Google-first provider management and academic-aware ranking.
    Integrates with DocumentValidator for quality control.
    """

    def __init__(
        self,
        providers: Optional[Dict] = None,
        validator: Optional[DocumentValidator] = None,
        db: Optional["AutoSynthDB"] = None
    ):
        """
        Initialize the search system.

        Args:
            providers: Dict mapping provider names to provider instances.
            validator: Optional DocumentValidator instance for quality control.
            db: Optional AutoSynthDB instance for caching.
        """
        # Initialize only enabled providers
        provider_names = providers.keys() if providers else ENABLED_PROVIDERS
        self.providers = {
            name: PROVIDER_REGISTRY[name]()
            for name in provider_names
            if name in PROVIDER_REGISTRY and name in ENABLED_PROVIDERS
        }
        
        # Fallback to default provider if no enabled providers
        if not self.providers and "google" in PROVIDER_REGISTRY:
            self.providers = {"google": PROVIDER_REGISTRY["google"]()}
        
        # Initialize validator and database
        self.validator = validator or DocumentValidator()
        self.db = db if db is not None else AutoSynthDB(
            db_type="cache",
            base_path=CACHE_DIR
        )
        self.ranking_weights = {
            "position": 2.0,
            "academic_pattern": 0.25,
            "domain_multiplier": 1.25,
        }
        self.quality_threshold = MIN_QUALITY_SCORE

    async def _get_cached_results(
        self,
        query: str,
        provider: str
    ) -> Optional[List[Dict]]:
        """Get cached results if available and fresh."""
        hashed_query = query_hash(query)
        try:
            results = await self.db.execute("""
                SELECT results FROM search_cache 
                WHERE query_hash = ? AND provider = ?
                AND timestamp > datetime('now', '-24 hours')
            """, (hashed_query, provider))
            
            if results:
                return json.loads(results[0]["results"])
            return None
            
        except Exception as e:
            logger.error(f"Cache read failed: {str(e)}")
            return None

    async def _cache_results(
        self,
        query: str,
        provider: str,
        results: List[Dict]
    ):
        """Cache search results."""
        hashed_query = query_hash(query)
        try:
            await self.db.execute(
                """
                INSERT OR REPLACE INTO search_cache 
                (query_hash, provider, results)
                VALUES (?, ?, ?)
                """,
                (hashed_query, provider, json.dumps(results))
            )
        except Exception as e:
            logger.error(f"Cache write failed: {str(e)}")

    async def _search_provider(
        self,
        provider: Any,
        query: str,
        respect_original_ranking: bool = True
    ) -> List[Dict]:
        """
        Execute search for a single provider with caching and ranking preservation.
        
        Args:
            provider: Search provider instance
            query: Search query string
            respect_original_ranking: Whether to preserve original ranking information
            
        Returns:
            List of search results
        """
        provider_name = provider.__class__.__name__
        
        # Check cache first
        cached_results = await self._get_cached_results(query, provider_name)
        if cached_results is not None:
            # Restore original ranking if needed
            if respect_original_ranking:
                for i, result in enumerate(cached_results):
                    result["original_rank"] = i
            return cached_results
            
        try:
            results = await asyncio.to_thread(
                provider.results,
                query=query,
                num_results=MAX_RESULTS_PER_PROVIDER
            )
            
            # Add original ranking before caching if needed
            if respect_original_ranking:
                for i, result in enumerate(results):
                    result["original_rank"] = i
                    
            # Cache successful results
            await self._cache_results(query, provider_name, results)
            return results
                
        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {str(e)}")
            # Try fallback provider if primary fails
            if provider_name != DEFAULT_PROVIDER and DEFAULT_PROVIDER in self.providers:
                logger.info(f"Attempting fallback to {DEFAULT_PROVIDER}")
                return await self._search_provider(
                    self.providers[DEFAULT_PROVIDER],
                    query,
                    respect_original_ranking
                )
            return []

    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Enhanced ranking using validator's academic patterns and quality metrics.
        """
        query_terms = set(query.lower().split())
        
        for i, result in enumerate(results):
            # Base score from original position
            position_score = len(results) - i
            
            # Text matching score
            text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
            term_matches = sum(term in text for term in query_terms) / max(len(query_terms), 1)
            
            # Academic scoring using validator patterns
            academic_score = 0.0
            for pattern in ACADEMIC_PATTERNS.values():
                if re.search(pattern, text):
                    academic_score += self.ranking_weights["academic_pattern"]  # Each pattern adds configured weight
            academic_score = min(1.0, academic_score)
            
            # Domain scoring
            url = result.get("link", "").lower()
            domain_score = self.ranking_weights["domain_multiplier"] if any(
                domain.replace("*.", "") in url
                for domain in ACADEMIC_DOMAINS
            ) else 1.0
            
            # Quality metrics from validator
            quality_score = self._get_quality_score(result)
            
            # Combined score with quality weighting
            result["score"] = (
                position_score * self.ranking_weights["position"] +
                term_matches +
                academic_score
            ) * domain_score * quality_score
            
            # Store metrics for debugging/analysis
            result["_metrics"] = {
                "position_score": position_score,
                "term_matches": term_matches,
                "academic_score": academic_score,
                "domain_score": domain_score,
                "quality_score": quality_score
            }
            
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Clean up internal metrics
        for result in results:
            result.pop("score", None)
            result.pop("_metrics", None)
            
        return results

    def _get_quality_score(self, result: Dict) -> float:
        """Calculate quality score using validator metrics.
        
        TODO: Add tests for quality score computation and make thresholds configurable if necessary.
        """
        # Extract text for validation
        text = (
            result.get("title", "") + "\n" +
            result.get("snippet", "") + "\n" +
            result.get("description", "")
        )
        
        # Basic structural checks
        if not text.strip():
            return 0.5
            
        # Use validator's structure check
        metrics = self.validator._check_structure(text)
        
        # Combine scores
        return (
            metrics.structure_score * 0.4 +  # Structure quality
            (0.3 if metrics.content_type != "unknown" else 0.0) +  # Known content type
            0.3  # Base score
        )

    async def search(
        self,
        query: str,
        selected_providers: Optional[List[str]] = None,
        num_results: int = DEFAULT_NUM_RESULTS,
        respect_original_ranking: bool = True,
        min_quality_score: float = MIN_QUALITY_SCORE
    ) -> List[Dict]:
        """
        Perform search with integrated validation and quality control.
        
        Args:
            query: Search query string
            selected_providers: List of provider names to use
            num_results: Number of results to return
            respect_original_ranking: If True, maintains provider's ranking order
            min_quality_score: Minimum quality score threshold
            
        Returns:
            List of ranked and validated search results
        """
        providers_to_use = [
            self.providers[name] 
            for name in (selected_providers or self.providers.keys())
            if name in self.providers and name in ENABLED_PROVIDERS
        ]
        
        if not providers_to_use:
            logger.warning("No enabled providers available. Using default Google provider.")
            if "google" in self.providers:
                providers_to_use = [self.providers["google"]]
            else:
                return []

        all_results = []
        for provider in providers_to_use:
            try:
                results = await self._search_provider(provider, query, respect_original_ranking)
                
                # Filter by quality score
                results = [
                    r for r in results 
                    if self._get_quality_score(r) >= min_quality_score
                ]
                
                all_results.extend(results)
                
                if len(all_results) >= num_results:
                    break
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed: {str(e)}")
                continue

        if all_results:
            ranked_results = self._rank_results(all_results, query)
            return ranked_results[:num_results]
            
        return [] 