"""
Document validation module for content quality and source verification.

Features:
- URL and domain validation
- Academic content detection
- Content quality assessment
- LLM-based validation
- Duplicate detection
"""

import asyncio
import logging
logger = logging.getLogger(__name__)
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
from collections import OrderedDict

from langchain.schema import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from .model import PhiValidator
from .db import AutoSynthDB
from .ranking.embeddings import EmbeddingGenerator
from .utils.hashing import fingerprint_hash

# Constants
MIN_ENGLISH_RATIO = 0.5
PREVIEW_LENGTH = 1000
DEFAULT_BATCH_SIZE = 5
DEFAULT_MIN_QUALITY = 0.7
MAX_RECENT_FINGERPRINTS = 1000

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    structure_score: float = 0.0
    llm_score: float = 0.0
    url_score: float = 0.0
    academic_score: float = 0.0
    domain_score: float = 0.0
    similarity_score: Optional[float] = None
    content_type: str = "unknown"
    validation_time: float = 0.0

class ValidationResult(BaseModel):
    """Structured validation result"""
    is_valid: bool = Field(..., description="Whether document passed validation")
    reason: str = Field(..., description="Reason for validation result")
    content_type: str = Field(..., description="Detected content type")
    quality_score: float = Field(..., description="Quality score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class FingerprintCache:
    """LRU cache for document fingerprints."""
    
    def __init__(self, max_size: int = MAX_RECENT_FINGERPRINTS):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def add(self, fingerprint: str) -> None:
        """Add fingerprint to cache."""
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[fingerprint] = time.time()
        
    def check_similar(self, fingerprint: str, threshold: int = 3) -> bool:
        """Check if fingerprint is similar to any in cache."""
        if not self.cache:
            self.misses += 1
            return False
            
        # For stable hashes, we compare character differences
        for existing in self.cache:
            diff = sum(1 for a, b in zip(fingerprint, existing) if a != b)
            if diff <= threshold:
                self.hits += 1
                self.cache.move_to_end(existing)
                return True
                
        self.misses += 1
        return False
        
    def get_stats(self) -> Dict[str, float]:
        """Get cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

class DocumentValidator:
    """
    Unified document validator combining rule-based and LLM validation.
    
    Features:
    - URL and domain validation
    - Academic content detection
    - Structural validation
    - LLM-based quality assessment
    - Duplicate detection
    - Batch processing
    """
    
    # Academic and research patterns
    ACADEMIC_PATTERNS = {
        "arxiv": r'arxiv\.org/abs/\d+\.\d+',
        "doi": r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b',
        "academic_keywords": r'\b(thesis|dissertation|journal|conference|proceedings|research|study)\b',
        "citations": r'\[\d+\]|\(\w+\s+et\s+al\.,\s+\d{4}\)'
    }
    
    # Allowed academic domains
    ACADEMIC_DOMAINS = [
        "arxiv.org", "github.com", "*.edu", "*.ac.*",
        "scholar.google.com", "semanticscholar.org"
    ]
    
    # URL patterns
    BLOCKED_URL_PATTERNS = [
        '/ads/', '/tracking/', '/analytics/', 
        '/sponsored/', '/advertisement/'
    ]
    
    PREFERRED_URL_PATTERNS = [
        '/paper/', '/article/', '/research/',
        '/publication/', '/doc/', '/pdf/'
    ]
    
    def __init__(
        self,
        llm: Optional[PhiValidator] = None,
        similarity_threshold: float = 0.95,
        db_type: str = "validation",
        fingerprint_threshold: int = 3,
        allowed_domains: Optional[List[str]] = None
    ):
        """Initialize the unified validator."""
        self.llm = llm or PhiValidator()
        self.similarity_threshold = similarity_threshold
        self.fingerprint_threshold = fingerprint_threshold
        self.allowed_domains = allowed_domains or self.ACADEMIC_DOMAINS
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.fingerprint_cache = FingerprintCache()
        self.vector_store = AutoSynthDB(
            db_type=db_type
        )
        
        # Performance tracking
        self.validation_times = {
            "structural": [],
            "url": [],
            "academic": [],
            "fingerprint": [],
            "vector": [],
            "llm": []
        }
        
        # Setup validation chain
        self.validation_chain = self._setup_validation_chain()
    
    def _setup_validation_chain(self) -> ChatPromptTemplate:
        """Setup enhanced validation chain with comprehensive criteria."""
        validation_prompt = ChatPromptTemplate(messages=[
            SystemMessage(content=(
                "You are a document validator specializing in content quality assessment. "
                "Evaluate the content against these criteria:\n"
                "1. Language Quality:\n"
                "   - Must be primarily in English (>80% English text)\n"
                "   - Well-formed sentences and proper grammar\n"
                "   - Professional or technical writing style\n"
                "2. Content Quality:\n"
                "   - Not gibberish or random characters\n"
                "   - Well-structured and coherent\n"
                "   - Contains meaningful technical information\n"
                "   - Appropriate level of detail\n"
                "3. Topic Relevance:\n"
                "   - Directly related to the given topic\n"
                "   - Contains substantive technical information\n"
                "   - Demonstrates domain expertise\n"
                "4. Content Type Detection:\n"
                "   - Code: Contains programming code or scripts\n"
                "   - Documentation: API docs, technical specs, etc.\n"
                "   - Tutorial: Step-by-step guides or examples\n"
                "   - Article: Technical articles or blog posts\n"
                "   - Reference: Reference materials or specifications"
            )),
            HumanMessagePromptTemplate.from_template(
                "Topic: {topic}\n"
                "Content Length: {content_length} chars\n"
                "Content Preview: {content}\n\n"
                "Evaluate this content's validity and relevance."
            )
        ])
        
        # Note: The bitwise OR (|) operator is used here to compose the validation chain,
        # as supported by the underlying LLM framework.
        return (
            validation_prompt 
            | self.llm 
            | StrOutputParser()
            | JsonOutputParser(pydantic_object=ValidationResult)
        )
    
    def _validate_url(self, doc: Document) -> Tuple[bool, float, str]:
        """Validate URL structure and domain."""
        url = doc.metadata.get("source_url", "").lower()
        if not url:
            return False, 0.0, "No URL provided"
            
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path
        
        # Check blocked patterns
        for pattern in self.BLOCKED_URL_PATTERNS:
            if pattern in path:
                return False, 0.0, f"URL contains blocked pattern: {pattern}"
        
        # Check domain
        domain_valid = any(
            domain.endswith(d.replace("*.", ".")) 
            for d in self.allowed_domains
        )
        
        # Check preferred patterns
        preferred = any(p in path for p in self.PREFERRED_URL_PATTERNS)
        url_score = 0.8 if preferred else 0.5
        
        if not domain_valid:
            return False, url_score, "Domain not in allowed list"
            
        return True, url_score, "URL validation passed"
    
    def _validate_academic_content(self, content: str) -> Tuple[bool, float, Dict]:
        """Check for academic characteristics."""
        matches = {
            key: bool(re.search(pattern, content.lower()))
            for key, pattern in self.ACADEMIC_PATTERNS.items()
        }
        
        score = sum(matches.values()) * 0.25  # 0.25 weight per pattern
        score = min(1.0, score)
        
        matched_patterns = [k for k, v in matches.items() if v]
        
        return bool(matched_patterns), score, {
            "pattern_matches": matches,
            "matched_patterns": matched_patterns
        }
    
    def _check_structure(self, content: str) -> ValidationMetrics:
        """Perform structural validation with enhanced quality scoring."""
        if not content.strip():
            return ValidationMetrics(content_type="empty")
            
        # Check English text ratio
        english_ratio = len(re.findall(r'[a-zA-Z\s]', content)) / len(content)
        if english_ratio < MIN_ENGLISH_RATIO:
            return ValidationMetrics(
                content_type="non_english",
                structure_score=english_ratio
            )
            
        metrics = ValidationMetrics()
        
        # Detect content type and calculate quality score
        if any(x in content for x in ['def ', 'class ', 'import ', 'function', '{', '}', ';']):
            metrics.content_type = "code"
            metrics.structure_score = sum([
                0.3,  # Base score
                0.1 if 'def ' in content else 0,
                0.1 if 'class ' in content else 0,
                0.1 if any(doc in content for doc in ['"""', "'''"]) else 0
            ])
            
        elif any(x in content for x in ['# ', '## ', '### ']):
            metrics.content_type = "documentation"
            metrics.structure_score = sum([
                0.3,  # Base score
                0.2 if len(content.split('\n')) > 10 else 0,
                0.2 if '```' in content else 0
            ])
            
        elif len(content.split('\n')) > 20:
            metrics.content_type = "article"
            metrics.structure_score = sum([
                0.3,  # Base score
                0.2 if len(content) > 1000 else 0,
                0.2 if re.search(r'step \d|^\d\.', content.lower()) else 0
            ])
            
        return metrics
    
    async def verify_quality(
        self,
        doc: Document,
        topic: str
    ) -> Tuple[bool, Dict]:
        """
        Verify document quality using multi-stage validation.
        
        Stages:
        1. URL validation
        2. Academic content detection
        3. Structural validation
        4. Duplicate detection
        5. LLM-based quality assessment
        
        Args:
            doc: Document to validate
            topic: Topic for relevance checking
            
        Returns:
            Tuple of (is_valid, metadata)
        """
        content = doc.page_content
        metrics = ValidationMetrics()
        
        # Stage 1: URL Validation
        start_time = time.time()
        url_valid, url_score, url_reason = self._validate_url(doc)
        metrics.url_score = url_score
        self.validation_times["url"].append(time.time() - start_time)
        
        if not url_valid:
            return False, self._create_metadata(doc, metrics, url_reason)
        
        # Stage 2: Academic Content
        start_time = time.time()
        academic_valid, academic_score, academic_meta = self._validate_academic_content(content)
        metrics.academic_score = academic_score
        self.validation_times["academic"].append(time.time() - start_time)
        
        # Stage 3: Structural Validation
        start_time = time.time()
        structural_metrics = self._check_structure(content)
        metrics.structure_score = structural_metrics.structure_score
        metrics.content_type = structural_metrics.content_type
        self.validation_times["structural"].append(time.time() - start_time)
        
        if metrics.structure_score < 0.3:
            return False, self._create_metadata(
                doc, metrics,
                "Failed structural validation"
            )
        
        # Stage 4: Duplicate Detection
        start_time = time.time()
        doc_fingerprint = fingerprint_hash(content)
        if self.fingerprint_cache.check_similar(doc_fingerprint, self.fingerprint_threshold):
            self.validation_times["fingerprint"].append(time.time() - start_time)
            return False, self._create_metadata(
                doc, metrics,
                "Duplicate content detected"
            )
            
        self.fingerprint_cache.add(doc_fingerprint)
        self.validation_times["fingerprint"].append(time.time() - start_time)
        
        # Stage 5: LLM Validation
        start_time = time.time()
        try:
            llm_result = await self.validation_chain.ainvoke({
                "topic": topic,
                "content_length": len(content),
                "content": content[:PREVIEW_LENGTH]
            })
            
            metrics.llm_score = llm_result.quality_score
            self.validation_times["llm"].append(time.time() - start_time)
            
            if not llm_result.is_valid:
                return False, self._create_metadata(
                    doc, metrics,
                    llm_result.reason
                )
                
        except Exception as e:
            logger.error(f"LLM validation failed: {str(e)}")
            self.validation_times["llm"].append(time.time() - start_time)
            return False, self._create_metadata(
                doc, metrics,
                f"LLM validation failed: {str(e)}"
            )
        
        # Calculate final quality score
        quality_score = (
            metrics.structure_score * 0.2 +
            metrics.llm_score * 0.4 +
            metrics.academic_score * 0.2 +
            metrics.url_score * 0.2
        )
        
        return quality_score >= DEFAULT_MIN_QUALITY, self._create_metadata(
            doc, metrics,
            "Validation successful"
        )
    
    def _create_metadata(
        self,
        doc: Document,
        metrics: ValidationMetrics,
        reason: str
    ) -> Dict:
        """Create comprehensive validation metadata."""
        return {
            "url": doc.metadata.get("source_url", ""),
            "content_type": metrics.content_type,
            "structure_score": metrics.structure_score,
            "llm_score": metrics.llm_score,
            "academic_score": metrics.academic_score,
            "url_score": metrics.url_score,
            "similarity_score": metrics.similarity_score,
            "validation_time": metrics.validation_time,
            "reason": reason
        }
    
    async def validate_batch(
        self,
        docs: List[Document],
        topic: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_quality_score: float = DEFAULT_MIN_QUALITY
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Validate a batch of documents efficiently.
        
        Args:
            docs: List of documents to validate
            topic: Topic for relevance checking
            batch_size: Size of batches for processing
            min_quality_score: Minimum quality score threshold
            
        Returns:
            Tuple of:
            - valid_docs: List[Document] of documents that passed validation.
            - validation_results: List[Dict] containing metadata for each document's validation outcome.
        """
        valid_docs = []
        results = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.verify_quality(doc, topic)
                for doc in batch
            ])
            
            for doc, (is_valid, metadata) in zip(batch, batch_results):
                results.append(metadata)
                if is_valid:
                    valid_docs.append(doc)
        
        return valid_docs, results
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get validation performance statistics."""
        stats = {
            "validation_times": {
                stage: {
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0
                }
                for stage, times in self.validation_times.items()
            },
            "fingerprint_cache": self.fingerprint_cache.get_stats()
        }
        return stats