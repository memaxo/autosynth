"""
Document processor module for AutoSynth pipeline.

This module handles document validation, duplicate detection, and quality assessment
using a combination of structural analysis, semantic similarity, and LLM-based validation.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from model import PhiValidator
from milvus_store import MilvusStore
from ranking.embeddings import EmbeddingGenerator

# Constants
MIN_ENGLISH_RATIO = 0.5
PREVIEW_LENGTH = 1000
DEFAULT_BATCH_SIZE = 5
DEFAULT_MIN_QUALITY = 0.7

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Metrics for document validation"""
    structure_score: float = 0.0
    llm_score: float = 0.0
    similarity_score: Optional[float] = None
    content_type: str = "unknown"
    validation_time: float = 0.0

class ValidationResult(BaseModel):
    """LLM validation result"""
    is_valid: bool = Field(..., description="Whether document passed validation")
    reason: str = Field(..., description="Reason for validation result")
    content_type: str = Field(..., description="Detected content type")
    quality_score: float = Field(..., description="Quality score between 0 and 1")

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class DocumentValidator:
    """
    Document validator for the AutoSynth pipeline.
    
    Handles:
    - Structural validation
    - Content quality assessment
    - Duplicate detection
    - Batch processing
    """
    
    def __init__(
        self,
        max_tokens: int = 100,
        temperature: float = 0.1,
        cache_dir: Optional[Path] = None,
        similarity_threshold: float = 0.9
    ):
        """
        Initialize the document validator.
        
        Args:
            max_tokens: Maximum tokens for LLM validation
            temperature: Temperature for LLM validation
            cache_dir: Directory for caching validation results
            similarity_threshold: Threshold for duplicate detection (0-1)
        """
        self.llm = PhiValidator(max_tokens=max_tokens, temperature=temperature)
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        
        # Initialize stores and generators
        self.milvus_store = MilvusStore()
        self.milvus_store.create_index()
        self.embedding_generator = EmbeddingGenerator('sentence-transformers/all-MiniLM-L6-v2')
        
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
        
        return (
            validation_prompt 
            | self.llm 
            | StrOutputParser()
            | JsonOutputParser(pydantic_object=ValidationResult)
        )
        
    def _check_structure(self, content: str) -> ValidationMetrics:
        """
        Perform structural validation with enhanced quality scoring.
        
        Args:
            content: Document content to validate
            
        Returns:
            ValidationMetrics with structure assessment results
        """
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
            
    async def validate_document(
        self,
        doc: Document,
        topic: str
    ) -> Tuple[bool, Dict]:
        """
        Perform comprehensive document validation.
        
        Args:
            doc: Document to validate
            topic: Topic for relevance checking
            
        Returns:
            Tuple of (is_valid, metadata)
        """
        start_time = time.time()
        try:
            # Structural validation
            metrics = self._check_structure(doc.page_content)
            if metrics.structure_score == 0:
                return False, self._create_metadata(
                    doc, metrics,
                    reason=f"Failed structural validation: {metrics.content_type}"
                )

            # Duplicate detection
            embedding = await self.embedding_generator.embed_texts(doc.page_content)
            results = self.milvus_store.search(embedding, limit=3)
            
            if results[0]:
                metrics.similarity_score = float(results[0][0].distance)
                if metrics.similarity_score > self.similarity_threshold:
                    return False, self._create_metadata(
                        doc, metrics,
                        reason="Document is near-duplicate of existing content"
                    )
                
            # LLM validation
            validation = await self.validation_chain.ainvoke({
                "topic": topic,
                "content": doc.page_content[:PREVIEW_LENGTH],
                "content_length": len(doc.page_content)
            })
            
            metrics.llm_score = validation.quality_score
            if not validation.is_valid:
                return False, self._create_metadata(
                    doc, metrics,
                    reason=validation.reason
                )
                
            return True, self._create_metadata(
                doc, metrics,
                reason=validation.reason
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False, {
                "source": doc.metadata.get("source", ""),
                "reason": f"Validation error: {str(e)}",
                "quality_score": 0.0,
                "validated_at": time.time()
            }
        finally:
            metrics.validation_time = time.time() - start_time
            
    def _create_metadata(
        self,
        doc: Document,
        metrics: ValidationMetrics,
        reason: str
    ) -> Dict:
        """Create standardized metadata dictionary."""
        return {
            "source": doc.metadata.get("source", ""),
            "content_type": metrics.content_type,
            "validation_reason": reason,
            "quality_score": max(metrics.structure_score, metrics.llm_score),
            "structure_score": metrics.structure_score,
            "llm_score": metrics.llm_score,
            "similarity_score": metrics.similarity_score,
            "validation_time": metrics.validation_time,
            "validated_at": time.time()
        }
            
    async def validate_batch(
        self,
        docs: List[Document],
        topic: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_quality_score: float = DEFAULT_MIN_QUALITY
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Validate documents in batches with quality filtering.
        
        Args:
            docs: List of documents to validate
            topic: Topic for relevance checking
            batch_size: Size of batches for concurrent processing
            min_quality_score: Minimum quality score to accept (0-1)
            
        Returns:
            Tuple of (valid_docs, validation_results)
        """
        valid_docs = []
        validation_results = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            logger.info(f"Validating batch {i//batch_size + 1}")
            
            # Process batch concurrently
            tasks = [self.validate_document(doc, topic) for doc in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter and process results
            for doc, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Batch validation error: {str(result)}")
                    continue
                    
                is_valid, metadata = result
                validation_results.append(metadata)
                
                if is_valid and metadata["quality_score"] >= min_quality_score:
                    doc.metadata.update(metadata)
                    valid_docs.append(doc)
            
            logger.info(
                f"Batch complete: {len(valid_docs)}/{len(batch)} documents valid "
                f"(quality threshold: {min_quality_score})"
            )
            
        return valid_docs, validation_results 