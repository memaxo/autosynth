"""
Document processor module for AutoSynth pipeline.

This module handles document processing, chunking, and quality assessment
using the unified DocumentValidator.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple
from langchain.schema import Document
from .validator import DocumentValidator

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MIN_QUALITY = 0.7

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Base class for processing errors."""
    pass

class Processor:
    """
    Document processor for content cleaning, chunking, and validation.
    
    Features:
    - Content cleaning and normalization
    - Document chunking with overlap
    - Quality validation via DocumentValidator
    - Batch processing capabilities
    """
    
    def __init__(
        self,
        validator: Optional[DocumentValidator] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Initialize processor with configuration.
        
        Args:
            validator: DocumentValidator instance
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.validator = validator or DocumentValidator()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove common noise patterns
        content = re.sub(r'<!--.*?-->', '', content)
        content = re.sub(r'<script.*?</script>', '', content)
        content = re.sub(r'<style.*?</style>', '', content)
        
        return content.strip()
        
    def _chunk_document(self, doc: Document) -> List[Document]:
        """Split document into overlapping chunks."""
        content = doc.page_content
        chunks = []
        
        # Simple chunking by character count
        start = 0
        while start < len(content):
            end = start + self.chunk_size
            if end > len(content):
                end = len(content)
                
            chunk = content[start:end]
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_start": start,
                    "chunk_end": end
                }
            ))
            
            start = end - self.chunk_overlap
            if start >= len(content):
                break
                
        return chunks
        
    async def clean_content(
        self,
        doc: Document,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Document:
        """
        Clean document content and optionally chunk it.
        
        Args:
            doc: Document to clean
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            
        Returns:
            Cleaned document or list of chunked documents
        """
        # Save original chunk settings
        orig_size = self.chunk_size
        orig_overlap = self.chunk_overlap
        
        try:
            if chunk_size is not None:
                self.chunk_size = chunk_size
            if chunk_overlap is not None:
                self.chunk_overlap = chunk_overlap
                
            # Clean content
            clean_content = self._clean_content(doc.page_content)
            clean_doc = Document(
                page_content=clean_content,
                metadata=doc.metadata
            )
            
            # Chunk if size specified
            if chunk_size:
                return self._chunk_document(clean_doc)
                
            return clean_doc
            
        except Exception as e:
            logger.error(f"Error cleaning document: {str(e)}")
            raise ProcessingError(f"Failed to clean document: {str(e)}")
            
        finally:
            # Restore original settings
            self.chunk_size = orig_size
            self.chunk_overlap = orig_overlap
            
    async def _process_single_document(
        self,
        doc: Document,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Optional[Document]:
        """
        Process a single document by cleaning and chunking it.
        
        Returns:
            Cleaned document, a list of chunks, or None if an error occurs.
        """
        try:
            return await self.clean_content(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as e:
            logger.error(f"Error cleaning document: {e}")
            return None
            
    async def verify_quality(
        self,
        doc: Document,
        topic: str,
        min_quality_score: float = DEFAULT_MIN_QUALITY
    ) -> bool:
        """
        Verify document quality using DocumentValidator.
        
        Args:
            doc: Document to validate
            topic: Topic for relevance checking
            min_quality_score: Minimum quality threshold
            
        Returns:
            Whether document passed validation
        """
        is_valid, _ = await self.validator.verify_quality(doc, topic)
        return is_valid
        
    async def process_batch(
        self,
        docs: List[Document],
        topic: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_quality_score: float = DEFAULT_MIN_QUALITY,
        batch_size: int = 5
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Process and validate a batch of documents concurrently.
        
        Args:
            docs: Documents to process
            topic: Topic for validation
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            min_quality_score: Minimum quality threshold
            batch_size: Size of validation batches
            
        Returns:
            Tuple of (valid_docs, validation_results)
        """
        # Process documents concurrently using a helper function
        results = await asyncio.gather(
            *[self._process_single_document(doc, chunk_size, chunk_overlap) for doc in docs],
            return_exceptions=True
        )
        processed_docs = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error processing document: {res}")
            elif res is None:
                continue
            elif isinstance(res, list):
                processed_docs.extend(res)
            else:
                processed_docs.append(res)
                
        # Validate documents
        return await self.validator.validate_batch(
            processed_docs,
            topic,
            batch_size=batch_size,
            min_quality_score=min_quality_score
        )
        
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get validation performance statistics."""
        return self.validator.get_performance_stats() 