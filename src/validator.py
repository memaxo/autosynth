"""
Document validation module for content quality and source verification.

Features:
- URL and domain validation
- Academic content detection
- Content quality assessment
- Configurable validation pipeline
"""

import asyncio
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse
import validators
import trafilatura
import logging
from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Structured validation result with detailed information."""
    is_valid: bool
    score: float  # 0-1 quality score
    reason: str
    metadata: Dict[str, Any]
    validator_name: str

class BaseValidator:
    """Base class for all validators with common functionality."""
    
    async def validate(self, doc: Document) -> ValidationResult:
        """
        Validate document and return structured result.
        
        Args:
            doc: Document to validate
            
        Returns:
            ValidationResult containing validation details
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _get_url_from_doc(self, doc: Document) -> str:
        """Helper to safely extract URL from document metadata."""
        return doc.metadata.get("source_url", "").lower()

class DomainValidator(BaseValidator):
    """Validates document sources against a list of allowed domains."""
    
    ACADEMIC_DOMAINS = [
        "arxiv.org", "github.com", "*.edu", "*.ac.*",
        "scholar.google.com", "semanticscholar.org"
    ]
    
    def __init__(self, allowed_domains: Optional[List[str]] = None):
        """
        Initialize with optional custom domain list.
        
        Args:
            allowed_domains: List of allowed domain patterns
        """
        self.allowed_domains = allowed_domains or self.ACADEMIC_DOMAINS
    
    async def validate(self, doc: Document) -> ValidationResult:
        url = self._get_url_from_doc(doc)
        if not url:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                reason="No URL provided",
                metadata={},
                validator_name=self.name
            )
            
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        is_allowed = any(
            domain.endswith(d.replace("*.", ".")) 
            for d in self.allowed_domains
        )
        
        return ValidationResult(
            is_valid=is_allowed,
            score=1.0 if is_allowed else 0.0,
            reason=f"Domain {'is' if is_allowed else 'is not'} in allowed list",
            metadata={"domain": domain, "url": url},
            validator_name=self.name
        )

class AcademicValidator(BaseValidator):
    """Validates content for academic characteristics."""
    
    PATTERNS = {
        "arxiv": r'arxiv\.org/abs/\d+\.\d+',
        "doi": r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b',
        "academic_keywords": r'\b(thesis|dissertation|journal|conference|proceedings|research|study)\b',
        "citations": r'\[\d+\]|\(\w+\s+et\s+al\.,\s+\d{4}\)'
    }
    
    PATTERN_WEIGHT = 0.25  # Score weight per pattern match
    
    async def validate(self, doc: Document) -> ValidationResult:
        content = doc.page_content.lower()
        matches = {
            key: bool(re.search(pattern, content))
            for key, pattern in self.PATTERNS.items()
        }
        
        score = sum(matches.values()) * self.PATTERN_WEIGHT
        score = min(1.0, score)  # Cap score at 1.0
        
        matched_patterns = [k for k, v in matches.items() if v]
        
        return ValidationResult(
            is_valid=bool(matched_patterns),
            score=score,
            reason=f"Found academic patterns: {matched_patterns}" if matched_patterns else "No academic patterns found",
            metadata={"pattern_matches": matches},
            validator_name=self.name
        )

class UrlStructureValidator(BaseValidator):
    """Validates URL structure and patterns."""
    
    BLOCKED_PATTERNS = [
        '/ads/', '/tracking/', '/analytics/', 
        '/sponsored/', '/advertisement/'
    ]
    
    PREFERRED_PATTERNS = [
        '/paper/', '/article/', '/research/',
        '/publication/', '/doc/', '/pdf/'
    ]
    
    async def validate(self, doc: Document) -> ValidationResult:
        url = self._get_url_from_doc(doc)
        if not url:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                reason="No URL provided",
                metadata={},
                validator_name=self.name
            )
            
        parsed = urlparse(url)
        path = parsed.path
        
        # Check blocked patterns first
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in path:
                return ValidationResult(
                    is_valid=False,
                    score=0.0,
                    reason=f"URL contains blocked pattern: {pattern}",
                    metadata={"blocked_pattern": pattern},
                    validator_name=self.name
                )
        
        # Check preferred patterns
        preferred = any(p in path for p in self.PREFERRED_PATTERNS)
        score = 0.8 if preferred else 0.5
        
        return ValidationResult(
            is_valid=True,
            score=score,
            reason="URL structure acceptable" + (" and preferred" if preferred else ""),
            metadata={"preferred": preferred, "url": url},
            validator_name=self.name
        )

class ValidatorPipeline:
    """Pipeline for running multiple validators in sequence."""
    
    def __init__(self, validators: Optional[List[BaseValidator]] = None):
        """
        Initialize the pipeline with optional custom validators.
        
        Args:
            validators: List of validator instances to use
        """
        self.validators = validators or [
            DomainValidator(),
            AcademicValidator(),
            UrlStructureValidator()
        ]
    
    async def run(self, doc: Document) -> Dict[str, Any]:
        """
        Run all validators and aggregate results.
        
        Args:
            doc: Document to validate
            
        Returns:
            Dictionary containing comprehensive validation results
        """
        try:
            results = await asyncio.gather(
                *[validator.validate(doc) for validator in self.validators]
            )
            
            total_score = sum(r.score for r in results)
            avg_score = total_score / len(results)
            is_valid = all(r.is_valid for r in results)
            
            return {
                "is_valid": is_valid,
                "overall_score": avg_score,
                "validator_results": results,
                "reasons": [r.reason for r in results],
                "metadata": {
                    "url": doc.metadata.get("source_url", ""),
                    "validator_scores": {r.validator_name: r.score for r in results}
                }
            }
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {str(e)}")
            raise

class SimpleValidator:
    """Simple document validator using trafilatura for content extraction."""
    
    MIN_WORD_COUNT = 50
    MAX_SCORE_WORDS = 1000  # Number of words for max score
    
    def __init__(self):
        self.extraction_config = trafilatura.extract.DEFAULT_CONFIG
        self.extraction_config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
        self.extraction_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")
    
    async def validate(self, doc: Document) -> Dict[str, Any]:
        """
        Validate document content and structure.
        
        Args:
            doc: Document to validate
            
        Returns:
            Dictionary containing validation results and cleaned content
        """
        url = doc.metadata.get("source_url", "")
        content = doc.page_content
        
        result = {
            "is_valid": False,
            "score": 0.0,
            "reason": "",
            "clean_content": None,
            "metadata": {}
        }
        
        try:
            # Validate URL if present
            if url and not validators.url(url):
                result["reason"] = "Invalid URL format"
                return result
            
            # Extract and clean content
            clean_text = trafilatura.extract(
                content,
                config=self.extraction_config,
                include_comments=False,
                include_tables=True,
                no_fallback=True
            )
            
            if not clean_text:
                result["reason"] = "Failed to extract clean content"
                return result
            
            # Validate content length
            word_count = len(clean_text.split())
            if word_count < self.MIN_WORD_COUNT:
                result["reason"] = f"Content too short (minimum {self.MIN_WORD_COUNT} words)"
                return result
            
            # Calculate quality score
            score = min(1.0, word_count / self.MAX_SCORE_WORDS)
            
            result.update({
                "is_valid": True,
                "score": score,
                "reason": "Content validated successfully",
                "clean_content": clean_text,
                "metadata": {
                    "word_count": word_count,
                    "original_length": len(content),
                    "cleaned_length": len(clean_text)
                }
            })
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            result["reason"] = f"Validation error: {str(e)}"
        
        return result

# Example usage:
# pipeline = ValidatorPipeline()  # Uses default validators
# result = await pipeline.run(doc)
# print(f"Valid: {result['is_valid']}, Score: {result['overall_score']:.2f}")
# for r in result['validator_results']:
#     print(f"{r.validator_name}: {r.reason} (score: {r.score})")

# Example usage:
# validator = SimpleValidator()
# result = await validator.validate(doc)
# if result["is_valid"]:
#     print(f"Valid content (score: {result['score']:.2f})")
#     print(f"Clean content length: {len(result['clean_content'])}")
# else:
#     print(f"Invalid: {result['reason']}")